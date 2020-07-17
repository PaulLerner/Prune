#!/usr/bin/env python
# encoding: utf-8

"""Usage:
named_id.py train <protocol> <experiment_dir> [options] [--from=<epoch>]
named_id.py validate <protocol> <train_dir> [options] [--evergreen --interactive]
named_id.py test <protocol> <validate_dir> [options] [--interactive]

Common options:

--subset=<subset>	 Protocol subset, one of 'train', 'development' or 'test' [default: train]
--batch=<batch>		 Batch size [default: 128]
--window=<window>	 Window size [default: 8]
--step=<step>		 Step size [default: 1]
--max_len=<max_len>	 Maximum # of tokens input to BERT. Maximum 512 [default: 256]
--mask               Compute attention_mask according to max_len.
--easy               Only keep text windows with named speakers in it.
--sep_change         Add a special "[SEP]" token between every speech turn.

Training options:
--from=<epoch>       Start training back from a specific checkpoint (epoch #)
--augment=<ratio>    If greater than 0, will generate `augment` synthetic examples per real example
                     See batchify for details.
                     Defaults to no augmentation.

Validation options:
--evergreen          Start with the latest checkpoints
--interactive        Open-up python debugger after each forward pass

File structure should look like:

<experiment_dir>
└───config.yml
│   <train_dir>
│   └────weights
│   │   └───*.tar
│   │   <validate_dir>
│   │   └───<test_dir>

config.yml is optional to set additional parameters (e.g. change the default model architecture)
It should look like:

architecture:
    nhead: 8
    num_encoder_layers: 6
    dropout: 0.1
training:
    lr: 0.001
    freeze: [bert]
"""

from docopt import docopt
from tqdm import tqdm
from pathlib import Path
import json
import yaml
import warnings

from pyannote.core import Segment
from pyannote.database import get_protocol
import Plumcot as PC

import re
import numpy as np

from torch import save, load, manual_seed, no_grad, argmax, Tensor
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.nn import NLLLoss
from transformers import BertTokenizer
from prune.sidnet import SidNet


# set random seed
np.random.seed(0)
manual_seed(0)

EPOCH_FORMAT = '{:04d}.tar'
BERT = 'bert-base-cased'

# constant paths
DATA_PATH = Path(PC.__file__).parent / 'data'
CHARACTERS_PATH = DATA_PATH.glob('*/characters.txt')


def batch_accuracy(targets, predictions, pad=0):
    """Compute accuracy at the batch level.
    Should work the same with torch and np.
    Ignores padded targets.
    """

    indices = targets != pad
    where = predictions[indices] == targets[indices]

    # switch between torch and np
    if isinstance(where, Tensor):
        where = where.nonzero(as_tuple=True)
        total = (~indices).nonzero(as_tuple=True)
    else:
        where = where.nonzero()
        total = (~indices).nonzero()

    batch_acc = where[0].shape[0] / total[0].shape[0]

    return batch_acc


def eval(batches, model, tokenizer, validate_dir,
         test=False, evergreen=False, interactive=False):
    """Load model from checkpoint and evaluate it on batches.
    When testing, only the best model should be tested.

    Parameters
    ----------
    batches: List[Tuple[Tensor]]:
        (input_ids, target_ids, audio_similarity, src_key_padding_mask, tgt_key_padding_mask)
        see batch_encode_multi
    model: SidNet
        instance of SidNet, ready to be loaded
    tokenizer: BertTokenizer
        used to decode output (i.e. de-tensorize, de-tokenize)
    validate_dir: Path
        Path to log validation accuracy and load model weights (from ../weights)
        Defaults to current working directory.
    test: bool, optional
        Whether to test only the best model.
        Defaults to False.
    evergreen: bool, optional
        Whether to start validation with the latest checkpoints.
        Defaults to False.
    interactive: bool, optional
        Opens-up python debugger after each forward pass.
        Defaults to False.
    """
    if test:
        raise NotImplementedError('test')
    weights_path = validate_dir.parent / 'weights'
    if not weights_path.exists():
        raise ValueError(f'Weights path "{weights_path}" does not exist.')

    criterion = NLLLoss(ignore_index=tokenizer.pad_token_id)
    tb = SummaryWriter(validate_dir)
    weights = sorted(weights_path.iterdir(), reverse=evergreen)
    for weight in tqdm(weights, desc='Evaluating'):
        checkpoint = load(weight)
        epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        with no_grad():
            epoch_loss, epoch_token_acc, epoch_word_acc = 0., 0., 0.
            for input_ids, target_ids, audio_similarity, src_key_padding_mask, tgt_key_padding_mask in batches:
                # forward pass
                output = model(input_ids, target_ids, audio_similarity,
                               src_key_padding_mask, tgt_key_padding_mask)
                # manage devices
                target_ids = target_ids.to(output.device)

                # get model prediction per token
                predictions = argmax(output, dim=2)

                # compute batch-token accuracy
                epoch_token_acc += batch_accuracy(target_ids, predictions, tokenizer.pad_token_id)

                # decode and compute word accuracy
                decoded_targets = batch2numpy(tokenizer, target_ids)
                decoded_predictions = batch2numpy(tokenizer, predictions)
                epoch_word_acc += batch_accuracy(decoded_targets, decoded_predictions, tokenizer.pad_token)

                if interactive:
                    breakpoint()

                # TODO fuse batch output at the document level and compute accuracy

                # compute loss
                #   reshape output like (batch_size * sequence_length, vocab_size)
                #   and target_ids like (batch_size * sequence_length)
                output = output.reshape(-1, model.vocab_size)
                target_ids = target_ids.reshape(-1)
                loss = criterion(output, target_ids)
                epoch_loss += loss.item()

            tb.add_scalar('Loss/eval', epoch_loss / len(batches), epoch)
            tb.add_scalar('Accuracy/eval/batch/token', epoch_token_acc / len(batches), epoch)
            tb.add_scalar('Accuracy/eval/batch/word', epoch_word_acc / len(batches), epoch)


def train(batches, model, tokenizer, train_dir=Path.cwd(),
          audio=None, lr=1e-3, max_grad_norm=None,
          epochs=100, freeze=['bert'], save_every=1, start_epoch=None):
    """Train the model for `epochs` epochs

    Parameters
    ----------
    batches: List[Tuple[Tensor]]
        (input_ids, target_ids, audio_similarity, src_key_padding_mask, tgt_key_padding_mask)
        see batch_encode_multi
    model: SidNet
        instance of SidNet, ready to be trained
    tokenizer: BertTokenizer
        used to get tokenization constants (e.g. tokenizer.pad_token_id == 0)
    train_dir: Path, optional
        Path to log training loss and save model weights (under experiment_path/weights)
        Defaults to current working directory.
    audio: `Wrappable`, optional
        Describes how raw speaker embeddings should be obtained.
        See pyannote.audio.features.wrapper.Wrapper documentation for details.
        Defaults to None, indicating that the model should rely only on the text.
    lr: float, optional
        Learning rate used to optimize model parameters.
        Defaults to 1e-3
    max_grad_norm: float, optional
        Clips gradient L2 norm at max_grad_norm
        Defaults to no clipping.
    epochs: int, optional
        Train the model for `epochs` epochs.
        Defaults to 100
    freeze : List[str], optional
        Names of modules to freeze.
        Defaults to freezing bert (['bert']).
    save_every: int, optional
        Save model weights and optimizer state every `save_every` epoch.
        Defaults to save at every epoch (1)
    start_epoch: int, optional
        Starts training back at start_epoch.
        Defaults to raise an error if training in an existing directory
    """
    optimizer = Adam(model.parameters(), lr=lr)

    weights_path = train_dir / 'weights'
    # load previous checkpoint
    if start_epoch is not None:
        checkpoint = load(weights_path / EPOCH_FORMAT.format(start_epoch))
        assert start_epoch == checkpoint["epoch"]
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # increment epoch
        start_epoch += 1
    else:
        # be careful not to erase previous weights
        weights_path.mkdir(exist_ok=False)
        # defaults to start from 0
        start_epoch = 0

    model.freeze(freeze)
    model.train()

    criterion = NLLLoss(ignore_index=tokenizer.pad_token_id)

    tb = SummaryWriter(train_dir)
    for epoch in tqdm(range(start_epoch, epochs+start_epoch), desc='Training'):
        # shuffle batches
        np.random.shuffle(batches)

        epoch_loss = 0.
        for input_ids, target_ids, audio_similarity, src_key_padding_mask, tgt_key_padding_mask in batches:
            optimizer.zero_grad()

            # forward pass
            output = model(input_ids, target_ids, audio_similarity,
                           src_key_padding_mask, tgt_key_padding_mask)
            # reshape output like (batch_size * sequence_length, vocab_size)
            # and target_ids like (batch_size * sequence_length)
            # and manage devices
            output = output.reshape(-1, model.vocab_size)
            target_ids = target_ids.reshape(-1).to(output.device)

            # calculate loss
            loss = criterion(output, target_ids)
            loss.backward()
            if max_grad_norm is not None:
                clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            epoch_loss += loss.item()

        tb.add_scalar('Loss/train', epoch_loss/len(batches), epoch)

        if epoch % save_every == 0:
            save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss
            }, weights_path / EPOCH_FORMAT.format(epoch))

    return model, optimizer


def batch2numpy(tokenizer, batch):
    """Decode batch to list of string using tokenizer
    then reshape the list of str to a np array of tokens"""

    decoded = tokenizer.batch_decode(batch)
    decoded_list = []
    for line in decoded:
        # trim sequence to max length
        line = line.split()[:max_length]
        # pad sequence to max_length
        line += [tokenizer.pad_token] * (max_length - len(line))
        decoded_list.append(line)
    return np.array(decoded_list, dtype=str)


def any_in_text(items, text):
    """Utility function.
    Returns True if any of the item in items is in text, False otherwise.
    """

    for item in items:
        if item in text:
            return True
    return False


def batchify(tokenizer, protocol, mapping, subset='train',
             batch_size=128, window_size=10, step_size=1,
             mask=True, easy=False, sep_change=False, augment=0):
    """
    Iterates over protocol subset, segment transcription in speaker turns,
    Divide transcription in windows then split windows in batches.
    And finally, encode batch (i.e. tokenize, tensorize...)

    Parameters
    ----------
    tokenizer: BertTokenizer
        used to tokenize, pad and tensorize text
    protocol: Protocol
        pyannote Protocol to get transcription from
    mapping: dict
        used to convert normalized speaker names into its most common name.
        Note that it's important that this name is as written in the input text.
    subset: str, optional
        subset of the protocol to get transcription from
        Defaults to training set.
    batch_size: int, optional
        Defaults to 128.
    window_size: int, optional
        Number of speaker turns in one window
        Defaults to 10.
    step_size: int, optional
        Defaults to 1.
    mask: bool, optional
        Compute attention_mask according to max_length.
        Defaults to True.
    easy: bool, optional
        Only keep windows with named speakers in it
        (the name must match one of the labels as provided in mapping)
        Defaults to keep every window regardless of it's content.
    sep_change: bool, optional
        Add special token tokenizer.sep_token ("[SEP]") between every speech turn.
        Defaults to keep input as is.
    augment: int, optional
        Data augmentation ratio.
        If greater than 0, will generate `augment` synthetic examples per real example
        by replacing speaker names in input text and target by a random name.
        Note that it doesn't have any effect if no speaker names (as provided in mapping)
        are present in the input text.
        Defaults to no augmentation.
    Returns
    -------
    batches: List[Tuple[Tensor]]:
        (input_ids, target_ids, audio_similarity, src_key_padding_mask, tgt_key_padding_mask)
        see batch_encode_multi
    """

    with open(mapping) as file:
        mapping = json.load(file)

    # load list of names
    if augment > 0:
        names = []
        for character_file in CHARACTERS_PATH:
            with open(character_file) as file:
                names += [line.split(',')[3].split()[0]
                          for line in file.read().split("\n") if line != '']

    batches = []
    text_windows, audio_windows, target_windows = [], [], []

    # iterate over protocol subset
    for current_file in tqdm(getattr(protocol, subset)(), desc='Loading transcriptions'):
        transcription = current_file['transcription']

        # format transcription into 3 lists: tokens, audio, targets
        # and segment it in speaker turns (i.e. speaker homogeneous)
        windows = []
        start, end = 0, 0
        tokens, audio, targets = [], [], []
        previous_speaker = None
        for token in transcription:
            if token._.speaker != previous_speaker:
                # mark speaker change with special token tokenizer.sep_token ("[SEP]")
                if sep_change:
                    tokens.append(tokenizer.sep_token)
                    targets.append(tokenizer.pad_token)
                    audio.append(tokenizer.pad_token)
                    end += 1
                windows.append((start, end))
                start = end
            tokens.append(token.text)
            # if audio alignment is not confident for token
            # then audio similarity matrix of token should be uniform
            # so it doesn't weigh on the text decision
            if token._.confidence > 0.5 and '#unknown#' not in token._.speaker:
                audio.append(Segment(token._.time_start, token._.time_end))
            else:
                audio.append(tokenizer.pad_token)
            # if we don't have a proper target we should mask the loss function
            targets.append(mapping.get(token._.speaker, tokenizer.pad_token))
            previous_speaker = token._.speaker
            end += 1
        windows.pop(0)

        # slide through the transcription speaker turns w.r.t. window_size, step_size
        # filter out windows w.r.t. easy
        # and augment them w.t.t. augment
        for i in range(0, len(windows) - window_size, step_size):
            start, _ = windows[i]
            _, end = windows[i + window_size - 1]
            text_window = " ".join(tokens[start:end])
            target_window = " ".join(targets[start:end])

            # set of actual targets (i.e. excluding [PAD], [SEP], etc.)
            target_set = set(targets[start:end]) - set(tokenizer.all_special_tokens)

            # easy mode -> Only keep windows with named speakers in it
            if easy and not any_in_text(target_set, text_window):
                continue

            text_windows.append(text_window)
            audio_windows.append(audio[start:end])
            target_windows.append(target_window)

            # add `augment` windows of synthetic data
            for augmentation in range(augment):
                synthetic_text = text_window
                synthetic_targets = target_window
                # augment data by replacing
                # speaker names in input text and target by a random name
                for target in target_set:
                    # except if the name is not present in the input text
                    # this would only add noise
                    if not re.search(target, text_window, flags=re.IGNORECASE):
                        continue
                    random_name = np.random.choice(names)
                    synthetic_text = re.sub(fr'\b{target}\b', random_name,
                                            synthetic_text, flags=re.IGNORECASE)
                    synthetic_targets = re.sub(fr'\b{target}\b', random_name,
                                               synthetic_targets, flags=re.IGNORECASE)
                audio_windows.append(audio[start:end])
                text_windows.append(synthetic_text)
                target_windows.append(synthetic_targets)

    # shuffle all windows
    indices = np.arange(len(text_windows))
    np.random.shuffle(indices)

    # split windows in batches w.r.t. batch_size 
    for i in tqdm(range(0, len(indices) - batch_size, batch_size), desc='Encoding batches'):
        text_batch, target_batch, audio_batch = [], [], None
        for j in indices[i: i + batch_size]:
            text_batch.append(text_windows[j])
            target_batch.append(target_windows[j])
            # TODO integrate audio
            # audio_batch.append(audio_windows[j])
        # encode batch (i.e. tokenize, tensorize...)
        batch = batch_encode_multi(tokenizer, text_batch, target_batch, audio_batch, mask=mask)
        batches.append(batch)
    return batches


def batch_encode_plus(tokenizer, text_batch, mask=True):
    """Shortcut function to encode a text (either input or target) batch
    using tokenizer.batch_encode_plus with the appropriate parameters.

    Parameters
    ----------
    tokenizer: BertTokenizer
    text_batch: List[str]
    mask: bool, optional
        Compute attention_mask according to max_length.
        Defaults to True.

    Returns
    -------
    input_ids: Tensor
        (batch_size, max_length). Encoded input tokens using BertTokenizer
    attention_mask: Tensor
        (batch_size, max_length). Used to mask input_ids.
        None if not mask.
    """
    text_encoded_plus = tokenizer.batch_encode_plus(text_batch,
                                                    add_special_tokens=False,
                                                    max_length=max_length,
                                                    pad_to_max_length='right',
                                                    return_tensors='pt',
                                                    return_attention_mask=mask)
    input_ids = text_encoded_plus['input_ids']
    attention_mask = text_encoded_plus['attention_mask'] if mask else None
    return input_ids, attention_mask


def batch_encode_multi(tokenizer, text_batch, target_batch, audio_batch=None, mask=True):
    """Encode input, target text and audio consistently in torch Tensor

    Parameters
    ----------
    tokenizer: BertTokenizer
        used to tokenize, pad and tensorize text
    text_batch: List[str]
        (batch_size, ) Input text
    target_batch: List[str]
        (batch_size, ) Target speaker name, aligned with text_batch
    audio_batch: List[Segment], optional
        (batch_size, ) Timestamps of the input text, aligned with text_batch[i].split(' ')
        Defaults to None (model only relies on the text).
    mask: bool, optional
        Compute attention_mask according to max_length.
        Defaults to True.

    Returns
    -------
    input_ids: Tensor
            (batch_size, max_length). Encoded input tokens using BertTokenizer
    target_ids: Tensor
        (batch_size, max_length). Encoded target tokens using BertTokenizer
    audio_similarity: Tensor, optional
        (batch_size, max_length, max_length). Similarity (e.g. cosine distance)
        between audio embeddings of words, aligned with target_ids.
        Defaults to None, indicating that the model should rely only on the text.
    src_key_padding_mask: Tensor, optional
        (batch_size, max_length). Used to mask input_ids.
        Defaults to None (no masking).
    tgt_key_padding_mask: Tensor, optional
        (batch_size, max_length). Used to mask target_ids.
        Defaults to None (no masking).
    """
    if audio_batch is not None:
        raise NotImplementedError("audio_batch")
        # TODO: compute audio similarity matrix
        # note that it must be aligned with bert tokenization
    else:
        audio_similarity = None

    # encode input text
    input_ids, src_key_padding_mask = batch_encode_plus(tokenizer, text_batch, mask=mask)

    # encode target text
    target_ids, tgt_key_padding_mask = batch_encode_plus(tokenizer, target_batch, mask=mask)

    return input_ids, target_ids, audio_similarity, src_key_padding_mask, tgt_key_padding_mask


def load_config(parent_path):
    """Returns empty dict if unable to load config file"""
    config_path = parent_path / 'config.yml'
    if not config_path.is_file():
        return dict()

    with open(config_path) as file:
        return yaml.load(file, Loader=yaml.SafeLoader)


if __name__ == '__main__':
    # parse arguments and get protocol
    args = docopt(__doc__)
    protocol_name = args['<protocol>']
    subset = args['--subset'] if args['--subset'] else 'train'
    batch_size = int(args['--batch']) if args['--batch'] else 128
    window_size = int(args['--window']) if args['--window'] else 8
    step_size = int(args['--step']) if args['--step'] else 1
    max_length = int(args['--max_len']) if args['--max_len'] else 256
    mask = args['--mask']
    easy = args['--easy']
    sep_change = args['--sep_change']
    augment = int(args['--augment']) if args['--augment'] else 0
    protocol = get_protocol(protocol_name)
    serie, _, _ = protocol_name.split('.')
    full_name = f'{protocol_name}.{subset}'
    mapping = DATA_PATH / serie / 'annotated_transcripts' / 'names_dict.json'

    # instantiate tokenizer
    tokenizer = BertTokenizer.from_pretrained(BERT)

    # get batches from protocol subset
    batches = batchify(tokenizer, protocol, mapping, subset,
                       batch_size=batch_size,
                       window_size=window_size,
                       step_size=step_size,
                       mask=mask,
                       easy=easy,
                       sep_change=sep_change,
                       augment=augment)

    if args['train']:
        start_epoch = int(args['--from']) if args['--from'] else None
        train_dir = Path(args['<experiment_dir>'], full_name)
        train_dir.mkdir(exist_ok=True)
        config = load_config(train_dir.parents[0])

        model = SidNet(BERT, tokenizer.vocab_size, **config.get('architecture', {}))
        model, optimizer = train(batches, model, tokenizer, train_dir,
                                 start_epoch=start_epoch,
                                 **config.get('training', {}))

    elif args['validate']:
        evergreen = args['--evergreen']
        interactive = args['--interactive']
        validate_dir = Path(args['<train_dir>'], full_name)
        validate_dir.mkdir(exist_ok=True)
        config = load_config(validate_dir.parents[1])

        model = SidNet(BERT, tokenizer.vocab_size, **config.get('architecture', {}))
        eval(batches, model, tokenizer, validate_dir,
             test=False, evergreen=evergreen, interactive=interactive)

    elif args['test']:
        interactive = args['--interactive']
        test_dir = Path(args['<validate_dir>'], full_name)
        test_dir.mkdir(exist_ok=True)
        config = load_config(test_dir.parents[2])

        model = SidNet(BERT, tokenizer.vocab_size, **config.get('architecture', {}))
        eval(batches, model, tokenizer, test_dir, test=True, interactive=interactive)


