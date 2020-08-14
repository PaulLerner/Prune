#!/usr/bin/env python
# encoding: utf-8

"""Usage:
named_id.py train <protocol> <experiment_dir> [options] [--from=<epoch>]
named_id.py validate <protocol> <train_dir> [options] [--evergreen --interactive]
named_id.py test <protocol> <validate_dir> [options] [--interactive]

Common options:

--subset=<subset>    Protocol subset, one of 'train', 'development' or 'test'.
                     Defaults to 'train', 'development' and 'test' in
                     'train', 'validate', and 'test' mode, respectively.
--batch=<batch>      Batch size [default: 128]
--window=<window>    Window size [default: 8]
--step=<step>        Step size [default: 1]
--max_len=<max_len>  Maximum # of tokens input to BERT. Maximum 512 [default: 256]
--easy               Only keep text windows with named speakers in it.
--sep_change         Add a special "[SEP]" token between every speech turn.

Training options:
--from=<epoch>       Start training back from a specific checkpoint (epoch #)
--augment=<ratio>    If different from 0, will generate `|augment|` synthetic examples per real example
                     If less than 0, will discard real example.
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
│   │   └───params.yml
│   │   │   <test_dir>
│   │   │   └───params.yml

config.yml is optional to set additional parameters (e.g. change the default model architecture)
It should look like:

architecture:
    nhead: 8
    num_layers: 6
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
from typing import List
from tabulate import tabulate
from itertools import zip_longest

from pyannote.core import Segment
from pyannote.core.utils.distance import pdist
from pyannote.database import get_protocol
from pyannote.audio.features.wrapper import Wrapper, Wrappable
import Plumcot as PC

import re
import numpy as np
from scipy.spatial.distance import squareform
from matplotlib import pyplot as plt

from torch import save, load, manual_seed, no_grad, argmax, Tensor, zeros, from_numpy, zeros_like
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.nn import BCELoss
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


def token_accuracy(targets: Tensor, predictions: Tensor, pad: int=0):
    """Compute accuracy at the token level.
    Ignores padded targets.
    """

    indices = targets != pad
    where = (predictions[indices] == targets[indices]).nonzero(as_tuple=True)
    return where[0].shape[0] / indices.nonzero(as_tuple=True)[0].shape[0]


def batch_word_accuracy(targets: List[str], predictions: List[str], pad='[PAD]'):
    correct, total = 0, 0
    for target, prediction in zip(targets, predictions):
        target, prediction = target.split(), prediction.split()
        for t, p in zip_longest(target, prediction, fillvalue=pad):
            if t == pad:
                continue
            if t == p:
                correct += 1
            total += 1
    return correct/total


def str_example(inp_eg, tgt_eg, pred_eg, step=20):
    example = []
    for i in range(0, len(inp_eg) - step, step):
        tab = tabulate((['inp:'] + inp_eg[i:i + step],
                        ['tgt:'] + tgt_eg[i:i + step],
                        ['hyp:'] + pred_eg[i:i + step])).split('\n')
        example += tab[1:]
    return '\n'.join(example)


def plot_output(output_eg, inp_eg, tgt_eg, save=None):
    # merge target and input into a single list
    tgt_eg = [f"{i} ({t})" for i, t in zip(inp_eg, tgt_eg)]
    plt.figure(figsize=(20, 20))
    max_len = len(inp_eg)
    # shift by 1 to discard [CLS] and [SEP] tokens
    plt.imshow(output_eg.detach().cpu().numpy()[:max_len, 1: max_len-1])
    plt.colorbar()
    plt.xticks(range(max_len), inp_eg[:max_len], fontsize='x-small', rotation='vertical')
    plt.yticks(range(max_len), tgt_eg[:max_len], fontsize='x-small', rotation='horizontal')
    if save is None:
        plt.show()
    else:
        plt.savefig(save/"output.png")


def eval(batches, model, tokenizer, log_dir,
         test=False, evergreen=False, interactive=False, step_size=1, window_size=10):
    """Load model from checkpoint and evaluate it on batches.
    When testing, only the best model should be tested.

    Parameters
    ----------
    batches: List[Tuple[Tensor]]:
        (text_batch, target_batch, input_ids, target_ids, audio_similarity, src_key_padding_mask, tgt_key_padding_mask)
        see batch_encode_multi
    model: SidNet
        instance of SidNet, ready to be loaded
    tokenizer: BertTokenizer
        used to decode output (i.e. de-tensorize, de-tokenize)
    log_dir: Path
        either:
        - [!test]: Path to log validation accuracy and load model weights (from ../weights)
        - [test]: Path to log test accuracy, load model weights (from ../../weights)
                  and load best epoch (from ../params.yml)
    test: bool, optional
        Whether to test only the best model.
        Defaults to False.
    evergreen: bool, optional
        Whether to start validation with the latest checkpoints.
        Defaults to False.
    interactive: bool, optional
        Opens-up python debugger after each forward pass.
        Defaults to False.
    step_size: int, optional
        Overlap between two subsequent text-windows (i.e. item in batch)
        Defaults to 1.
    window_size: int, optional
        Number of speaker turns in one window
        Defaults to 10.
    """
    if test:
        weights_path = log_dir.parents[1] / 'weights'
        with open(log_dir.parent / 'params.yml') as file:
            epoch = yaml.load(file, Loader=yaml.SafeLoader)["epoch"]
        weights = [weights_path/EPOCH_FORMAT.format(epoch)]
    else:
        weights_path = log_dir.parents[0] / 'weights'
        weights = sorted(weights_path.iterdir(), reverse=evergreen)

    criterion = BCELoss(reduction='none')
    tb = SummaryWriter(log_dir)
    best = 0.
    for weight in tqdm(weights, desc='Evaluating'):
        checkpoint = load(weight)
        epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        with no_grad():
            epoch_loss, epoch_word_acc = 0., 0.
            uris, file_token_acc, file_word_acc = [], [], []
            previous_uri = None
            for uri, windows, inp, tgt, input_ids, target_ids, audio_similarity, src_key_padding_mask, tgt_key_padding_mask in batches:
                # forward pass: (batch_size, sequence_length, sequence_length)
                output = model(input_ids, audio_similarity, src_key_padding_mask)
                # manage devices
                target_ids = target_ids.to(output.device)

                # TODO handle file-level stuff
                if False and uri != previous_uri:
                    # compute file-level accuracy
                    if previous_uri is not None:
                        uris.append(previous_uri)
                        # TODO use torch.mode on prediction_ids rather than argmax on scores
                        file_pred_ids = argmax(file_output, dim=1)

                        # compute token accuracy
                        file_token_acc.append(token_accuracy(file_target_ids,
                                                             file_pred_ids,
                                                             tokenizer.pad_token_id))

                        # decode and compute word accuracy
                        file_target = tokenizer.batch_decode(file_target_ids.unsqueeze(0),
                                                             clean_up_tokenization_spaces=False)
                        file_pred = tokenizer.batch_decode(file_pred_ids.unsqueeze(0),
                                                           clean_up_tokenization_spaces=False)
                        file_word_acc.append(batch_word_accuracy(file_target, file_pred,
                                                                 pad=tokenizer.pad_token))
                        # TODO audio ER

                    # reset file-level variables
                    file_length = windows[-1][-1] - windows[0][0]
                    i, shift = 0, 0
                    file_output = zeros((file_length, model.vocab_size),
                                        dtype=output.dtype,
                                        device=output.device)
                    file_target_ids = zeros((file_length,),
                                            dtype=target_ids.dtype,
                                            device=target_ids.device)

                # TODO save target and output for future file-level accuracy
                for t, o in zip(target_ids, output):
                    break
                    for start, end in windows[i: i+window_size]:
                        # trim to max_length
                        shifted_start, shifted_end = min(start-shift, max_length), min(end-shift, max_length)
                        # HACK: this doesn't hold only at the end of TheBigBangTheory.Season03.Episode05 for some reason
                        if len(file_target_ids[shifted_start+shift: shifted_end+shift]) == len(t[shifted_start: shifted_end]):
                            file_target_ids[shifted_start+shift: shifted_end+shift] = t[shifted_start: shifted_end]
                            file_output[shifted_start+shift: shifted_end+shift] += o[shifted_start: shifted_end]
                        else:
                            continue
                    i += step_size
                    # shift between batch and original file
                    shift = windows[i][0]#start
                # get model prediction per token: (batch_size, sequence_length)
                relative_out = argmax(output, dim=2)
                # retrieve token ids from input (batch_size, sequence_length) and manage device
                prediction_ids = zeros_like(input_ids, device=output.device)
                for j, (input_window_id, relative_window_out) in enumerate(zip(input_ids, relative_out)):
                    prediction_ids[j] = input_window_id[relative_window_out]

                # decode and compute word accuracy
                predictions = tokenizer.batch_decode(prediction_ids, clean_up_tokenization_spaces=False)
                epoch_word_acc += batch_word_accuracy(tgt, predictions, tokenizer.pad_token)

                # calculate loss
                loss = criterion(output, target_ids)
                loss = reduce_loss(loss, tgt_key_padding_mask)
                epoch_loss += loss.item()
                previous_uri = uri

                if interactive:
                    eg = np.random.randint(len(tgt))
                    inp_eg, tgt_eg, pred_eg = inp[eg], tgt[eg], predictions[eg]
                    # print random example
                    print(str_example(inp_eg.split(), tgt_eg.split(), pred_eg.split()))
                    # plot model output
                    plot_output(output[eg], tokenizer.tokenize(inp_eg), 
                                tokenizer.tokenize(tgt_eg), log_dir)

                    # print current metrics
                    metrics = {
                        'Loss/eval': [epoch_loss],
                        'Accuracy/eval/batch/word': [epoch_word_acc]
                    }
                    print(tabulate(metrics, headers='keys'))
                    breakpoint()

            # TODO compute file-level accuracy for the last file
            # uris.append(previous_uri)
            # file_pred_ids = argmax(file_output, dim=1)
            # file_token_acc.append(token_accuracy(file_target_ids,
            #                                      file_pred_ids,
            #                                      tokenizer.pad_token_id))
            # file_target = tokenizer.batch_decode(file_target_ids.unsqueeze(0),
            #                                      clean_up_tokenization_spaces=False)
            # file_pred = tokenizer.batch_decode(file_pred_ids.unsqueeze(0),
            #                                    clean_up_tokenization_spaces=False)
            # file_word_acc.append(batch_word_accuracy(file_target, file_pred,
            #                                          pad=tokenizer.pad_token))
            #
            # # average file-accuracies
            # uris.append('TOTAL')
            # file_token_acc.append(np.mean(file_token_acc))
            # file_word_acc.append(np.mean(file_word_acc))
            # # print file-accuracies
            # if test:
            #     results = tabulate(zip(uris, file_token_acc, file_word_acc),
            #                        headers=['uri', 'token-level', 'word-level'],
            #                        tablefmt='latex')
            #     print(f'Epoch #{epoch} | Accuracies per file:\n{results}')

            # log tensorboard
            # tb.add_scalar('Accuracy/eval/file/token', file_token_acc[-1], epoch)
            # tb.add_scalar('Accuracy/eval/file/word', file_word_acc[-1], epoch)
            tb.add_scalar('Loss/eval', epoch_loss / len(batches), epoch)
            epoch_word_acc /= len(batches)
            tb.add_scalar('Accuracy/eval/batch/word', epoch_word_acc, epoch)
            if epoch_word_acc > best:
                best = epoch_word_acc
                with open(log_dir / 'params.yml', 'w') as file:
                    yaml.dump({"accuracy": best, "epoch": epoch}, file)


def reduce_loss(loss, tgt_key_padding_mask):
    """Masks loss using tgt_key_padding_mask then mean-reduce"""
    # mask and average loss
    return loss[tgt_key_padding_mask.bool()].mean()


def train(batches, model, tokenizer, train_dir=Path.cwd(),
          lr=1e-3, max_grad_norm=None,
          epochs=100, freeze=['bert'], save_every=1, start_epoch=None):
    """Train the model for `epochs` epochs

    Parameters
    ----------
    batches: List[Tuple[Tensor]]
        (text_batch, target_batch, input_ids, target_ids, audio_similarity, src_key_padding_mask, tgt_key_padding_mask)
        see batch_encode_multi
    model: SidNet
        instance of SidNet, ready to be trained
    tokenizer: BertTokenizer
        used to get tokenization constants (e.g. tokenizer.pad_token_id == 0)
    train_dir: Path, optional
        Path to log training loss and save model weights (under experiment_path/weights)
        Defaults to current working directory.
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

    criterion = BCELoss(reduction='none')

    tb = SummaryWriter(train_dir)
    for epoch in tqdm(range(start_epoch, epochs+start_epoch), desc='Training'):
        # shuffle batches
        np.random.shuffle(batches)

        epoch_loss = 0.
        for _, _, _, _, input_ids, target_ids, audio_similarity, src_key_padding_mask, tgt_key_padding_mask in batches:
            optimizer.zero_grad()

            # forward pass
            output = model(input_ids, audio_similarity, src_key_padding_mask)
            # manage devices
            target_ids = target_ids.to(output.device)

            # calculate loss
            loss = criterion(output, target_ids)
            # mask and reduce loss
            loss = reduce_loss(loss, tgt_key_padding_mask)
            loss.backward()

            if max_grad_norm is not None:
                clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            epoch_loss += loss.item()

        tb.add_scalar('Loss/train', epoch_loss/len(batches), epoch)
        tb.add_scalar('lr', lr, epoch)

        if (epoch+1) % save_every == 0:
            save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss
            }, weights_path / EPOCH_FORMAT.format(epoch))

    return model, optimizer


def any_in_text(items, text):
    """Utility function.
    Returns True if any of the item in items is in text, False otherwise.
    """

    for item in items:
        if item in text:
            return True
    return False


def batchify(tokenizer, protocol, mapping, subset='train', audio_emb=None,
             batch_size=128, window_size=10, step_size=1,
             mask=True, easy=False, sep_change=False, augment=0, shuffle=True):
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
    audio_emb: `Wrappable`, optional
        Describes how raw speaker embeddings should be obtained.
        See pyannote.audio.features.wrapper.Wrapper documentation for details.
        Defaults to None, indicating that the model should rely only on the text.
    batch_size: int, optional
        Remainder batch (of size <= batch_size) is kept.
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
        If different from 0, will generate `|augment|` synthetic examples per real example
        by replacing speaker names in input text and target by a random name.
        Note that it doesn't have any effect if no speaker names (as provided in mapping)
        are present in the input text. If less than 0, will discard real example.
        Defaults to no augmentation.
    shuffle: bool, optional
        Whether to shuffle windows before batching.
        Should be set to False when testing to get file-homogeneous batches,
        and to True while training to ensure stochasticity.
        Defaults to True

    Yields
    -------
    batch: Tuple[str, List[Tuple[int]], List[str], Tensor]:
        (uri, windows, text_batch, target_batch, input_ids, target_ids, audio_similarity, src_key_padding_mask, tgt_key_padding_mask)
        - see batch_encode_multi.
        - uri: str,
          file-identifier of the batch and is set to None if shuffle, as batch
          are then file-heterogeneous
        - batch_windows: List[Tuple[int]]
          indices of the start and end tokens index of the speaker turns in the batch
          Empty if shuffling or augmenting data
    """
    assert not tokenizer.do_basic_tokenize, "Basic tokenization is handle beforehand"
    with open(mapping) as file:
        mapping = json.load(file)

    if audio_emb is not None:
        audio_emb = Wrapper(audio_emb)

    # load list of names
    if augment != 0:
        names = []
        for character_file in CHARACTERS_PATH:
            with open(character_file) as file:
                names += [line.split(',')[3].split()[0]
                          for line in file.read().split("\n") if line != '']
        names = np.array(names)

    text_windows, audio_windows, target_windows, audio_masks = [], [], [], []

    # iterate over protocol subset
    for current_file in tqdm(getattr(protocol, subset)(), desc='Loading transcriptions'):
        if not shuffle:
            batch_windows, text_windows, audio_windows, target_windows, audio_masks = [], [], [], [], []
        transcription = current_file['transcription']
        uri = current_file['uri']

        current_audio_emb = None
        # get audio embeddings from current_file
        if audio_emb is not None:
            current_audio_emb = audio_emb(current_file)

        # format transcription into 3 lists: tokens, audio, targets
        # and segment it in speaker turns (i.e. speaker homogeneous)
        windows = []
        start, end = 0, 0
        tokens, audio, targets = [], [], []
        previous_speaker = None
        for word in transcription:
            if word._.speaker != previous_speaker:
                # mark speaker change with special token tokenizer.sep_token ("[SEP]")
                if sep_change:
                    tokens.append(tokenizer.sep_token)
                    targets.append(tokenizer.pad_token)
                    audio.append(None)
                    end += 1
                windows.append((start, end))
                start = end

            # get audio embedding for word if alignment timing is confident enough
            if audio_emb is not None and word._.confidence > 0.5:
                segment = Segment(word._.time_start, word._.time_end)
                segment = current_audio_emb.crop(segment, mode="loose")
                # skip segments so small we don't have any embedding for it
                if len(segment) < 1:
                    segment = None
                # average audio-embedding over the segment frames
                else:
                    segment = np.mean(segment, axis=0)
            else:
                segment = None
            # if we don't have a proper target we should mask the loss function
            target = mapping.get(word._.speaker, tokenizer.pad_token)

            # handle basic tokenization (e.g. punctuation) before Word-Piece
            # in order to align input text and speakers
            for token in tokenizer.basic_tokenizer.tokenize(word.text):
                tokens.append(token)
                targets.append(target)
                audio.append(segment)
                end += 1
            previous_speaker = word._.speaker
        windows.append((start, end))
        windows.pop(0)

        # compute token windows in the batch
        # except if augmenting data or in easy mode (FIXME)
        if not shuffle and augment == 0 and not easy:
            for start, end in windows:
                end = len(tokenizer.tokenize(" ".join(targets[start:end])))
                batch_windows.append((start, start+end))

        # slide through the transcription speaker turns w.r.t. window_size, step_size
        # filter out windows w.r.t. easy
        # and augment them w.t.t. augment
        for i in range(0, len(windows) - window_size + 1, step_size):
            start, _ = windows[i]
            _, end = windows[i + window_size - 1]
            text_window = " ".join(tokens[start:end])
            target_window = " ".join(targets[start:end])
            audio_window, audio_mask = align_audio_targets(tokenizer,
                                                           audio[start:end],
                                                           target_window,
                                                           audio_emb)
            # set of actual targets (i.e. excluding [PAD], [SEP], etc.)
            target_set = sorted(set(targets[start:end]) - set(tokenizer.all_special_tokens))

            # easy mode -> Only keep windows with named speakers in it
            if easy and not any_in_text(target_set, text_window):
                continue

            # augment < 0 (=) discard real example
            if augment >= 0:
                text_windows.append(text_window)
                audio_windows.append(audio_window)
                target_windows.append(target_window)
                audio_masks.append(audio_mask)

            # add `augment` windows of synthetic data
            for augmentation in range(abs(augment)):
                synthetic_text = text_window
                synthetic_targets = target_window
                # augment data by replacing
                # speaker names in input text and target by a random name
                for target in target_set:
                    # except if the name is not present in the input text
                    # this would only add noise
                    # TODO make this optional
                    if False and not re.search(target, text_window, flags=re.IGNORECASE):
                        continue
                    random_name = np.random.choice(names)
                    synthetic_text = re.sub(fr'\b{target}\b', random_name,
                                            synthetic_text, flags=re.IGNORECASE)
                    synthetic_targets = re.sub(fr'\b{target}\b', random_name,
                                               synthetic_targets, flags=re.IGNORECASE)
                audio_window, audio_mask = align_audio_targets(tokenizer,
                                                               audio[start:end],
                                                               synthetic_targets,
                                                               audio_emb)
                audio_windows.append(audio_window)
                audio_masks.append(audio_mask)
                text_windows.append(synthetic_text)
                target_windows.append(synthetic_targets)
        # yield file-homogeneous batches along with file-uri
        if not shuffle:
            indices = np.arange(len(text_windows))
            for batch in batchify_windows(tokenizer, text_windows, target_windows,
                                          audio_windows, indices, batch_size=batch_size,
                                          mask=mask, audio_masks=audio_masks):
                # skip fully-padded batches, this might happen with unknown speakers
                if (batch[-1] == tokenizer.pad_token_id).all():
                    continue

                yield (uri, batch_windows) + batch
    if shuffle:
        # shuffle all windows
        indices = np.arange(len(text_windows))
        np.random.shuffle(indices)
        for batch in tqdm(batchify_windows(tokenizer, text_windows, target_windows,
                                           audio_windows, indices, batch_size=batch_size,
                                           mask=mask, audio_masks=audio_masks),
                          desc='Encoding batches'):
                # skip fully-padded batches, this might happen with unknown speakers
                if (batch[-1] == tokenizer.pad_token_id).all():
                    continue
                yield (None, []) + batch


def align_audio_targets(tokenizer, audio_window, target_window, audio_emb=None):
    """align audio windows with word-piece tokenization"""
    mask = []
    if audio_emb is None:
        return None, mask
    tokens = tokenizer.tokenize(target_window)
    aligned_audio = []
    previous_a = np.ones((1, audio_emb.dimension))
    for i, (a, tgt) in enumerate(zip_longest(audio_window, tokens)):
        if i >= max_length:
            break
        # sub-word -> add audio representation of the previous word
        if tgt.startswith('##'):
            aligned_audio.append(previous_a)
        else:
            if a is None:
                mask.append(i)
                a = np.ones(audio_emb.dimension)
            a = a.reshape(1, -1)
            aligned_audio.append(a)
            previous_a = a
    mask = np.array(mask, dtype=int)
    aligned_audio = np.concatenate(aligned_audio)
    return aligned_audio, mask


def batchify_windows(tokenizer, text_windows, target_windows, audio_windows, indices,
                     batch_size=128, mask=True, audio_masks=None):
    """
    Parameters
    ----------
    see batchify

    Yields
    -------
    see batch_encode_multi
    """
    # split windows in batches w.r.t. batch_size
    # keep remainder (i.e. last batch of size <= batch_size)
    for i in range(0, len(indices), batch_size):
        text_batch, target_batch, audio_batch, audio_mask_batch = [], [], [], []
        for j in indices[i: i + batch_size]:
            text_batch.append(text_windows[j])
            target_batch.append(target_windows[j])
            audio_window = audio_windows[j]
            if audio_window is not None:
                audio_batch.append(audio_window)
                audio_mask_batch.append(audio_masks[j])
        # encode batch (i.e. tokenize, tensorize...)
        batch = batch_encode_multi(tokenizer, text_batch, target_batch, audio_batch,
                                   mask=mask, audio_mask_batch=audio_mask_batch)

        # append original text and target to be able to evaluate
        # (FIXME: this might add extra memory usage, unnecessary to train the model)
        yield (text_batch, target_batch) + batch


def batch_encode_plus(tokenizer, text_batch, mask=True, is_pretokenized=False, 
                      add_special_tokens=False):
    """Shortcut function to encode a text (either input or target) batch
    using tokenizer.batch_encode_plus with the appropriate parameters.

    Parameters
    ----------
    tokenizer: BertTokenizer
    text_batch:
        - List[List[str]] if is_pretokenized
        - List[str] otherwise
    mask: bool, optional
        Compute attention_mask according to max_length.
        Defaults to True.
    is_pretokenized, add_special_tokens: bool, optional
        see tokenizer.batch_encode_plus
        Defaults to False
    Returns
    -------
    input_ids: Tensor
        (batch_size, max_length). Encoded input tokens using BertTokenizer
    attention_mask: Tensor
        (batch_size, max_length). Used to mask input_ids.
        None if not mask.
    """
    text_encoded_plus = tokenizer.batch_encode_plus(text_batch,
                                                    add_special_tokens=add_special_tokens,
                                                    max_length=max_length,
                                                    pad_to_max_length='right',
                                                    return_tensors='pt',
                                                    return_attention_mask=mask,
                                                    is_pretokenized=is_pretokenized)
    input_ids = text_encoded_plus['input_ids']
    attention_mask = text_encoded_plus['attention_mask'] if mask else None
    return input_ids, attention_mask


def batch_encode_multi(tokenizer, text_batch, target_batch, audio_batch=None,
                       mask=True, audio_mask_batch=None):
    """Encode input, target text and audio consistently in torch Tensor

    Parameters
    ----------
    tokenizer: BertTokenizer
        used to tokenize, pad and tensorize text
    text_batch: List[str]
        (batch_size, ) Input text
    target_batch: List[str]
        (batch_size, ) Target speaker names
    audio_batch: List[np.ndarray], optional
        (batch_size, ) Audio embeddings of the input text, aligned with target_ids
        Defaults to None (model only relies on the text).
    mask: bool, optional
        Compute attention_mask according to max_length.
        Defaults to True.
    audio_mask_batch: List[np.ndarray], optional
        indices where audio embeddings are not reliable
        and thus should not weight model's output

    Returns
    -------
    input_ids: Tensor
        (batch_size, max_length). Encoded input tokens using BertTokenizer
    relative_targets: Tensor
        (batch_size, max_length, max_length). one-hot target index w.r.t. input_ids
        e.g. "My name is Paul ." -> one-hot([3, 3, 3, 3, 3])
    audio_similarity: Tensor, optional
        (batch_size, max_length, max_length). Similarity (e.g. cosine distance)
        between audio embeddings of words, aligned with target_ids.
        Defaults to None, indicating that the model should rely only on the text.
    src_key_padding_mask: Tensor, optional
        (batch_size, max_length). Used to mask input_ids.
    tgt_key_padding_mask: Tensor, optional
        (batch_size, max_length). Used to mask relative_targets.
    """
    if len(audio_batch) != 0:
        # compute audio similarity matrix (with numpy as torch doesn't have squareform, yet)
        audio_similarity = np.zeros((len(audio_batch), max_length, max_length), dtype=np.float32)
        for i, (fX, audio_mask) in enumerate(zip(audio_batch, audio_mask_batch)):
            d = squareform(pdist(fX, metric='cosine'))
            # distance to similarity
            d = 1-d
            # mask similarity matrix : masked items are only similar to themselves
            d[audio_mask] = 0
            d[audio_mask, audio_mask] = 1
            audio_similarity[i, : d.shape[0], : d.shape[1]] = d
        # np.ndarray to Tensor
        audio_similarity = from_numpy(audio_similarity)
    else:
        audio_similarity = None

    # tokenize and encode input text: (batch_size, max_length)
    input_ids, src_key_padding_mask = batch_encode_plus(tokenizer, text_batch,
                                                        mask=mask, is_pretokenized=False,
                                                        add_special_tokens=True)
    # encode target text: (batch_size, max_length)
    target_ids, tgt_key_padding_mask = batch_encode_plus(tokenizer, target_batch,
                                                         mask=mask, is_pretokenized=False,
                                                         add_special_tokens=False)

    # fix tgt_key_padding_mask for previously padded targets
    tgt_key_padding_mask[target_ids==tokenizer.pad_token_id] = tokenizer.pad_token_id

    # convert targets to relative targets: (batch_size, max_length, max_length)
    relative_targets = zeros(target_ids.shape + (max_length,))
    for i, (input_id, target_id) in enumerate(zip(input_ids, target_ids)):
        for j, t in enumerate(target_id):
            if t == tokenizer.pad_token_id:
                continue
            where = input_id == t
            # speaker name is not mentioned in input -> pad target
            if not where.any():
                tgt_key_padding_mask[i, j] = tokenizer.pad_token_id
                continue
            where = where.nonzero().reshape(-1)
            relative_targets[i, j, where] = 1.

    return input_ids, relative_targets, audio_similarity, src_key_padding_mask, tgt_key_padding_mask


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
    batch_size = int(args['--batch']) if args['--batch'] else 128
    window_size = int(args['--window']) if args['--window'] else 8
    step_size = int(args['--step']) if args['--step'] else 1
    max_length = int(args['--max_len']) if args['--max_len'] else 256
    mask = True
    easy = args['--easy']
    sep_change = args['--sep_change']
    augment = int(args['--augment']) if args['--augment'] else 0
    protocol = get_protocol(protocol_name)
    serie, _, _ = protocol_name.split('.')
    mapping = DATA_PATH / serie / 'annotated_transcripts' / 'names_dict.json'

    # instantiate tokenizer
    tokenizer = BertTokenizer.from_pretrained(BERT)
    # override basic-tokenization parameter as we need to align speakers with input tokens
    tokenizer.do_basic_tokenize = False

    if args['train']:
        subset = args['--subset'] if args['--subset'] else 'train'
        start_epoch = int(args['--from']) if args['--from'] else None
        train_dir = Path(args['<experiment_dir>'], f'{protocol_name}.{subset}')
        train_dir.mkdir(exist_ok=True)
        config = load_config(train_dir.parents[0])
        architecture = config.get('architecture', {})
        audio = config.get('audio')
        model = SidNet(BERT, max_length, **architecture)
        # get batches from protocol subset
        batches = list(batchify(tokenizer, protocol, mapping, subset, audio_emb=audio,
                                batch_size=batch_size,
                                window_size=window_size,
                                step_size=step_size,
                                mask=mask,
                                easy=easy,
                                sep_change=sep_change,
                                augment=augment,
                                shuffle=True))
        model, optimizer = train(batches, model, tokenizer, train_dir,
                                 start_epoch=start_epoch,
                                 **config.get('training', {}))
    elif args['validate']:
        subset = args['--subset'] if args['--subset'] else 'development'
        evergreen = args['--evergreen']
        interactive = args['--interactive']
        validate_dir = Path(args['<train_dir>'], f'{protocol_name}.{subset}')
        validate_dir.mkdir(exist_ok=True)
        config = load_config(validate_dir.parents[1])

        architecture = config.get('architecture', {})
        audio = config.get('audio')
        model = SidNet(BERT, max_length, **architecture)

        # get batches from protocol subset
        batches = list(batchify(tokenizer, protocol, mapping, subset, audio_emb=audio,
                                batch_size=batch_size,
                                window_size=window_size,
                                step_size=step_size,
                                mask=mask,
                                easy=easy,
                                sep_change=sep_change,
                                augment=augment,
                                shuffle=False))
        eval(batches, model, tokenizer, validate_dir,
             test=False, evergreen=evergreen, interactive=interactive,
             step_size=step_size, window_size=window_size)

    elif args['test']:
        subset = args['--subset'] if args['--subset'] else 'test'
        interactive = args['--interactive']
        test_dir = Path(args['<validate_dir>'], f'{protocol_name}.{subset}')
        test_dir.mkdir(exist_ok=True)
        config = load_config(test_dir.parents[2])

        architecture = config.get('architecture', {})
        audio = config.get('audio')
        model = SidNet(BERT, max_length, **architecture)

        # get batches from protocol subset
        batches = list(batchify(tokenizer, protocol, mapping, subset, audio_emb=audio,
                                batch_size=batch_size,
                                window_size=window_size,
                                step_size=step_size,
                                mask=mask,
                                easy=easy,
                                sep_change=sep_change,
                                augment=augment,
                                shuffle=False))

        eval(batches, model, tokenizer, test_dir,
             test=True, interactive=interactive,
             step_size=step_size, window_size=window_size)


