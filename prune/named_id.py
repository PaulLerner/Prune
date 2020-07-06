#!/usr/bin/env python
# encoding: utf-8

"""Usage:
named_id.py train <protocol> [--subset=<subset> --batch=<batch> --window=<window> --step=<step>]

--subset=<subset>	 Protocol subset, one of 'train', 'development' or 'test' [default: train]
--batch=<batch>		 Batch size [default: 128]
--window=<window>	 Window size [default: 8]
--step=<step>		 Step size [default: 1]
"""

from docopt import docopt
from tqdm import tqdm
from pathlib import Path
import json

from pyannote.core import Segment
from pyannote.database import get_protocol
import Plumcot as PC

import numpy as np

from torch import save, manual_seed
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from transformers import BertTokenizer
from prune.sidnet import SidNet

np.random.seed(0)
manual_seed(0)
DATA_PATH = Path(PC.__file__).parent / 'data'
pad_token, pad_int = '[PAD]', 0
max_length = 256


def train(batches, bert='bert-base-cased', vocab_size=28996, audio=None, lr=1e-3, 
          epochs=100, freeze=['bert'], save_every=1, **kwargs):
    """Train the model for `epochs` epochs

    Parameters
    ----------
    batches: List[Tuple[Tensor]]:
        (input_ids, target_ids, audio_similarity, src_key_padding_mask, tgt_key_padding_mask)
        see batch_encode_multi
    bert: `str`, optional
        Model name or path, see BertTokenizer.from_pretrained
        Defaults to 'bert-base-cased'.
    vocab_size: `int`, optional
        Output dimension of the model (should be inferred from vocab_size of BertTokenizer)
        Defaults to 28996
    audio: `Wrappable`, optional
        Describes how raw speaker embeddings should be obtained.
        See pyannote.audio.features.wrapper.Wrapper documentation for details.
        Defaults to None, indicating that the model should rely only on the text.
    lr: float, optional
        Learning rate used to optimize model parameters.
        Defaults to 1e-3
    epochs: int, optional
        Train the model for `epochs` epochs.
        Defaults to 100
    freeze : List[str], optional
        Names of modules to freeze.
        Defaults to freezing bert (['bert']).
    save_every: int, optional
        Save model weights and optimizer state every `save_every` epoch.
        Defaults to save at every epoch (1)
    **kwargs:
        Additional arguments are passed to the model Transformer
    """
    model = SidNet(bert, vocab_size, audio, **kwargs)
    model.freeze(freeze)
    print(model)
    model.train()

    criterion = CrossEntropyLoss(ignore_index=pad_int)
    optimizer = Adam(model.parameters(), lr=lr)

    tb = SummaryWriter()
    for epoch in tqdm(range(epochs), desc='Training'):
        # shuffle batches
        np.random.shuffle(batches)

        epoch_loss = 0.
        for input_ids, target_ids, audio_similarity, src_key_padding_mask, tgt_key_padding_mask in batches:
            optimizer.zero_grad()

            # forward pass
            output = model(input_ids, target_ids, audio_similarity,
                           src_key_padding_mask, tgt_key_padding_mask)

            # calculate loss
            loss = criterion(output, target_ids)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        tb.add_scalar('Loss/train', epoch_loss, epoch)

        if epoch % save_every == 0:
            save(model, f'weights/{model.__class__.__name__}.{epoch:04d}.pt')
            save(optimizer, f'weights/{optimizer.__class__.__name__}.{epoch:04d}.pt')

    return model, optimizer


def batchify(protocol, mapping, subset='train', bert='bert-base-cased',
             batch_size=128, window_size=10, step_size=1):
    """Iterates over protocol subset, segment transcription in speaker turns,
    Divide transcription in windows then split windows in batches.
    And finally, encode batch (i.e. tokenize, tensorize...)

    mapping is used to convert normalized speaker names into its most common name.
    Note that it's important that this name is as written in the input text.

    Returns
    -------
    batches: List[Tuple[Tensor]]:
        (input_ids, target_ids, audio_similarity, src_key_padding_mask, tgt_key_padding_mask)
        see batch_encode_multi
    vocab_size: int
        tokenizer.vocab_size
    """

    with open(mapping) as file:
        mapping = json.load(file)

    tokenizer = BertTokenizer.from_pretrained(bert)
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
                windows.append((start, end))
                start = end
            tokens.append(token.text)
            # if audio alignment is not confident for token
            # then audio similarity matrix of token should be uniform
            # so it doesn't weigh on the text decision
            if token._.confidence > 0.5 and '#unknown#' not in token._.speaker:
                audio.append(Segment(token._.time_start, token._.time_end))
            else:
                audio.append(pad_token)
            # if we don't have a proper target we should mask the loss function
            targets.append(mapping.get(token._.speaker, pad_token))
            previous_speaker = token._.speaker
            end += 1
        windows.pop(0)

        # slide through the transcription speaker turns w.r.t. window_size, step_size
        for i in range(0, len(windows) - window_size, step_size):
            start, _ = windows[i]
            _, end = windows[i + window_size - 1]
            text_windows.append(" ".join(tokens[start:end]))
            audio_windows.append(audio[start:end])
            target_windows.append(" ".join(targets[start:end]))

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
        batch = batch_encode_multi(tokenizer, text_batch, target_batch, audio_batch)
        batches.append(batch)
    return batches, tokenizer.vocab_size


def batch_encode_plus(tokenizer, text_batch):
    """Shortcut function to encode a text (either input or target) batch
    using tokenizer.batch_encode_plus with the appropriate parameters.

    Returns
    -------
    input_ids: Tensor
        (batch_size, max_length). Encoded input tokens using BertTokenizer
    attention_mask: Tensor
            (batch_size, max_length). Used to mask input_ids.
    """
    text_encoded_plus = tokenizer.batch_encode_plus(text_batch,
                                                    add_special_tokens=False,
                                                    max_length=max_length,
                                                    pad_to_max_length='right',
                                                    return_tensors='pt',
                                                    return_attention_mask=True,
                                                    return_special_tokens_mask=True)
    input_ids = text_encoded_plus['input_ids']
    attention_mask = text_encoded_plus['attention_mask']
    return input_ids, attention_mask


def batch_encode_multi(tokenizer, text_batch, target_batch, audio_batch=None):
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
    input_ids, src_key_padding_mask = batch_encode_plus(tokenizer, text_batch)

    # encode target text
    target_ids, tgt_key_padding_mask = batch_encode_plus(tokenizer, target_batch)
    # fix tgt_key_padding_mask when targets where previously tagged as '[PAD]' -> 0
    # FIXME is there a better way to do this?
    tgt_key_padding_mask[target_ids == 0] = pad_int

    return input_ids, target_ids, audio_similarity, src_key_padding_mask, tgt_key_padding_mask


if __name__ == '__main__':
    # parse arguments and get protocol
    args = docopt(__doc__)
    protocol_name = args['<protocol>']
    subset = args['--subset'] if args['--subset'] else 'train'
    batch_size = int(args['--batch']) if args['--batch'] else 128
    window_size = int(args['--window']) if args['--window'] else 8
    step_size = int(args['--step']) if args['--step'] else 1
    protocol = get_protocol(protocol_name)
    serie, _, _ = protocol_name.split('.')
    mapping = DATA_PATH / serie / 'annotated_transcripts' / 'names_dict.json'

    # get batches from protocol subset
    batches, vocab_size = batchify(protocol, mapping, subset,
                                   batch_size=batch_size, 
                                   window_size=window_size, 
                                   step_size=step_size)

    if args['train']:
        model, optimizer = train(batches, vocab_size=vocab_size)


