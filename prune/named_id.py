#!/usr/bin/env python
# encoding: utf-8

"""Usage:
named_id.py train <protocol> <experiment_dir> [options] [--from=<epoch>] [(--augment=<ratio> [--uniform])]
named_id.py validate <protocol> <train_dir> [options] [--evergreen --interactive]
named_id.py test <protocol> <validate_dir> [options] [--interactive]
named_id.py oracle <protocol> [options]
named_id.py visualize <protocol> [<validate_dir>]

Common options:

--subset=<subset>    Protocol subset, one of 'train', 'development' or 'test'.
                     Defaults to 'train', 'development' and 'test' in
                     'train', 'validate', and 'test' mode, respectively.
--batch=<batch>      Batch size (# of windows) [default: 128]
--window=<window>    Window size (# of speaker turns) [default: 8]
--step=<step>        Step size (overlap between windows) [default: 1]
--max_len=<max_len>  Maximum # of tokens input to BERT. Maximum 512 [default: 256]
--easy               Only keep text windows with named speakers in it.
--sep_change         Add a special "[SEP]" token between every speech turn.

Training options:
--from=<epoch>       Start training back from a specific checkpoint (epoch #)
--augment=<ratio>    If different from 0, will generate `|augment|` synthetic examples per real example
                     If less than 0, will discard real example.
                     See batchify for details.
                     Defaults to no augmentation.
--uniform            When augmenting data, pick fake names with uniform distribution
                     regardless of their frequency in the name database.

Validation options:
--evergreen          Start with the latest checkpoints
--interactive        Open-up python debugger after each forward pass

To use meta-protocols, name it like: "X.SpeakerDiarization.<serie1>+<serie2>"
(with '+' separated serie names so that we're able to load the appropriate mapping)
e.g. "X.SpeakerDiarization.BuffyTheVampireSlayer+Friends"

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
│   │   │   │   eval

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
from collections import Counter

from pyannote.core import Segment
from pyannote.core.utils.distance import pdist
from pyannote.database import get_protocol
from pyannote.audio.features.wrapper import Wrapper, Wrappable
import Plumcot as PC

import re
import numpy as np
from scipy.spatial.distance import squareform
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

from torch import save, load, manual_seed, no_grad, argmax, Tensor, zeros, from_numpy, \
    zeros_like, LongTensor, ones, float, cat
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.nn import BCELoss, DataParallel
from transformers import BertTokenizer
from prune.sidnet import SidNet
from prune.convert import id_to_annotation

# set random seed
np.random.seed(0)
manual_seed(0)

EPOCH_FORMAT = '{:04d}.tar'
BERT = 'bert-base-cased'
PROPN = 'PROPN'

# constant paths
DATA_PATH = Path(PC.__file__).parent / 'data'
CHARACTERS_PATH = DATA_PATH.glob('*/characters.txt')


def proper_entity(token):
    return token.ent_kb_id_ != '' and token.pos_ == PROPN and token.ent_kb_id_ not in PC.loader.NA


def format_acc(ratio):
    return f'{ratio * 100:.2f}'


def token_accuracy(targets: Tensor, predictions: Tensor, pad: int=0):
    """Compute accuracy at the token level.
    Ignores padded targets.
    """

    indices = targets != pad
    where = (predictions[indices] == targets[indices]).nonzero(as_tuple=True)
    return where[0].shape[0] / indices.nonzero(as_tuple=True)[0].shape[0]


def batch_word_accuracy(targets: List[str], predictions: List[str],
                        pad='[PAD]', split=True):
    """Computes word accuracy at the batch level

    Parameters
    ----------
    targets, predictions:
        - List[str] if split
        - List[List[str]] otherwise
    pad: str, optional
        special token to be ignored
        Defaults to '[PAD]'
    split : bool, optional
        Whether to split items in targets, predictions
        Defaults to True

    Returns
    -------
    accuracy: float
    """
    correct, total = 0, 0
    for target, prediction in zip(targets, predictions):
        if split:
            target, prediction = target.split(), prediction.split()
        for t, p in zip_longest(target, prediction, fillvalue=pad):
            if t == pad:
                continue
            if t == p:
                correct += 1
            total += 1
    return correct/total


def speaker_alias_accuracy(speaker_ids: List[List[str]], predictions: List[str], aliases: dict,
                            pad='[PAD]', split=True):
    """Computes alias accuracy at the batch level
    We consider that the model hypothesis is correct if it's one of the targets aliases
    -> if the target was mentioned with this name in the input

    Parameters
    ----------
    speaker_ids: List[List[str]]
    predictions:
        - List[str] if split
        - List[List[str]] otherwise
    aliases: dict
        Aliases of targets: {str: Set[str]}
    pad: str, optional
        special token to be ignored
        Defaults to '[PAD]'
    split : bool, optional
        Whether to split items in predictions
        Defaults to True

    Returns
    -------
    accuracy: float

    See Also
    --------
    batch_word_accuracy
    """
    correct, total = 0, 0
    for speaker_id, prediction in zip(speaker_ids, predictions):
        if split:
            prediction = prediction.split()
        for id, p in zip_longest(speaker_id, prediction, fillvalue=pad):
            alias = aliases.get(id, set())
            if id == pad or not alias:
                continue
            if p in alias:
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
    merge = []
    i = 0
    for token in inp_eg:
        merge.append(f"{token} ({tgt_eg[i]})")
        if not token.startswith('##'):
            i += 1
    max_len = len(inp_eg)
    plt.figure(figsize=(max_len//6, max_len//6))
    # shift by 1 to discard [CLS] and [SEP] tokens
    plt.imshow(output_eg.detach().cpu().numpy()[:max_len, 1: max_len-1])
    plt.colorbar()
    plt.xticks(range(max_len), inp_eg[:max_len], fontsize='x-small', rotation='vertical')
    plt.yticks(range(max_len), merge[:max_len], fontsize='x-small', rotation='horizontal')
    if save is None:
        plt.show()
    else:
        plt.savefig(save/"output.png")


def mode(prediction, pad='[PAD]'):
    """Returns most common predicted item or pad if no items were predicted"""
    if prediction:
        return prediction.most_common(1)[0][0]
    return pad


def eval(batches, model, tokenizer, log_dir,
         test=False, evergreen=False, interactive=False, step_size=1, window_size=10):
    """Load model from checkpoint and evaluate it on batches.
    When testing, only the best model should be tested.

    Parameters
    ----------
    batches: see batchify
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
    params_file = log_dir.parent / 'params.yml'
    if test:
        weights_path = log_dir.parents[1] / 'weights'
        with open(params_file) as file:
            epoch = yaml.load(file, Loader=yaml.SafeLoader)["epoch"]
        weights = [weights_path/EPOCH_FORMAT.format(epoch)]
        best = 0.
        rttm_out = log_dir/'hypothesis.rttm'
        if rttm_out.exists():
            raise FileExistsError(f"'{rttm_out}' already exists, you probably don't want "
                                  f"to append any more data to it")
    else:
        weights_path = log_dir.parents[0] / 'weights'
        weights = sorted(weights_path.iterdir(), reverse=evergreen)
        if params_file.exists():
            with open(params_file) as file:
                best = yaml.load(file, Loader=yaml.SafeLoader)["accuracy"]
        else:
            best = 0.

    criterion = BCELoss(reduction='none')
    tb = SummaryWriter(log_dir)
    for weight in tqdm(weights, desc='Evaluating'):
        checkpoint = load(weight, map_location=model.src_device_obj)
        epoch = checkpoint["epoch"]
        model.module.load_state_dict(checkpoint['model_state_dict'])
        # manage device FIXME this should be ok after map_location ??
        model.module.to(model.src_device_obj)
        model.eval()
        with no_grad():
            epoch_loss, epoch_word_acc, epoch_alias_acc = 0., 0., 0.
            uris, file_token_acc, file_word_acc, file_alias_acc = [], [], [], []
            previous_uri = None
            for uri, timings, windows, aliases, inp, tgt, speaker_id_batch, audio_batch, audio_mask, input_ids, target_ids, src_key_padding_mask, tgt_key_padding_mask in batches:
                # forward pass: (batch_size, sequence_length, sequence_length)
                output = model(input_ids, audio_batch, src_key_padding_mask, audio_mask)
                # manage devices
                target_ids = target_ids.to(output.device)

                # get model prediction per token: (batch_size, sequence_length)
                relative_out = argmax(output, dim=2)
                # retrieve token ids from input (batch_size, sequence_length) and manage device
                prediction_ids = zeros_like(input_ids, device=output.device)
                for j, (input_window_id, relative_window_out) in enumerate(zip(input_ids, relative_out)):
                    prediction_ids[j] = input_window_id[relative_window_out]

                # decode and compute word accuracy
                predictions = tokenizer.batch_decode(prediction_ids, clean_up_tokenization_spaces=False)
                epoch_word_acc += batch_word_accuracy(tgt, predictions, tokenizer.pad_token)

                # compute alias accuracy
                if aliases:
                    epoch_alias_acc += speaker_alias_accuracy(speaker_id_batch, predictions, aliases,
                                                              pad=tokenizer.pad_token)

                # calculate loss
                loss = criterion(output, target_ids)
                loss = reduce_loss(loss, tgt_key_padding_mask)
                epoch_loss += loss.item()

                # handle file-level stuff
                if uri != previous_uri:
                    # compute file-level accuracy and save output if testing
                    if previous_uri is not None:
                        uris.append(previous_uri)
                        # merge window-level predictions
                        file_predictions = [mode(p, tokenizer.pad_token) for p in file_predictions]
                        # compute word accuracy
                        file_word_acc.append(batch_word_accuracy([file_target],
                                                                 [file_predictions],
                                                                 pad=tokenizer.pad_token,
                                                                 split=False))
                        # compute speaker alias accuracy
                        if aliases:
                            file_alias_acc.append(speaker_alias_accuracy([file_speaker_id],
                                                                         [file_predictions],
                                                                         aliases,
                                                                         pad=tokenizer.pad_token,
                                                                         split=False))
                        # convert hypothesis to pyannote Annotation
                        if test:
                            hypothesis = id_to_annotation(file_predictions,
                                                          previous_timings,
                                                          previous_uri)

                            # save hypothesis to rttm
                            with open(rttm_out, 'a') as file:
                                hypothesis.write_rttm(file)

                    # reset file-level variables
                    file_length = windows[-1][-1] - windows[0][0]
                    i, shift = 0, 0
                    file_target = [tokenizer.pad_token] * file_length
                    file_speaker_id = [tokenizer.pad_token] * file_length
                    file_predictions = [Counter() for _ in range(file_length)]

                # save target and output for future file-level accuracy
                for target_i, pred_i, speaker_id_i in zip_longest(tgt, predictions, speaker_id_batch):
                    target_i, pred_i = target_i.split(), pred_i.split()
                    for start, end in windows[i: i+window_size]:
                        file_target[start:end] = target_i[start-shift: end-shift]
                        if speaker_id_i is not None:
                            file_speaker_id[start:end] = speaker_id_i[start-shift: end-shift]
                        for counter, p in zip(file_predictions[start:end], pred_i[start-shift: end-shift]):
                            counter[p] += 1
                    i += step_size
                    # shift between batch and original file
                    shift = windows[i][0]  # start

                if interactive:
                    eg = np.random.randint(len(tgt))
                    inp_eg, tgt_eg, pred_eg = inp[eg], tgt[eg], predictions[eg]
                    # print random example
                    print(str_example(inp_eg.split(), tgt_eg.split(), pred_eg.split()))
                    # plot model output
                    plot_output(output[eg], tokenizer.tokenize(inp_eg), 
                                tgt_eg.split(), log_dir)

                    # print current metrics
                    metrics = {
                        'Loss/eval': [epoch_loss],
                        'Accuracy/eval/batch/word': [format_acc(epoch_word_acc)],
                        'Accuracy/eval/batch/speaker_alias': [format_acc(epoch_alias_acc)]
                    }
                    print(tabulate(metrics, headers='keys', disable_numparse=True))
                    breakpoint()

                previous_uri = uri
                previous_timings = timings

            # compute file-level accuracy for the last file
            uris.append(previous_uri)
            # merge window-level predictions
            file_predictions = [mode(p, tokenizer.pad_token) for p in file_predictions]
            # compute word accuracy
            file_word_acc.append(batch_word_accuracy([file_target],
                                                     [file_predictions],
                                                     pad=tokenizer.pad_token,
                                                     split=False))
            # compute speaker alias accuracy
            if aliases:
                file_alias_acc.append(speaker_alias_accuracy([file_speaker_id],
                                                             [file_predictions],
                                                             aliases,
                                                             pad=tokenizer.pad_token,
                                                             split=False))
            # convert hypothesis to pyannote Annotation
            if test:
                hypothesis = id_to_annotation(file_predictions, timings, uri)
                # save hypothesis to rttm
                with open(rttm_out, 'a') as file:
                    hypothesis.write_rttm(file)

            # average file-accuracies
            uris.append('TOTAL')
            file_word_acc.append(np.mean(file_word_acc))
            if file_alias_acc:
                file_alias_acc.append(np.mean(file_alias_acc))

            # log tensorboard
            tb.add_scalar('Accuracy/eval/file/word', file_word_acc[-1], epoch)
            epoch_loss /= len(batches)
            tb.add_scalar('Loss/eval', epoch_loss, epoch)
            epoch_word_acc /= len(batches)
            tb.add_scalar('Accuracy/eval/batch/word', epoch_word_acc, epoch)
            epoch_alias_acc /= len(batches)
            tb.add_scalar('Accuracy/eval/batch/speaker_alias', epoch_alias_acc, epoch)

            # format file-accuracies in % with .2f
            file_word_acc = [format_acc(acc) for acc in file_word_acc]
            file_alias_acc = [format_acc(acc) for acc in file_alias_acc]

            # print and write metrics
            if test:
                epoch_alias_acc = format_acc(epoch_alias_acc) if aliases else "-"
                metrics = {
                    'Loss/eval': [epoch_loss],
                    'Accuracy/eval/batch/word': [format_acc(epoch_word_acc)],
                    'Accuracy/eval/batch/speaker_alias': [epoch_alias_acc]
                }
                metrics = tabulate(metrics, headers='keys', tablefmt='latex', disable_numparse=True)
                metrics += tabulate(zip_longest(uris, file_word_acc, file_alias_acc, fillvalue='-'),
                                    headers=['uri', 'word-level', 'alias'],
                                    tablefmt='latex',
                                    disable_numparse=True)
                print(metrics)
                with open(log_dir / 'eval', 'w') as file:
                    file.write(metrics)
            # dump best metrics
            elif epoch_word_acc > best:
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
    batches: see batch_encode_multi
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
    optimizer = Adam(model.module.parameters(), lr=lr)

    weights_path = train_dir / 'weights'
    # load previous checkpoint
    if start_epoch is not None:
        checkpoint = load(weights_path / EPOCH_FORMAT.format(start_epoch)
                          ,map_location=model.src_device_obj)
        assert start_epoch == checkpoint["epoch"]
        model.module.load_state_dict(checkpoint['model_state_dict'])
        # manage device FIXME this should be ok after map_location ??                 
        model.module.to(model.src_device_obj)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # increment epoch
        start_epoch += 1
    else:
        # put parallelized module to model.src_device_obj
        model.module.to(model.src_device_obj)
        # be careful not to erase previous weights
        weights_path.mkdir(exist_ok=False)
        # defaults to start from 0
        start_epoch = 0

    model.module.freeze(freeze)
    model.train()
    criterion = BCELoss(reduction='none')

    tb = SummaryWriter(train_dir)
    for epoch in tqdm(range(start_epoch, epochs+start_epoch), desc='Training'):
        # shuffle batches
        np.random.shuffle(batches)

        epoch_loss = 0.
        for _, _, _, _, _, _, _, audio_batch, audio_mask, input_ids, target_ids, src_key_padding_mask, tgt_key_padding_mask in batches:
            optimizer.zero_grad()

            # forward pass
            output = model(input_ids, audio_batch, src_key_padding_mask, audio_mask)
            # manage devices
            target_ids = target_ids.to(output.device)

            # calculate loss
            loss = criterion(output, target_ids)
            # mask and reduce loss
            loss = reduce_loss(loss, tgt_key_padding_mask)
            loss.backward()

            if max_grad_norm is not None:
                clip_grad_norm_(model.module.parameters(), max_grad_norm)

            optimizer.step()
            epoch_loss += loss.item()

        tb.add_scalar('Loss/train', epoch_loss/len(batches), epoch)
        tb.add_scalar('lr', lr, epoch)

        if (epoch+1) % save_every == 0:
            save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
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
             batch_size=128, window_size=10, step_size=1, mask=True, easy=False,
             sep_change=False, augment=0, uniform=False, shuffle=True, oracle=False):
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
    uniform: bool, optional
        When augmenting data, pick fake names with uniform distribution
        regardless of their frequency in the name database.
        Has no effect if augment==0
    shuffle: bool, optional
        Whether to shuffle windows before batching.
        This is also synonymous with training.
        Should be set to False when testing to get file-homogeneous batches,
        and to True while training to ensure stochasticity.
        Defaults to True
    oracle: bool, optional
        Compute oracle accuracy for protocol's subset
        Enforces shuffle = False
        Oracles knows who the speaker is if it's name (case-insensitive)
        is mentioned in the input. Most of the other arguments are not relevant
        in this case, and yields (uri, common_accuracy, alias_accuracy, n_tokens) instead of what's documented below.
        Defaults to False

    Yields
    -------
    batch: Tuple[str, List[Tuple[int]], List[str], Tensor]:
        (uri, timings, windows, aliases, text_batch, target_batch, speaker_id_batch, audio_batch, audio_mask, input_ids, target_ids, src_key_padding_mask, tgt_key_padding_mask)
        - see batch_encode_multi.
        - uri: str,
          file-identifier of the batch and is set to None if shuffle, as batch
          are then file-heterogeneous
        - timings: List[Segment]
          timestamps of the input words
          Empty if shuffling
        - windows: List[Tuple[int]]
          indices of the start and end words index of the speaker turns in the batch
          Empty if shuffling
        - aliases: dict
          mapping between speaker identifier and names it has been mentioned
          {str: Set[str]}
        - speaker_id_batch: List[str]
          speaker identifier (typically "firstname_lastname") as opposed to
          target_batch which is the speaker most common name (typically "firstname")
    """
    assert not tokenizer.do_basic_tokenize, "Basic tokenization is handle beforehand"

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
        if uniform:
            names = np.unique(names)

    text_windows, audio_windows, target_windows, audio_masks, speaker_id_windows = [], [], [], [], []
    aliases = {}
    if oracle and shuffle:
        shuffle = False
        warnings.warn("Setting 'shuffle = False' because 'oracle' mode is on.")
    # iterate over protocol subset
    for current_file in tqdm(getattr(protocol, subset)(), desc='Loading transcriptions'):
        if not shuffle:
            oracle_correct, oracle_total = 0, 0
            oracle_alias_correct, oracle_alias_total = 0, 0
            n_tokens = []
            aliases = {}
            text_windows, audio_windows, target_windows, audio_masks, speaker_id_windows = [], [], [], [], []
        transcription = current_file.get('entity', current_file['transcription'])
        uri = current_file['uri']

        current_audio_emb = None
        # get audio embeddings from current_file
        if audio_emb is not None:
            current_audio_emb = audio_emb(current_file)

        # format transcription into 5 lists: tokens, audio, targets, speaker_ids, timings
        # and segment it in speaker turns (i.e. speaker homogeneous)
        windows = []
        start, end = 0, 0
        tokens, audio, targets, speaker_ids, timings = [], [], [], [], []
        previous_speaker = None
        for word in transcription:
            if word._.speaker != previous_speaker:
                # mark speaker change with special token tokenizer.sep_token ("[SEP]")
                if sep_change:
                    tokens.append(tokenizer.sep_token)
                    targets.append(tokenizer.pad_token)
                    audio.append(None)
                    timings.append(Segment())
                    end += 1
                windows.append((start, end))
                start = end

            # get audio embedding and timing for word if forced-alignment is confident enough
            if word._.confidence > 0.5:
                timing = Segment(word._.time_start, word._.time_end)
                if audio_emb is not None:
                    segment = current_audio_emb.crop(timing, mode="loose")
                    # skip segments so small we don't have any embedding for it
                    if len(segment) < 1:
                        segment = None
                    # average audio-embedding over the segment frames
                    else:
                        segment = np.mean(segment, axis=0)
                else:
                    segment = None
            else:
                segment = None
                timing = Segment()
            # if we don't have a proper target we should mask the loss function
            target = mapping.get(word._.speaker, tokenizer.pad_token)

            # handle basic tokenization (e.g. punctuation) before Word-Piece
            # in order to align input text and speakers
            basic_token = tokenizer.basic_tokenizer.tokenize(word.text)
            for token in basic_token:
                tokens.append(token)
                targets.append(target)
                speaker_ids.append(word._.speaker)
                audio.append(segment)
                timings.append(timing)
                end += 1

            if proper_entity(word):
                aliases.setdefault(word.ent_kb_id_, set())
                # HACK: there might be some un-basic-tokenized words in there such as "Angelas's"
                # so we keep only the first basic-token
                # TODO fix this upstream
                aliases[word.ent_kb_id_].add(basic_token[0])

            previous_speaker = word._.speaker
        windows.append((start, end))
        windows.pop(0)

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
            speaker_id_windows.append(speaker_ids[start: end])
            # compute oracle-accuracy
            if oracle:
                n_tokens.append(end-start)
                # speaker alias accuracy
                for speaker_id in speaker_ids[start: end]:
                    if not aliases:
                        break
                    alias = aliases.get(speaker_id)
                    if not alias:
                        continue
                    for target in alias:
                        if re.search(target, text_window, flags=re.IGNORECASE):
                            oracle_alias_correct += 1
                            break
                    oracle_alias_total += 1
                # regular "common-name" accuracy
                for target in targets[start:end]:
                    if target in tokenizer.all_special_tokens:
                        continue
                    if re.search(target, text_window, flags=re.IGNORECASE):
                        oracle_correct += 1
                    oracle_total += 1

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
        if not shuffle and not oracle:
            indices = np.arange(len(text_windows))
            for batch in batchify_windows(tokenizer, text_windows, target_windows,
                                          audio_windows, speaker_id_windows, indices, batch_size=batch_size,
                                          mask=mask, audio_masks=audio_masks, yield_text=True):
                # skip fully-padded batches, this might happen with unknown speakers
                if (batch[-1] == tokenizer.pad_token_id).all():
                    continue

                yield (uri, timings, windows, aliases) + batch
        # yield (uri, oracle_accuracy)
        elif oracle:
            alias_accuracy = None if oracle_alias_total == 0. else oracle_alias_correct/oracle_alias_total
            yield uri, oracle_correct/oracle_total, alias_accuracy, n_tokens

    if shuffle:
        # shuffle all windows
        indices = np.arange(len(text_windows))
        np.random.shuffle(indices)
        for batch in tqdm(batchify_windows(tokenizer, text_windows, target_windows,
                                           audio_windows, speaker_id_windows, indices, batch_size=batch_size,
                                           mask=mask, audio_masks=audio_masks, yield_text=False),
                          desc='Encoding batches'):
            # skip fully-padded batches, this might happen with unknown speakers
            if (batch[-1] == tokenizer.pad_token_id).all():
                continue
            yield (None, [], [], aliases) + batch


def align_audio_targets(tokenizer, audio_window, target_window, audio_emb=None):
    """align audio windows with word-piece tokenization"""
    # init mask to all items
    mask = ones(max_length, dtype=bool)
    if audio_emb is None:
        return None, mask
    tokens = tokenizer.tokenize(target_window)
    # init embeddings to 1
    aligned_audio = ones(max_length, audio_emb.dimension, dtype=float)
    for i, (a, tgt) in enumerate(zip_longest(audio_window, tokens)):
        if i >= max_length:
            break
        # sub-word -> add audio representation of the previous word
        if tgt.startswith('##'):
            aligned_audio[i] = aligned_audio[i-1]
        elif a is not None:
            mask[i] = False
            aligned_audio[i] = Tensor(a)
    return aligned_audio, mask


def batchify_windows(tokenizer, text_windows, target_windows, audio_windows, speaker_id_windows, indices,
                     batch_size=128, mask=True, audio_masks=None, yield_text=True):
    """
    Parameters
    ----------
    see batchify
    yield_text: bool, optional
        Whether to yield text_batch, target_batch and speaker_id_batch

    Yields
    -------
    see batch_encode_multi
    """
    # split windows in batches w.r.t. batch_size
    # keep remainder (i.e. last batch of size <= batch_size)
    for i in range(0, len(indices), batch_size):
        text_batch, target_batch, audio_batch, audio_mask, speaker_id_batch = [], [], [], [], []
        for j in indices[i: i + batch_size]:
            text_batch.append(text_windows[j])
            target_batch.append(target_windows[j])
            speaker_id_batch.append(speaker_id_windows[j])
            audio_window = audio_windows[j]
            if audio_window is not None:
                audio_batch.append(audio_window.unsqueeze(0))
                audio_mask.append(audio_masks[j].unsqueeze(0))
        if len(audio_batch) != 0:
            audio_batch = cat(audio_batch, dim=0)
            audio_mask = cat(audio_mask, dim=0)
        else:
            audio_batch, audio_mask = None, None
        # encode batch (i.e. tokenize, tensorize...)
        batch = batch_encode_multi(tokenizer, text_batch, target_batch, mask=mask)

        # append original text and target to be able to evaluate
        if not yield_text:
            text_batch, target_batch, speaker_id_batch = None, None, None
        yield (text_batch, target_batch, speaker_id_batch, audio_batch, audio_mask) + batch


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


def batch_encode_multi(tokenizer, text_batch, target_batch, mask=True):
    """Encode input, target text and audio consistently in torch Tensor

    Parameters
    ----------
    tokenizer: BertTokenizer
        used to tokenize, pad and tensorize text
    text_batch: List[str]
        (batch_size, ) Input text
    target_batch: List[str]
        (batch_size, ) Target speaker names
    mask: bool, optional
        Compute attention_mask according to max_length.
        Defaults to True.

    Returns
    -------
    input_ids: Tensor
        (batch_size, max_length). Encoded input tokens using BertTokenizer
    relative_targets: Tensor
        (batch_size, max_length, max_length). one-hot target index w.r.t. input_ids
        e.g. "My name is Paul ." -> one-hot([3, 3, 3, 3, 3])
    src_key_padding_mask: Tensor, optional
        (batch_size, max_length). Used to mask input_ids.
    tgt_key_padding_mask: Tensor, optional
        (batch_size, max_length). Used to mask relative_targets.
    """
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

    return input_ids, relative_targets, src_key_padding_mask, tgt_key_padding_mask


def visualize(words, model, tokenizer, validate_dir=None):
    """
    Parameters
    ----------
    words: Iterable[str]
    model: SidNet
    tokenizer: BertTokenizer
    validate_dir: Path
    """
    # load model from validate_dir
    if validate_dir is not None:
        with open(validate_dir / 'params.yml') as file:
            epoch = yaml.load(file, Loader=yaml.SafeLoader)["epoch"]
        weight = validate_dir.parent / 'weights' / EPOCH_FORMAT.format(epoch)
        checkpoint = load(weight, map_location=model.src_device_obj)
        epoch = checkpoint["epoch"]
        model.module.load_state_dict(checkpoint['model_state_dict'])
    # else keep pre-trained BERT
    else:
        validate_dir = Path.cwd()
    model.module.to(model.src_device_obj)
    model.eval()

    # tokenize and encode words
    tokens = tokenizer.tokenize(' '.join(words))
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = LongTensor(input_ids).unsqueeze(0).to(model.src_device_obj)

    # get token embeddings
    embeddings = model.module.bert.embeddings.word_embeddings(input_ids)
    embeddings = embeddings.squeeze(0).detach().cpu().numpy()

    # apply t-SNE
    tsne = TSNE(n_components=2, metric="cosine")
    embeddings_2d = tsne.fit_transform(embeddings)

    # plot the result
    assert len(tokens) == embeddings_2d.shape[0], \
        f"Shape mismatch between token ({len(tokens)}) and embeddings ({embeddings_2d.shape})"
    plt.figure(figsize=(15, 15))
    plt.scatter(*embeddings_2d.T)
    for token, xy in zip(tokens, embeddings_2d):
        plt.annotate(token, xy)
    save_path = validate_dir / "embeddings_TSNE.png"
    plt.savefig(save_path)
    print(f"Succesfully saved figure to {save_path}")


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
    uniform = args['--uniform']
    protocol = get_protocol(protocol_name)

    # handle meta-protocols
    serie, _, x = protocol_name.split('.')
    if serie == 'X':
        series = x.split('+')
    else:
        series = [serie]

    # load mapping(s)
    mapping = {}
    for serie in series:
        mapping_path = DATA_PATH / serie / 'annotated_transcripts' / 'names_dict.json'
        with open(mapping_path) as file:
            mapping.update(json.load(file))

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
        model = DataParallel(SidNet(BERT, max_length, **architecture))
        # get batches from protocol subset
        batches = list(batchify(tokenizer, protocol, mapping, subset, audio_emb=audio,
                                batch_size=batch_size,
                                window_size=window_size,
                                step_size=step_size,
                                mask=mask,
                                easy=easy,
                                sep_change=sep_change,
                                augment=augment,
                                uniform=uniform,
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
        model = DataParallel(SidNet(BERT, max_length, **architecture))

        # get batches from protocol subset
        batches = list(batchify(tokenizer, protocol, mapping, subset, audio_emb=audio,
                                batch_size=batch_size,
                                window_size=window_size,
                                step_size=step_size,
                                mask=mask,
                                easy=easy,
                                sep_change=sep_change,
                                augment=augment,
                                uniform=uniform,
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
        model = DataParallel(SidNet(BERT, max_length, **architecture))

        # get batches from protocol subset
        batches = list(batchify(tokenizer, protocol, mapping, subset, audio_emb=audio,
                                batch_size=batch_size,
                                window_size=window_size,
                                step_size=step_size,
                                mask=mask,
                                easy=easy,
                                sep_change=sep_change,
                                augment=augment,
                                uniform=uniform,
                                shuffle=False))

        eval(batches, model, tokenizer, test_dir,
             test=True, interactive=interactive,
             step_size=step_size, window_size=window_size)
    elif args['visualize']:
        validate_dir = args['<validate_dir>']
        if validate_dir is not None:
            validate_dir = Path(validate_dir)
            config = load_config(validate_dir.parents[1])
        else:
            config = {}
        architecture = config.get('architecture', {})
        model = DataParallel(SidNet(BERT, max_length, **architecture))
        # get list of names
        words = set(mapping.values())
        visualize(words, model, tokenizer, validate_dir)
    elif args['oracle']:
        subset = args['--subset'] if args['--subset'] else 'test'
        full_name = f"{protocol_name}.{subset}"
        # get oracle accuracy for protocol subset
        uris, accuracies, alias_accuracies, n_tokens = [], [], [], []
        for uri, common_accuracy, alias_accuracy,  n_token in batchify(tokenizer, protocol, mapping, subset,
                                                                      batch_size=batch_size,
                                                                      window_size=window_size,
                                                                      step_size=step_size,
                                                                      mask=mask,
                                                                      easy=easy,
                                                                      sep_change=sep_change,
                                                                      augment=augment,
                                                                      shuffle=False,
                                                                      oracle=True):
            uris.append(uri)
            accuracies.append(common_accuracy)
            if alias_accuracy is not None:
                alias_accuracies.append(alias_accuracy)
            n_tokens.extend(n_token)
        n_tokens = f"{np.mean(n_tokens):.2f} $\\pm$ {np.std(n_tokens):.2f}"
        uris.append(full_name)
        accuracies.append(np.mean(accuracies))
        if alias_accuracies:
            alias_accuracies.append(np.mean(alias_accuracies))
        caption = (f"Oracle accuracy (word/batch-level), protocol {full_name}, "
                   f"Windows of {window_size} speaker turns with {step_size} step. "
                   f"Average \\# of words: {n_tokens}.")
        # print oracle accuracy
        print(tabulate(zip_longest(uris, accuracies, alias_accuracies, fillvalue='-'),
                       headers=('uri', 'common-accuracy', 'alias-accuracy'),
                       tablefmt='latex'))
        print("\\caption{%s}" % caption)

