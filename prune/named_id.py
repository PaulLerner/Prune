#!/usr/bin/env python
# encoding: utf-8

"""Usage:
named_id.py train <protocol> <experiment_dir> [options] [--from=<epoch>] [(--augment=<ratio> [--uniform])]
named_id.py validate <protocol> <train_dir> [options] [--evergreen --interactive]
named_id.py test <protocol> <validate_dir> [options] [--interactive]
named_id.py oracle <protocol> [options]
named_id.py visualize <protocol> [<validate_dir>]

Common options:

--save=<save>        Path to load/save formatted data (windows and synthetic names)
                     Defaults to <experiment_dir> parent
--subset=<subset>    Protocol subset, one of 'train', 'development' or 'test'.
                     Defaults to 'train', 'development' and 'test' in
                     'train', 'validate', and 'test' mode, respectively.
--batch=<batch>      Batch size (# of windows) [default: 128]
--window=<window>    Window size (# of speaker turns) [default: 8]
--step=<step>        Step size (overlap between windows) [default: 1]
--max_len=<max_len>  Maximum # of tokens input to BERT. Maximum 512 [default: 256]
--easy               Only keep text windows with named speakers in it.

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
import pickle
import yaml
import warnings
from typing import List
from tabulate import tabulate
from itertools import zip_longest
from collections import Counter

from pyannote.core import Segment
from pyannote.database import get_protocol
from pyannote.audio.features.wrapper import Wrapper
import Plumcot as PC
from prune.utils import warn_deprecated

import re
import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

from torch import load, manual_seed, no_grad, argmax, Tensor, zeros, from_numpy, \
    zeros_like, LongTensor, ones, float, cat, BoolTensor, isnan
from torch import save as torch_save
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.nn import BCELoss, DataParallel
from transformers import BertTokenizer
from prune.sidnet import SidNet


# ignore torch DeprecationWarning which has been removed in later versions
warnings.filterwarnings("ignore", message="pickle support for Storage will be removed in 1.5", module="torch")

# set random seed
np.random.seed(0)
manual_seed(0)

EPOCH_FORMAT = '{:04d}.tar'
WINDOW_FORMAT = '{}.{:04d}.pickle'
BERT = 'bert-base-cased'
PROPN = 'PROPN'
WP_START='##'

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


def batch_word_accuracy(targets: List[np.ndarray], predictions: List[str],
                        pad='[PAD]', split=True, confidence=[[]]):
    """Computes word accuracy at the batch level

    Parameters
    ----------
    targets: List[np.ndarray[str]]
    predictions:
        - List[str] if split
        - List[List[str]] otherwise
    pad: str, optional
        special token to be ignored
        Defaults to '[PAD]'
    split : bool, optional
        Whether to split items in targets, predictions
        Defaults to True
    confidence : List[List[float]]
        Confidence of the model in it's prediction

    Returns
    -------
    accuracy: float
    correct_conf, wrong_conf: List[float]
        Confidence of the model in correct and wrong predictions, respectively.
    """
    correct, total = 0, 0
    correct_conf, wrong_conf = [], []
    for target, prediction, conf in zip_longest(targets, predictions, confidence, fillvalue=[]):
        if split:
            prediction = prediction.split()
        for t, p, c in zip_longest(target, prediction, conf, fillvalue=pad):
            if t == pad:
                continue
            if t == p:
                correct += 1
                if c != pad:
                    correct_conf.append(c)
            elif c != pad:
                wrong_conf.append(c)
            total += 1
    if total == 0:
        return None
    return correct/total, correct_conf, wrong_conf


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
        if not token.startswith(WP_START):
            i += 1
    inp_eg.append("[UNK_SPK]")
    max_len = max_length+1
    plt.figure(figsize=(max_len//6, max_len//6))
    # shift by 1 to discard [CLS] and [SEP] tokens
    plt.imshow(output_eg.detach().cpu().numpy())
    plt.colorbar()
    plt.xticks(range(max_len), inp_eg[:max_len], fontsize='x-small', rotation='vertical')
    plt.yticks(range(max_len), merge[:max_len], fontsize='x-small', rotation='horizontal')
    if save is None:
        plt.show()
    else:
        plt.savefig(save/"output.png")
    plt.close()


def plot_accuracy(rights, wrongs, save=None, ylabel='Accuracy', xlabel='Confidence'):
    bins = 50
    n_wrongs, bins, _ = plt.hist(wrongs, bins=bins)
    n_rights, _, _ = plt.hist(rights, bins=bins)
    totals = n_rights+n_wrongs
    plt.close()
    plt.figure(figsize=(16, 10))
    plt.scatter(bins[1:], n_rights / totals,
                linewidths=totals / totals.mean(), alpha=.5)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    if save:
        plt.savefig(save)
    else:
        plt.show()
    plt.close()


def mode(prediction, pad='[PAD]'):
    """Mode of a Counter instance

    Parameters
    ----------
    prediction: dict[int, Counter]
    pad: str, optional
        Return value if prediction is empty
        Defaults to '[PAD]'

    Returns
    -------
    mode: str
        Most common item of prediction, or pad if prediction is empty
    confidence: float
        Ratio of mode in prediction
    """
    if prediction['scores']:
        mode, count = prediction['scores'].most_common(1)[0]
        return mode, count/prediction['total']
    return pad, 0.


def eval(batches_parameters, model, tokenizer, log_dir,
         test=False, evergreen=False, interactive=False, step_size=1, window_size=10):
    """Load model from checkpoint and evaluate it on batches.
    When testing, only the best model should be tested.

    Parameters
    ----------
    batches_parameters: dict
        should be passed to batchify(**batches_parameters)
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
    else:
        weights_path = log_dir.parents[0] / 'weights'
        weights = sorted(weights_path.iterdir(), reverse=evergreen)
        if params_file.exists():
            with open(params_file) as file:
                best = yaml.load(file, Loader=yaml.SafeLoader)["accuracy"]
        else:
            best = 0.
    # load batches before-hand since we shouldn't need augmentation (make this optional to ease on the RAM ?)
    batches = list(tqdm(batchify(**batches_parameters), desc='Loading batches'))

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
            epoch_loss, epoch_word_acc, epoch_alias_acc = [], [], 0.
            correct_confs, wrong_confs, file_correct_confs, file_wrong_confs = [], [], [], []
            uris, file_word_accs, file_alias_accs = [], [], []
            previous_uri = None
            for batch in batches:
                # unpack tensors from batch
                text_id_window, target_id_window, audio_window, audio_mask_window, src_key_padding_mask, tgt_key_padding_mask = \
                    get_tensors(**batch)
                # unpack text from batch
                uri, speaker_turns, aliases, text_window, target_window, speaker_id_window = get_text(**batch)

                # forward pass: (batch_size, sequence_length, sequence_length)
                output = model(text_id_window, audio_window, src_key_padding_mask, audio_mask_window)
                # manage devices
                target_id_window = target_id_window.to(output.device)

                # get model prediction per token: (batch_size, sequence_length)
                # trim output to max_length to discard unknown speakers and get output w.r.t. input
                confidence, relative_out = output[:, :, :max_length].max(dim=2)
                # retrieve token ids from input (batch_size, sequence_length) and manage device
                prediction_ids = zeros_like(text_id_window, device=output.device)
                for j, (input_window_id, relative_window_out) in enumerate(zip(text_id_window, relative_out)):
                    prediction_ids[j] = input_window_id[relative_window_out]

                # decode and compute word accuracy
                predictions, aligned_confidence = [], []
                for j, prediction_id in enumerate(prediction_ids):
                    aligned_confidence_j = []
                    tokens = tokenizer.convert_ids_to_tokens(prediction_id)
                    for k, token in enumerate(tokens):
                        # filter out sub-words
                        if not token.startswith(WP_START):
                            aligned_confidence_j.append(confidence[j, k].item())
                    aligned_confidence.append(aligned_confidence_j)
                    predictions.append(tokenizer.convert_tokens_to_string(tokens))
                batch_word_acc, correct_conf, wrong_conf = batch_word_accuracy(target_window,
                                                                               predictions,
                                                                               tokenizer.pad_token,
                                                                               confidence=aligned_confidence)
                # handle fully-padded batches
                if batch_word_acc is not None:
                    epoch_word_acc.append(batch_word_acc)
                    correct_confs.append(correct_conf)
                    wrong_confs.append(wrong_conf)

                # compute alias accuracy
                if aliases:
                    epoch_alias_acc += speaker_alias_accuracy(speaker_id_window, predictions, aliases,
                                                              pad=tokenizer.pad_token)

                # calculate loss
                loss = criterion(output, target_id_window)
                loss = reduce_loss(loss, tgt_key_padding_mask)
                # handle fully-padded batches
                if not isnan(loss):
                    epoch_loss.append(loss.item())

                # handle file-level stuff
                if uri != previous_uri:
                    # compute file-level accuracy
                    if previous_uri is not None:
                        uris.append(previous_uri)
                        # merge window-level predictions
                        tmp, file_predictions = file_predictions, []
                        for p in tmp:
                            m, c = mode(p, tokenizer.pad_token)
                            file_predictions.append(m)
                            file_confidence.append(c)
                        # compute word accuracy
                        file_word_acc, file_correct_conf, file_wrong_conf = batch_word_accuracy(
                                                                                [file_target],
                                                                                [file_predictions],
                                                                                pad=tokenizer.pad_token,
                                                                                split=False,
                                                                                confidence=[file_confidence])
                        file_word_accs.append(file_word_acc)
                        file_correct_confs += file_correct_conf
                        file_wrong_confs += file_wrong_conf

                        # compute speaker alias accuracy
                        if aliases:
                            file_alias_accs.append(speaker_alias_accuracy([file_speaker_id],
                                                                          [file_predictions],
                                                                          aliases,
                                                                          pad=tokenizer.pad_token,
                                                                          split=False))
                        # TODO audio ER

                    # reset file-level variables
                    shift = 0
                    file_target, file_speaker_id, file_predictions, file_confidence = [], [], []

                i = 0
                # save target and output for future file-level accuracy
                for target_i, pred_i, speaker_id_i, speaker_turn_i, conf_i in zip_longest(
                                                                                  target_window,
                                                                                  predictions,
                                                                                  speaker_id_window,
                                                                                  speaker_turns,
                                                                                  aligned_confidence):
                    pred_i = pred_i.split()
                    target_i = list(target_i)
                    for start, end in speaker_turn_i:
                        if len(file_target) < end:
                            file_target += target_i[start-shift: end-shift]
                        if speaker_id_i is not None:
                            if len(file_speaker_id) < end:
                                file_speaker_id += speaker_id_i[start-shift: end-shift]
                        for j, (p, c) in enumerate(zip(
                                                    pred_i[start-shift: end-shift],
                                                    conf_i[start-shift: end-shift])):
                            if len(file_predictions) < start+j+1:
                                first_prediction = {'scores': Counter({p: c}),
                                                    'total': 1}
                                file_predictions.append(first_prediction)
                            else:
                                file_predictions[start+j]['scores'][p] += c
                                file_predictions[start + j]['total'] += 1
                    # shift between batch and original file
                    shift = speaker_turn_i[i][0]  # start
                    i += step_size
                if interactive:
                    eg = np.random.randint(len(target_window))
                    inp_eg, tgt_eg, pred_eg = text_window[eg], target_window[eg], predictions[eg]
                    # print random example
                    print(str_example(inp_eg, tgt_eg, pred_eg.split()))
                    # plot model output
                    plot_output(output[eg], tokenizer.tokenize(inp_eg), tgt_eg, log_dir)

                    # print current metrics
                    metrics = {
                        'Loss/eval': [np.mean(epoch_loss)],
                        'Accuracy/eval/batch/word': [format_acc(np.mean(epoch_word_acc))],
                        'Accuracy/eval/batch/speaker_alias': [format_acc(epoch_alias_acc)]
                    }
                    print(tabulate(metrics, headers='keys', disable_numparse=True))
                    breakpoint()

                previous_uri = uri

            # compute file-level accuracy for the last file
            uris.append(previous_uri)
            # merge window-level predictions
            tmp, file_predictions = file_predictions, []
            for p in tmp:
                m, c = mode(p, tokenizer.pad_token)
                file_predictions.append(m)
                file_confidence.append(c)
            # compute word accuracy
            file_word_acc, file_correct_conf, file_wrong_conf = batch_word_accuracy(
                [file_target],
                [file_predictions],
                pad=tokenizer.pad_token,
                split=False,
                confidence=[file_confidence])
            file_word_accs.append(file_word_acc)
            file_correct_confs += file_correct_conf
            file_wrong_confs += file_wrong_conf
            # compute speaker alias accuracy
            if aliases:
                file_alias_accs.append(speaker_alias_accuracy([file_speaker_id],
                                                              [file_predictions],
                                                              aliases,
                                                              pad=tokenizer.pad_token,
                                                              split=False))
            # average file-accuracies
            uris.append('TOTAL')
            file_word_accs.append(np.mean(file_word_accs))
            if file_alias_accs:
                file_alias_accs.append(np.mean(file_alias_accs))

            # log tensorboard
            tb.add_scalar('Accuracy/eval/file/word', file_word_acc[-1], epoch)
            epoch_loss = np.mean(epoch_loss)
            tb.add_scalar('Loss/eval', epoch_loss, epoch)
            epoch_word_acc = np.mean(epoch_word_acc)
            tb.add_scalar('Accuracy/eval/batch/word', epoch_word_acc, epoch)
            epoch_alias_acc /= len(batches)
            tb.add_scalar('Accuracy/eval/batch/speaker_alias', epoch_alias_acc, epoch)

            # format file-accuracies in % with .2f
            file_word_accs = [format_acc(acc) for acc in file_word_accs]
            file_alias_accs = [format_acc(acc) for acc in file_alias_accs]

            # write metrics and visualize
            if test:
                # print and write metrics
                epoch_alias_acc = format_acc(epoch_alias_acc) if aliases else "-"
                metrics = {
                    'Loss/eval': [epoch_loss],
                    'Accuracy/eval/batch/word': [format_acc(epoch_word_acc)],
                    'Accuracy/eval/batch/speaker_alias': [epoch_alias_acc]
                }
                metrics = tabulate(metrics, headers='keys', tablefmt='latex', disable_numparse=True)
                metrics += tabulate(zip_longest(uris, file_word_accs, file_alias_accs, fillvalue='-'),
                                    headers=['uri', 'word-level', 'alias'],
                                    tablefmt='latex',
                                    disable_numparse=True)
                print(metrics)
                with open(log_dir / 'eval', 'w') as file:
                    file.write(metrics)

                # visualize accuracy w.r.t window-level confidence
                plot_accuracy(correct_confs, wrong_confs,
                              log_dir/"accuracy_wrt_window_confidence.png")
                # visualize accuracy w.r.t file-level confidence (in majority voting)
                plot_accuracy(file_correct_confs, file_wrong_confs,
                              log_dir/"accuracy_wrt_file_majority_vote_confidence.png")

            # dump best metrics
            elif epoch_word_acc > best:
                best = float(epoch_word_acc)
                with open(log_dir / 'params.yml', 'w') as file:
                    yaml.dump({"accuracy": best, "epoch": epoch}, file)


def reduce_loss(loss, tgt_key_padding_mask):
    """Masks loss using tgt_key_padding_mask then mean-reduce"""
    # mask and average loss
    return loss[tgt_key_padding_mask.bool()].mean()


def get_tensors(text_id_window=None, target_id_window=None, audio_window=None, audio_mask_window=None,
                src_key_padding_mask=None, tgt_key_padding_mask=None, **kwargs):
    """
    Returns a tuple:
    ---------------
    text_id_window
    target_id_window
    audio_window
    audio_mask_window
    src_key_padding_mask
    tgt_key_padding_mask

    See batchify

    Additional parameters **kwargs are not used
    """
    return text_id_window, target_id_window, audio_window, audio_mask_window, src_key_padding_mask, tgt_key_padding_mask


def get_text(uri=None, speaker_turn_window=None, aliases=None,
             text_window=None, target_window=None, speaker_id_window=None, **kwargs):
    return uri, speaker_turn_window, aliases, text_window, target_window, speaker_id_window


def train(batches_parameters, model, tokenizer, train_dir=Path.cwd(),
          lr=1e-3, max_grad_norm=None,
          epochs=100, freeze=['bert'], save_every=1, start_epoch=None):
    """Train the model for `epochs` epochs

    Parameters
    ----------
    batches_parameters: dict
        should be passed to batchify(**batches_parameters)
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
        checkpoint = load(weights_path / EPOCH_FORMAT.format(start_epoch), map_location=model.src_device_obj)
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
        # load batches
        batches = batchify(**batches_parameters)

        epoch_loss = 0.
        for b, batch in enumerate(batches):
            optimizer.zero_grad()
            text_id_window, target_id_window, audio_window, audio_mask_window, src_key_padding_mask, tgt_key_padding_mask = \
                get_tensors(**batch)
            # forward pass
            output = model(text_id_window, audio_window, src_key_padding_mask, audio_mask_window)
            # manage devices
            target_id_window = target_id_window.to(output.device)

            # calculate loss
            loss = criterion(output, target_id_window)
            # mask and reduce loss
            loss = reduce_loss(loss, tgt_key_padding_mask)
            loss.backward()

            if max_grad_norm is not None:
                clip_grad_norm_(model.module.parameters(), max_grad_norm)

            optimizer.step()
            epoch_loss += loss.item()

        tb.add_scalar('Loss/train', epoch_loss/b, epoch)
        tb.add_scalar('lr', lr, epoch)

        if (epoch+1) % save_every == 0:
            torch_save({
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


def encode(string, tokenizer):
    return tokenizer.encode(string, return_tensors='pt',
                            is_pretokenized=False, add_special_tokens=False)


def format_transcription(current_file, tokenizer, mapping, audio_emb=None, window_size=10,
                         step_size=1, easy=False, oracle=False):
    """
    1. tokenize and tensorize transcription such that
       we have a 1-1 mapping between input tokens and targets
    2. segment transcription in speaker turns
    3. Divide transcription in windows

    Parameters
    ----------
    current_file: ProtocolFile
        as provided by pyannote Protocol

    See batchify


    Yields
    ------
    window: dict
    {
        uri, str
            File identifier
        speaker_turn_window, List[Tuple[int]]
            Indices of the speaker turns in the windows
        aliases, dict
            Aliases of targets: {str: Set[str]}
        text_window, np.ndarray[str]
            list of input text
        target_window, np.ndarray[str]
            list of target names
        text_id_window, List[Tensor]
            list of tokenized input text
        target_id_window, List[Tensor]
            list of tokenized target names
        speaker_id_window, List[str]
            list of speaker identifiers
        audio_window, Tensor
            Tensor of variable length of audio embeddings
        audio_mask_window, Tensor
            Same length as audio_window.
            When to mask audio_window (see torch.nn.Transformer)
    }

    Note
    ----
    There should be a 1-1 mapping between every *_window
    """
    transcription = current_file.get('entity', current_file['transcription'])
    uri = current_file['uri']

    if audio_emb is not None:
        audio_emb = Wrapper(audio_emb)

    current_audio_emb = None
    # get audio embeddings from current_file
    if audio_emb is not None:
        current_audio_emb = audio_emb(current_file)

    # format transcription into 5 lists: tokens, audio, audio_masks, targets, speaker_ids
    # and segment it in speaker turns (i.e. speaker homogeneous)
    speaker_turns = []
    aliases = {}
    start, end = 0, 0
    tokens, audio, audio_masks, targets, speaker_ids = [], [], [], [], []
    token_ids, target_ids = [], []
    previous_speaker = None
    for word in transcription:
        # segment in speaker turns
        if word._.speaker != previous_speaker:
            speaker_turns.append((start, end))
            start = end
        previous_speaker = word._.speaker
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
        # in order to align input text and targets
        basic_token = tokenizer.basic_tokenizer.tokenize(word.text)
        for token in basic_token:
            tokens.append(token)
            targets.append(target)
            token_ids.append(encode(token, tokenizer))
            target_ids.append(encode(target, tokenizer))
            speaker_ids.append(word._.speaker)
            if audio_emb is not None:
                if segment is None:
                    audio.append(zeros(1, audio_emb.dimension, dtype=float))
                    audio_masks.append(True)
                else:
                    audio.append(Tensor(segment).unsqueeze(0))
                    audio_masks.append(False)
            end += 1

        if proper_entity(word):
            aliases.setdefault(word.ent_kb_id_, set())
            # HACK: there might be some un-basic-tokenized words in there such as "Angelas's"
            # so we keep only the first basic-token
            # TODO fix this upstream
            aliases[word.ent_kb_id_].add(basic_token[0])
    speaker_turns.append((start, end))
    speaker_turns.pop(0)

    # turn audio and audio_masks to Tensors
    if len(audio) != 0:
        audio, audio_masks = cat(audio, dim=0), BoolTensor(audio_masks)
    else:
        audio, audio_masks = None, None

    # slide through the transcription speaker turns w.r.t. window_size, step_size
    # filter out windows w.r.t. easy
    for i in range(0, len(speaker_turns) - window_size + 1, step_size):
        start, _ = speaker_turns[i]
        _, end = speaker_turns[i + window_size - 1]
        text_window = np.array(tokens[start:end])
        target_window = np.array(targets[start:end])
        # compute oracle-accuracy
        if oracle:
            raise NotImplementedError("oracle")
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

        # easy mode -> Only keep windows with named speakers in it
        if easy and not any_in_text(np.unique(target_window), text_window):
            continue
        window = {
            "uri": uri,
            "speaker_turn_window": speaker_turns[i: i + window_size],
            "aliases": aliases,
            "text_window": text_window,
            "target_window": target_window,
            "text_id_window": token_ids[start: end],
            "target_id_window": target_ids[start: end],
            "speaker_id_window": speaker_ids[start: end],
            "audio_window": audio[start: end] if audio is not None else None,
            "audio_mask_window": audio_masks[start: end] if audio is not None else None
        }
        yield window


def save_load_windows(save_windows, shuffle, protocol, subset, *args, **kwargs):
    """Either save or load windows according to save_windows
    Then yields them in random or sorted order according to shuffle

    Parameters
    ----------
    see batchify
    *args, **kwargs: Any
        additional arguments are passed to format_transcription

    Yields
    -------
    window: dict
        See format_transcription
    """
    # load windows from save according to shuffle
    if save_windows.exists():
        window_paths = list(save_windows.iterdir())
        if shuffle:
            np.random.shuffle(window_paths)
        else:
            window_paths.sort()
        for window_path in window_paths:
            with open(window_path, "rb") as file:
                window = pickle.load(file)
            yield window
    # iterate over protocol subset and save windows
    else:
        save_windows.mkdir()
        for current_file in tqdm(getattr(protocol, subset)(), desc='Loading transcriptions'):
            uri = current_file['uri']
            for i, window in enumerate(format_transcription(current_file, *args, **kwargs)):
                with open(save_windows / WINDOW_FORMAT.format(uri, i), 'wb') as file:
                    pickle.dump(window, file)
                # yield window directly in sorted order
                if not shuffle:
                    yield window
        # else recurrent call to load windows in random order
        if shuffle:
            for window in save_load_windows(save_windows, shuffle, protocol, subset, *args, **kwargs):
                yield window


def augment_window(uri=None, speaker_turn_window=None, aliases=None, text_window=None, target_window=None,
                   text_id_window=None, target_id_window=None,
                   speaker_id_window=None, audio_window=None, audio_mask_window=None, augment=None, names=None):
    """
    Augment data by replacing target names in text_window and target_window by a synthetic name from names

    Parameters
    ----------
    See batchify and format_transcription

    Returns
    -------
    synthetic_window, see format_transcription
    """
    for _ in range(abs(augment)):
        # copy original sample
        synthetic_text_id, synthetic_target_id = text_id_window.copy(), target_id_window.copy()
        # iterate over unique targets (str)
        for target in np.unique(target_window):
            # skip padded targets
            if target == tokenizer.pad_token:
                continue
            # pick random name
            i = np.random.randint(0, len(names))
            random_name = names[i]
            # replace in text (ids)
            for j in (text_window == target).nonzero()[0]:
                synthetic_text_id[j] = random_name
            # replace in targets (ids)
            for j in (target_window == target).nonzero()[0]:
                synthetic_target_id[j] = random_name
        yield dict(uri=uri, speaker_turn_window=speaker_turn_window, aliases=aliases,
                   text_window=text_window, target_window=target_window,
                   text_id_window=synthetic_text_id, target_id_window=synthetic_target_id,
                   speaker_id_window=speaker_id_window, audio_window=audio_window, audio_mask_window=audio_mask_window)


def handle_augmentation(tokenizer, protocol, mapping, subset='train', audio_emb=None, batch_size=128,
                        window_size=10, step_size=1, mask=True, easy=False, sep_change=False,
                        augment=0, uniform=False, shuffle=True, oracle=False, save=Path.cwd()):
    """

    Parameters
    ----------
    See batchify

    Yields
    -------
    Window according to shuffle and augment

    See Also
    --------
    save_load_windows and augment_window
    """
    arguments = {
        'tokenizer': BERT,
        'protocol': f'{protocol_name}.{subset}',
        'audio': bool(audio_emb),
        'window_size': window_size,
        'step_size': step_size,
        'easy': easy
    }
    arguments_str = ','.join(['_'.join(map(str, item))
                              for item in sorted(arguments.items())])
    save.mkdir(exist_ok=True)
    save_windows = save / arguments_str
    save_synthetic_names = save / f'synthetic_names_{"uniform" if uniform else "weighted"}.pickle'

    # load list of names (either from save_synthetic_names or CHARACTERS_PATH)
    names = None
    if augment != 0:
        if save_synthetic_names.exists():
            with open(save_synthetic_names, 'rb') as file:
                names = pickle.load(file)
        else:
            names = []
            for character_file in CHARACTERS_PATH:
                with open(character_file) as file:
                    for line in file.read().split("\n"):
                        if line != '':
                            names.append(line.split(',')[3].split()[0])
            if uniform:
                names = set(names)
            # encode name afterwards to be able to convert to set if uniform
            names = [encode(name, tokenizer) for name in names]
            with open(save_synthetic_names, 'wb') as file:
                pickle.dump(names, file)

    for window in save_load_windows(save_windows, shuffle, protocol, subset, tokenizer,
                                    mapping, audio_emb, window_size, step_size, easy, oracle):
        # augment < 0 ==> discard original sample
        if not augment < 0:
            yield window
        # FIXME: original and augmented windows will be next to each other
        for other_window in augment_window(**window, augment=augment, names=names):
            yield other_window


def batchify(tokenizer, protocol, mapping, subset='train', audio_emb=None, batch_size=128,
             window_size=10, step_size=1, mask=True, easy=False, sep_change=False,
             augment=0, uniform=False, shuffle=True, oracle=False, save=Path.cwd()):
    """
    A. handle_augmentation
        1. Iterates over protocol subset and format_transcription (or load from save)
            a. tokenize and tensorize transcription such that
               we have a 1-1 mapping between input tokens and targets
            b. segment transcription in speaker turns
            c. Divide transcription in windows
        2. (augment data if needed)
    B. reshape_window
        1. reshape text and targets to -1 and align audio with text
        2. add [CLS] and [SEP] tokens at the start-end of each text window
        3. compute relative targets
        4. pad windows to max_length and compute mask accordingly
    C. split windows in batches

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
    save: Path, optional
        Where to save/load from windows (formatted in format_transcription)
        Appends stringified arguments then:
            - loads if path exists
            - else save windows there

    Yields
    -------
    batch: dict
    {
        uri, List[str]
            File identifier
        speaker_turn_window, List[List[Tuple[int]]]
            Indices of the speaker turns in the windows
        aliases, List[dict]
            Aliases of targets: {str: Set[str]}
        text_window, List[np.ndarray[str]]
            list of input text
        target_window, List[np.ndarray[str]]
            list of target names
        text_id_window, Tensor[batch_size, max_length]
            Input word-pieces
        target_id_window, Tensor[batch_size, max_length, max_length]
            one-hot target index w.r.t. text_id_window
            e.g. "My name is Paul ." -> one-hot([3, 3, 3, 3, 3])
        speaker_id_window, List[str]
            list of speaker identifiers
            (typically "firstname_lastname") as opposed to
            target_id_window which is the speaker most common name (typically "firstname")
        audio_window, Tensor[batch_size, max_length]
            Tensor of audio embeddings, aligned with text_id_window
        audio_mask_window, Tensor[batch_size, max_length]
            When to mask audio_window (see torch.nn.Transformer)
        src_key_padding_mask, Tensor[batch_size, max_length]
            see BertTokenizer.batch_encode_plus
        tgt_key_padding_mask, Tensor[batch_size, max_length]
            see BertTokenizer.batch_encode_plus
    }
    """
    assert not tokenizer.do_basic_tokenize, "Basic tokenization is handle beforehand"
    warn_deprecated([(mask, 'mask'), (sep_change, 'sep_change')])
    batch_save = []
    previous_uri = None
    # split windows in batches
    for window in handle_augmentation(tokenizer, protocol, mapping, subset, audio_emb, batch_size,
                                      window_size, step_size, mask, easy, sep_change,
                                      augment, uniform, shuffle, oracle, save):
        # reshape text and targets and align audio with text
        window = reshape_window(tokenizer, **window)
        # discard fully-padded when training (i.e. shuffling)
        if shuffle and not window["tgt_key_padding_mask"].bool().any():
            continue
        uri = window['uri']
        # when not shuffling: yield uri-homogeneous batches (i.e. of the same file)
        # even if batch-size is lesser than the requested batch_size
        if not shuffle and previous_uri is not None and uri != previous_uri and len(batch_save) != 0:
            yield cat_window(batch_save)
            batch_save = []
        batch_save.append(window)
        previous_uri = uri
        # yield batch of size batch_size
        if len(batch_save) >= batch_size:
            yield cat_window(batch_save)
            batch_save = []
    # yield any remaining leftovers...
    if len(batch_save) != 0:
        yield cat_window(batch_save)


def cat_window(batch_save):
    """Concatenate a list of windows (see reshape_window) into a single batch:
    - Tensors are concatenated to a single tensor along the first dimension
    - Lists and np arrays are concatenated in lists
    - Only one value of str (uri), dict (aliases) or NoneType (maybe audio) is kept arbitrarily.

    Returns
    -------
    batch: dict
        See batchify
    """
    batch = {}
    for w in batch_save:
        for key, value in w.items():
            if isinstance(value, Tensor):
                batch.setdefault(key, [])
                batch[key].append(value.unsqueeze(0))
            elif isinstance(value, (np.ndarray, list)):
                batch.setdefault(key, [])
                batch[key].append(value)
            elif isinstance(value, (str, dict)) or value is None:
                batch[key] = value
            else:
                raise TypeError(f"Unexpected type in window: '{type(value)}'.\n From '{key}': {value}")

    for key, value in batch.items():
        if isinstance(value, list) and isinstance(value[0], Tensor):
            batch[key] = cat(value, dim=0)
    return batch


def reshape_window(tokenizer, uri=None, speaker_turn_window=None, aliases=None, text_window=None, target_window=None,
                   text_id_window=None, target_id_window=None,
                   speaker_id_window=None, audio_window=None, audio_mask_window=None):
    """
    1. reshape text and targets to -1 and align audio with text
    2. add [CLS] and [SEP] tokens at the start-end of each text window
    3. compute relative targets
    4. pad windows to max_length and compute mask accordingly

    Parameters
    ----------
    See format_transcription output

    Returns
    -------
    window: dict
    {
        uri, str
            File identifier
        speaker_turn_window, List[Tuple[int]]
            Indices of the speaker turns in the windows
        aliases, dict
            Aliases of targets: {str: Set[str]}
        text_window, np.ndarray[str]
            list of input text
        target_window, np.ndarray[str]
            list of target names
        text_id_window, Tensor[max_length]
            Input word-pieces
        target_id_window, Tensor[max_length, max_length+1]
            one-hot target index w.r.t. text_id_window
            e.g. "My name is Paul ." -> one-hot([3, 3, 3, 3, 3])
            the extra unit is for unknown speakers (when speaker name in not in the input)
        speaker_id_window, List[str]
            list of speaker identifiers
        audio_window, Tensor[max_length]
            Tensor of audio embeddings, aligned with text_id_window
        audio_mask_window, Tensor[max_length]
            When to mask audio_window (see torch.nn.Transformer)
        src_key_padding_mask, Tensor[max_length]
            see BertTokenizer.batch_encode_plus
        tgt_key_padding_mask, Tensor[max_length, max_length+1]
            see BertTokenizer.batch_encode_plus
            the extra unit is for unknown speakers (when speaker name in not in the input)
    }
    """
    # get [CLS] and [SEP] tokens
    cls_token_id, sep_token_id = LongTensor([tokenizer.cls_token_id]), LongTensor([tokenizer.sep_token_id])
    flat_text = []
    if audio_window is not None:
        aligned_audio = zeros((max_length, audio_window.shape[1]), dtype=float)
        aligned_audio_mask = ones(max_length, dtype=bool)
    else:
        aligned_audio = None
        aligned_audio_mask = None
    # align audio with text word-pieces, flatten text word-pieces and add special tokens [CLS] and [SEP]
    j = 1
    for i, text in enumerate(text_id_window):
        if aligned_audio is not None:
            aligned_audio[j: j+text.shape[1]] = audio_window[i].unsqueeze(0)
            aligned_audio_mask[j: j+text.shape[1]] = audio_mask_window[i]
        flat_text.append(text.reshape(-1))
        j += text.shape[1]
        if j >= max_length-1:
            break
    # add special tokens and trim to max_length
    flat_text = cat([cls_token_id] + flat_text + [sep_token_id])[: max_length]

    # flatten target and trim to max_length
    target_id_window = cat([target.reshape(-1) for target in target_id_window])[: max_length]

    # compute relative targets
    relative_targets = zeros(max_length, max_length+1)
    tgt_key_padding_mask = zeros(max_length, max_length+1, dtype=int)
    for j, t in enumerate(target_id_window):
        if t == tokenizer.pad_token_id:
            continue
        where = flat_text == t
        # speaker name is not mentioned in input -> target unknown speaker (last item)
        if not where.any():
            where = -1
        # else target mentions index
        else:
            where = where.nonzero().reshape(-1)
        # un-mask target
        tgt_key_padding_mask[j, where] = 1
        relative_targets[j, where] = 1.

    # pad input to max length and compute mask accordingly
    src_key_padding_mask = ones(max_length, dtype=int)
    src_key_padding_mask[flat_text.shape[0]:] = 0
    padding = zeros(max_length-flat_text.shape[0], dtype=int)
    flat_text = cat((flat_text, padding))

    # return updated window
    return dict(uri=uri, speaker_turn_window=speaker_turn_window, aliases=aliases,
                text_window=text_window, target_window=target_window,
                text_id_window=flat_text, target_id_window=relative_targets,
                speaker_id_window=speaker_id_window, audio_window=aligned_audio, audio_mask_window=aligned_audio_mask,
                src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)


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
    sep_change = False
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
        save = Path(args['--save']) if args['--save'] else train_dir.parents[1]
        config = load_config(train_dir.parents[0])
        architecture = config.get('architecture', {})
        audio = config.get('audio')
        model = DataParallel(SidNet(BERT, max_length+1, **architecture))
        # set batches parameters
        batches_parameters = dict(tokenizer=tokenizer, protocol=protocol, mapping=mapping, subset=subset,
                                  audio_emb=audio, batch_size=batch_size, window_size=window_size, step_size=step_size,
                                  mask=mask, easy=easy, sep_change=sep_change, augment=augment, uniform=uniform,
                                  shuffle=True, save=save)
        model, optimizer = train(batches_parameters, model, tokenizer, train_dir,
                                 start_epoch=start_epoch,
                                 **config.get('training', {}))
    elif args['validate']:
        subset = args['--subset'] if args['--subset'] else 'development'
        evergreen = args['--evergreen']
        interactive = args['--interactive']
        validate_dir = Path(args['<train_dir>'], f'{protocol_name}.{subset}')
        validate_dir.mkdir(exist_ok=True)
        save = Path(args['--save']) if args['--save'] else validate_dir.parents[2]
        config = load_config(validate_dir.parents[1])

        architecture = config.get('architecture', {})
        audio = config.get('audio')
        model = DataParallel(SidNet(BERT, max_length+1, **architecture))

        # set batches parameters
        batches_parameters = dict(tokenizer=tokenizer, protocol=protocol, mapping=mapping, subset=subset,
                                  audio_emb=audio, batch_size=batch_size, window_size=window_size, step_size=step_size,
                                  mask=mask, easy=easy, sep_change=sep_change, augment=augment, uniform=uniform,
                                  shuffle=False, save=save)
        eval(batches_parameters, model, tokenizer, validate_dir,
             test=False, evergreen=evergreen, interactive=interactive,
             step_size=step_size, window_size=window_size)

    elif args['test']:
        subset = args['--subset'] if args['--subset'] else 'test'
        interactive = args['--interactive']
        test_dir = Path(args['<validate_dir>'], f'{protocol_name}.{subset}')
        test_dir.mkdir(exist_ok=True)
        save = Path(args['--save']) if args['--save'] else test_dir.parents[3]
        config = load_config(test_dir.parents[2])

        architecture = config.get('architecture', {})
        audio = config.get('audio')
        model = DataParallel(SidNet(BERT, max_length+1, **architecture))

        # set batches parameters
        batches_parameters = dict(tokenizer=tokenizer, protocol=protocol, mapping=mapping, subset=subset,
                                  audio_emb=audio, batch_size=batch_size, window_size=window_size, step_size=step_size,
                                  mask=mask, easy=easy, sep_change=sep_change, augment=augment, uniform=uniform,
                                  shuffle=False, save=save)

        eval(batches_parameters, model, tokenizer, test_dir,
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

