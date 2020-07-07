#!/usr/bin/env python
# encoding: utf-8

from typing import List

import numpy as np

from transformers import BertModel
from torch.nn import Transformer, Module, Linear, LogSoftmax
from torch.cuda import device_count
from torch import device


DEVICES = [device('cpu')] if device_count() == 0 else \
          [device(f"cuda:{i}") for i in range(device_count())]


def total_params(module):
    """Beware to freeze the relevant parameters before computing this."""
    trainable, total = 0, 0
    for param in module.parameters():
        size = np.prod(param.size())
        total += size
        if param.requires_grad:
            trainable += size
    return trainable, total


class SidNet(Module):
    """Named-Speaker Identification Network

    Should be very similar to a neural translation system although we translate from:
        language -> speaker names
    Note that it's important that the target name is as written in the input text.
    Embeds input text using BERT (TODO make it more generic)
    then learn the sequence-to-sequence translation using a Transformer

    Decisions can be weighed using audio embeddings
    (similar voices should be tagged similarly)

    Parameters
    ----------
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
    **kwargs:
        Additional arguments are passed to self.seq2seq (torch.nn.Transformer)

    References
    ----------
    TODO
    """

    def __init__(self, bert='bert-base-cased', vocab_size=28996, audio=None, **kwargs):
        super().__init__()
        # put bert and output layer in the first device
        # and the seq2seq in the last (hopefully another) one
        self.bert = BertModel.from_pretrained(bert).to(device=DEVICES[0])
        self.hidden_size = self.bert.config.hidden_size
        self.vocab_size = vocab_size
        self.src_mask = None
        self.tgt_mask = None
        self.seq2seq = Transformer(d_model=self.hidden_size, **kwargs).to(device=DEVICES[-1])
        self.linear = Linear(self.hidden_size, self.vocab_size).to(device=DEVICES[0])
        self.activation = LogSoftmax(dim=2)

    def freeze(self, names: List[str]):
        """Freeze parts of the model

        Shamelessly stolen from pyannote.audio.train.model

        Parameters
        ----------
        names : list of string
            Names of modules to freeze.
        """
        for name, module in self.named_modules():
            if name in names:
                for parameter in module.parameters(recurse=True):
                    parameter.requires_grad = False

    def __str__(self):
        """Stringify model architecture along with trainable and total # of parameters"""
        lines = []
        for name, module in self.named_modules():
            if name == "":
                name = 'model'
                indent = ''
            else:
                indent = '    ' * (name.count('.') + 1)
                name = name.split('.')[-1]
            trainable, total = total_params(module)
            lines.append(f'{indent} {name} ({module.__class__.__name__}): '
                         f'{trainable:,d} ({total:,d} total)')
        return '\n'.join(lines)

    def forward(self, input_ids, target_ids,
                audio_similarity=None, src_key_padding_mask=None, tgt_key_padding_mask=None):
        """Apply model

        Parameters
        ----------
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

        Returns
        -------
        output: Tensor
            (batch_size, max_length). Model's hypothesis encoded like input_ids
        """
        # manage devices
        device_ = next(self.bert.parameters()).device
        input_ids, target_ids = input_ids.to(device_), target_ids.to(device_)
        if src_key_padding_mask is not None:
            src_key_padding_mask = src_key_padding_mask.to(device_)
            
        # pass input text trough bert
        hidden_states = self.bert(input_ids, src_key_padding_mask)[0]

        # embed targets using bert embeddings
        embedded_targets = self.bert.embeddings(target_ids)

        # reshape BertModel output like (sequence_length, batch_size, hidden_size)
        # to fit torch.nn.Transformer and manage devices
        device_ = next(self.seq2seq.parameters()).device
        hidden_states = hidden_states.transpose(0, 1).to(device_)
        embedded_targets = embedded_targets.transpose(0, 1).to(device_)

        # FIXME are all these masks done the right way ?
        if self.src_mask is None or self.src_mask.shape(0) != len(hidden_states):
            self.src_mask = self.seq2seq.generate_square_subsequent_mask(
                                len(hidden_states)).to(device_)
        if self.tgt_mask is None or self.tgt_mask.shape(0) != len(embedded_targets):
            self.tgt_mask = self.seq2seq.generate_square_subsequent_mask(
                                len(embedded_targets)).to(device_)
        # convert HuggingFace mask to PyTorch mask and manage devices
        #     HuggingFace: 1    -> NOT MASKED, 0     -> MASKED
        #     PyTorch:     True -> MASKED,     False -> NOT MASKED
        if src_key_padding_mask is not None:
            src_key_padding_mask = ~src_key_padding_mask.bool().to(device_)
        if tgt_key_padding_mask is not None:
            tgt_key_padding_mask = ~tgt_key_padding_mask.bool().to(device_)

        text_output = self.seq2seq(hidden_states, embedded_targets,
                                   src_mask=self.src_mask, tgt_mask=self.tgt_mask,
                                   src_key_padding_mask=src_key_padding_mask,
                                   tgt_key_padding_mask=tgt_key_padding_mask)
        # manage devices
        device_ = next(self.linear.parameters()).device
        text_output = self.linear(text_output.to(device_))

        # reshape output like (batch_size, sequence_length, vocab_size)
        text_output = text_output.transpose(0, 1)

        # TODO weigh output using audio_similarity here
        if audio_similarity is not None:
            audio_similarity = audio_similarity.to(device_)

        # activate with LogSoftmax
        output = self.activation(text_output)
        return output

