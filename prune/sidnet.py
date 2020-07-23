#!/usr/bin/env python
# encoding: utf-8

from typing import List

import numpy as np

from transformers import BertModel
from torch.nn import TransformerEncoder, TransformerEncoderLayer, Module, Linear, \
    LogSoftmax, LayerNorm, Identity
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


def get_device(module):
    """Returns device of the first parameter of the module,
    or None if the module has no parameters
    """
    for p in module.parameters():
        return p.device
    return None


class SidNet(Module):
    """Named-Speaker Identification Network

    Identifies named-speakers in dialogues using Transformer Encoder's Self-attention

    Note that it's important that the target name is as written in the input text.
    Embeds input text using BERT (TODO make it more generic)
    then identifies speakers using a Transformer Encoder

    Decisions can be weighed using audio embeddings
    (similar voices should be tagged similarly)

    Parameters
    ----------
    bert: `str`, optional
        Model name or path, see BertTokenizer.from_pretrained
        Defaults to 'bert-base-cased'.
    audio: `Wrappable`, optional
        Describes how raw speaker embeddings should be obtained.
        See pyannote.audio.features.wrapper.Wrapper documentation for details.
        Defaults to None, indicating that the model should rely only on the text.
    num_layers: `int`, optional
        The number of sub-encoder-layers in the encoder.
        If set to 0 then self.encoder is Identity
        I.e. Bert output is fed directly to the Linear layer.
        Defaults to 6
    nhead: `int`, optional
        The number of heads in each encoder layer.
        Defaults to 8
    dim_feedforward: `int`, optional
        The dimension of the feedforward network model (FFN) in each encoder layer.
        Defaults to 2048
    dropout: `float`, optional
        The dropout rate in-between each block (attn or FFN) of each encoder layer
        Defaults to 0.1
    activation: `str`, optional
        The activation function of intermediate layer of the FFN: 'relu' or 'gelu'
         Defaults to 'relu'
    tie_weights: `bool`, optional
        Tie embedding and classification layer weights as in Press and Wolf, 2016.
        This prohibits the use of an additive bias in the classification layer (FIXME),
        at least because we don't want the weights and biases to be on 2 different devices.

    References
    ----------
    Press, O., Wolf, L., 2016.
    Using the output embedding to improve language models. arXivpreprint arXiv:1608.05859.
    """

    def __init__(self, bert='bert-base-cased', audio=None, num_layers=6, nhead=8,
                 dim_feedforward=2048, dropout=0.1, activation='relu', tie_weights=False):

        super().__init__()
        # put bert in the first device
        # and the encoder and output layer in the last (hopefully another) one
        # obviously, the output layer will end up on the first device if we're tying weights

        self.bert = BertModel.from_pretrained(bert).to(device=DEVICES[0])
        self.hidden_size = self.bert.config.hidden_size
        self.vocab_size = self.bert.config.vocab_size
        self.encoder_num_layers = num_layers
        # 0 encoder layers (=) feed BERT output directly to self.linear
        if self.encoder_num_layers == 0:
            self.encoder = Identity()
        # else init encoder as usual
        else:
            # init encoder_layer with the parameters
            encoder_layer = TransformerEncoderLayer(self.hidden_size, nhead, 
                                                    dim_feedforward,
                                                    dropout, activation)
            encoder_norm = LayerNorm(self.hidden_size)
            # init encoder with encoder_layer
            self.encoder = TransformerEncoder(encoder_layer, self.encoder_num_layers,
                                              encoder_norm).to(device=DEVICES[-1])

        # handle classification layer and weight-tying
        self.linear = Linear(self.hidden_size, self.vocab_size,
                             bias=not tie_weights).to(device=DEVICES[-1])
        if tie_weights:
            self.linear.weight = self.bert.embeddings.word_embeddings.weight

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

    def forward(self, input_ids, audio_similarity=None, src_key_padding_mask=None):
        """Apply model

        Parameters
        ----------
        input_ids: Tensor
            (batch_size, max_length). Encoded input tokens using BertTokenizer
        audio_similarity: Tensor, optional
            (batch_size, max_length, max_length). Similarity (e.g. cosine distance)
            between audio embeddings of words, aligned with target_ids.
            Defaults to None, indicating that the model should rely only on the text.
        src_key_padding_mask: Tensor, optional
            (batch_size, max_length). Used to mask input_ids.
            Defaults to None (no masking).

        Returns
        -------
        output: Tensor
            (batch_size, max_length). Model's hypothesis encoded like input_ids
        """
        # manage devices
        device_ = get_device(self.bert)
        input_ids = input_ids.to(device_)
        if src_key_padding_mask is not None:
            src_key_padding_mask = src_key_padding_mask.to(device_)
            
        # pass input text trough bert
        hidden_states = self.bert(input_ids, src_key_padding_mask)[0]

        # reshape BertModel output like (sequence_length, batch_size, hidden_size)
        # to fit torch.nn.Transformer and manage devices
        device_ = get_device(self.encoder)
        hidden_states = hidden_states.transpose(0, 1).to(device_)

        # convert HuggingFace mask to PyTorch mask and manage devices
        #     HuggingFace: 1    -> NOT MASKED, 0     -> MASKED
        #     PyTorch:     True -> MASKED,     False -> NOT MASKED
        if src_key_padding_mask is not None:
            src_key_padding_mask = ~src_key_padding_mask.bool().to(device_)

        text_output = self.encoder(hidden_states, mask=None,
                                   src_key_padding_mask=src_key_padding_mask)

        # manage devices
        device_ = get_device(self.linear)
        text_output = self.linear(text_output.to(device_))

        # reshape output like (batch_size, sequence_length, vocab_size)
        text_output = text_output.transpose(0, 1)

        # TODO weigh output using audio_similarity here
        if audio_similarity is not None:
            audio_similarity = audio_similarity.to(device_)

        # activate with LogSoftmax
        output = self.activation(text_output)
        return output

