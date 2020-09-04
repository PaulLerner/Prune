#!/usr/bin/env python
# encoding: utf-8

from typing import List

import numpy as np

from transformers import BertModel
from torch.nn import TransformerEncoder, TransformerEncoderLayer, Module, Linear, \
    Sigmoid, LayerNorm, Identity
from torch import Tensor


def total_params(module):
    """Beware to freeze the relevant parameters before computing this."""
    trainable, total = 0, 0
    for param in module.parameters():
        size = np.prod(param.size())
        total += size
        if param.requires_grad:
            trainable += size
    return trainable, total


class Identity(Identity):
    """Same as torch.nn.Identity but supports additional arguments
    as proposed in https://github.com/pytorch/pytorch/issues/42015
    """

    def forward(self, input: Tensor, *args, **kwargs) -> Tensor:
        return input


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
    out_size: `int`, optional
        Output size of the model.
        Defaults to 256.
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

    References
    ----------
    TODO
    """

    def __init__(self, bert='bert-base-cased', out_size=256, num_layers=6, nhead=8,
                 dim_feedforward=2048, dropout=0.1, activation='relu'):

        super().__init__()
        self.bert = BertModel.from_pretrained(bert)
        self.hidden_size = self.bert.config.hidden_size
        self.out_size = out_size
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
                                              encoder_norm)

        # handle classification layer and weight-tying
        self.linear = Linear(self.hidden_size, self.out_size)

        self.activation = Sigmoid()

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
            (batch_size, max_length, max_length). Similarity (e.g. 1 - cosine distance)
            between audio embeddings of words, aligned with target_ids.
            Defaults to None, indicating that the model should rely only on the text.
        src_key_padding_mask: Tensor, optional
            (batch_size, max_length). Used to mask input_ids.
            Defaults to None (no masking).

        Returns
        -------
        output: Tensor
            (batch_size, sequence_length, out_size)
            Model's scores after Sigmoid
        """
        # pass input text trough bert
        hidden_states = self.bert(input_ids, src_key_padding_mask)[0]

        # reshape BertModel output like (sequence_length, batch_size, hidden_size)
        # to fit torch.nn.Transformer
        hidden_states = hidden_states.transpose(0, 1)

        # convert HuggingFace mask to PyTorch mask
        #     HuggingFace: 1    -> NOT MASKED, 0     -> MASKED
        #     PyTorch:     True -> MASKED,     False -> NOT MASKED
        if src_key_padding_mask is not None:
            src_key_padding_mask = ~src_key_padding_mask.bool()

        text_output = self.encoder(hidden_states, mask=None,
                                   src_key_padding_mask=src_key_padding_mask)

        text_output = self.linear(text_output)

        # reshape output like (batch_size, sequence_length, out_size)
        text_output = text_output.transpose(0, 1)

        # weigh output using audio_similarity
        if audio_similarity is not None:
            audio_similarity = audio_similarity
            output = audio_similarity @ text_output
        else:
            output = text_output

        # activate with Sigmoid
        output = self.activation(output)
        return output

