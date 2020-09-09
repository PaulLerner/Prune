#!/usr/bin/env python
# encoding: utf-8

from typing import List

import numpy as np

from transformers import BertModel
from torch.nn import TransformerDecoder, TransformerDecoderLayer, Module, Linear, \
    Sigmoid, LayerNorm, Embedding
from torch import Tensor, arange


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

    Identifies named-speakers in dialogues using Self-attention:
    Embeds input text using BERT (TODO make it more generic)
    then identifies speakers using either:
    - a simple linear decoder
    - a multimodal Transformer decoder (taking embedded audio and text as input)

    Note that it's important that the target name is as written in the input text.

    Parameters
    ----------
    bert: `str`, optional
        Model name or path, see BertTokenizer.from_pretrained
        Defaults to 'bert-base-cased'.
    out_size: `int`, optional
        Output size of the model.
        Defaults to 256.
    num_layers: `int`, optional
        The number of sub-decoder-layers in the decoder.
        If set to 0 then Bert output is fed directly to the Linear layer.
        Defaults to 6
    nhead: `int`, optional
        The number of heads in each decoder layer.
        Defaults to 8
    dim_feedforward: `int`, optional
        The dimension of the feedforward network model (FFN) in each decoder layer.
        Defaults to 2048
    dropout: `float`, optional
        The dropout rate in-between each block (attn or FFN) of each decoder layer
        Defaults to 0.1
    activation: `str`, optional
        The activation function of intermediate layer of the FFN: 'relu' or 'gelu'
         Defaults to 'relu'
    audio_dim: `int`, optional
        Dimension of the audio embeddings.
        Defaults to 512.

    References
    ----------
    TODO
    """

    def __init__(self, bert='bert-base-cased', out_size=256, num_layers=6, nhead=8,
                 dim_feedforward=2048, dropout=0.1, activation='relu', audio_dim=512):

        super().__init__()
        self.bert = BertModel.from_pretrained(bert)
        self.hidden_size = self.bert.config.hidden_size
        self.out_size = out_size
        self.audio_dim = audio_dim
        self.decoder_num_layers = num_layers
        # 0 decoder layers (=) feed BERT output directly to self.linear
        if self.decoder_num_layers == 0:
            self.decoder = None
        # else init decoder as usual
        else:
            self.position_ids = None
            self.position_embeddings = Embedding(self.out_size, self.hidden_size)
            # linear layer so that audio and text embeddings have the same dimension
            self.resize_audio = Linear(self.audio_dim, self.hidden_size)
            # init decoder_layer with the parameters
            decoder_layer = TransformerDecoderLayer(self.hidden_size, nhead,
                                                    dim_feedforward,
                                                    dropout, activation)
            decoder_norm = LayerNorm(self.hidden_size)
            # init decoder with decoder_layer
            self.decoder = TransformerDecoder(decoder_layer, self.decoder_num_layers,
                                              decoder_norm)

        # handle classification layer
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
            extra_repr = f': {module.extra_repr()}' if module.extra_repr() else ''
            lines.append(f'{indent}{name} ({module.__class__.__name__}{extra_repr}) -> '
                         f'{trainable:,d} trainable parameters ({total:,d} total)')
        return '\n'.join(lines)

    def forward(self, input_ids, audio=None, src_key_padding_mask=None, audio_mask=None):
        """Apply model

        Parameters
        ----------
        input_ids: Tensor
            (batch_size, max_length). Encoded input tokens using BertTokenizer
        audio: Tensor, optional
            (batch_size, max_length, audio_dim). audio embeddings of words, aligned with target_ids.
            Defaults to None.
        src_key_padding_mask: Tensor, optional
            (batch_size, max_length). Used to mask input_ids.
            Defaults to None (no masking).
        audio_mask: Tensor, optional
            (batch_size, max_length). Used to mask audio.
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

        # attend text with audio
        if self.decoder is not None:
            if audio is None:
                raise ValueError(f"Expected 'audio' to be a Tensor of embeddings but got '{audio}'. "
                                 f"See documentation below:\n{self.__doc__}")
            audio = self.resize_audio(audio)
            if self.position_ids is None or self.position_ids.shape != input_ids.shape:
                self.position_ids = arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand(input_ids.shape)
            # sum audio embeddings with position embeddings
            audio += self.position_embeddings(self.position_ids)
            audio = audio.transpose(0, 1)
            output = self.decoder(audio, hidden_states, tgt_key_padding_mask=audio_mask)
        else:
            output = hidden_states

        output = self.linear(output)

        # reshape output like (batch_size, sequence_length, out_size)
        output = output.transpose(0, 1)

        # activate with Sigmoid
        output = self.activation(output)
        return output

