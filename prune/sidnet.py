#!/usr/bin/env python
# encoding: utf-8

from typing import List

from transformers import BertModel, BertConfig, BertTokenizer
from torch.nn import Transformer, Module, Linear, CrossEntropyLoss

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

    def __init__(self, bert='bert-base-cased', audio=None, **kwargs):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert)
        self.hidden_size = self.embedding.config.hidden_size
        self.seq2seq = Transformer(d_model=self.hidden_size, **kwargs)
        self.linear = Linear(self.hidden_size, self.tokenizer.vocab_size)

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

    def forward(self, input_ids, targets_ids,
                audio_similarity=None, attention_mask=None, tgt_mask=None,
                src_key_padding_mask=None, tgt_key_padding_mask=None):
        """Apply model

        Parameters
        ----------
        input_ids: Tensor
            (batch_size, max_length). Encoded input tokens using BertTokenizer
        targets_ids: Tensor
            (batch_size, max_length). Encoded target tokens using BertTokenizer
        audio_similarity: Tensor, optional
            (batch_size, max_length, max_length). Similarity (e.g. cosine distance)
            between audio embeddings of words, aligned with targets_ids.
            Defaults to None, indicating that the model should rely only on the text.
        src_key_padding_mask: Tensor, optional
            (batch_size, max_length). Used to mask input_ids.
            Defaults to None (no masking).
        tgt_key_padding_mask: Tensor, optional
            (batch_size, max_length). Used to mask targets_ids.
            Defaults to None (no masking).

        Returns
        -------
        output: Tensor
            (batch_size, max_length). Model's hypothesis encoded like input_ids
        """
        # pass input text trough bert
        hidden_states = self.bert(input_ids, attention_mask)[0]

        # embed targets using bert embeddings
        embedded_targets = self.bert.embeddings(targets_ids)

        # reshape BertModel output like (sequence_length, batch_size, hidden_size)
        # to fit torch.nn.Transformer
        hidden_states = hidden_states.transpose(0, 1)
        embedded_targets = embedded_targets.transpose(0, 1)

        # FIXME are all these masks done the right way ?
        src_mask = self.seq2seq.generate_square_subsequent_mask(len(hidden_states))
        tgt_mask = self.seq2seq.generate_square_subsequent_mask(len(embedded_targets))
        text_output = self.seq2seq(hidden_states, embedded_targets,
                                   src_mask=src_mask, tgt_mask=tgt_mask,
                                   src_key_padding_mask=src_key_padding_mask,
                                   tgt_key_padding_mask=tgt_key_padding_mask)

        text_output = self.linear(text_output)

        # TODO weigh output using audio_similarity here
        output = text_output
        return output

