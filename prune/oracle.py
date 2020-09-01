#!/usr/bin/env python
# encoding: utf-8

"""Usage: oracle.py <protocol> [--subset=<subset>]"""

from docopt import docopt
from tqdm import tqdm
from pathlib import Path
import json
from tabulate import tabulate

from collections import Counter
import re
import numpy as np
from transformers import BertTokenizer

from pyannote.database import get_protocol
import Plumcot as PC


EPOCH_FORMAT = '{:04d}.tar'
BERT = 'bert-base-cased'

# constant paths
DATA_PATH = Path(PC.__file__).parent / 'data'
CHARACTERS_PATH = DATA_PATH.glob('*/characters.txt')


def oracle(tokenizer, protocol, mapping, subset='test'):
    with open(mapping) as file:
        mapping = json.load(file)
    for current_file in tqdm(getattr(protocol, subset)(), desc='Loading transcriptions'):
        transcription = current_file['transcription']
        uri = current_file['uri']

        tokens, targets = [], []
        oracle_correct, oracle_total = 0, 0
        for word in transcription:
            target = mapping.get(word._.speaker, tokenizer.pad_token)

            # handle basic tokenization (e.g. punctuation) before Word-Piece
            # in order to align input text and speakers
            for token in tokenizer.basic_tokenizer.tokenize(word.text):
                tokens.append(token)
                targets.append(target)
        n_token = len(tokens)
        tokens = " ".join(tokens)
        targets = Counter(targets)
        for target, count in targets.most_common():
            if target in tokenizer.all_special_tokens:
                continue
            if re.search(target, tokens, flags=re.IGNORECASE):
                oracle_correct += count
            oracle_total += count
        yield uri, oracle_correct/oracle_total, n_token


if __name__ == '__main__':
    # parse arguments and get protocol
    args = docopt(__doc__)
    protocol_name = args['<protocol>']
    protocol = get_protocol(protocol_name)
    serie, _, _ = protocol_name.split('.')
    mapping = DATA_PATH / serie / 'annotated_transcripts' / 'names_dict.json'

    # instantiate tokenizer
    tokenizer = BertTokenizer.from_pretrained(BERT)
    # override basic-tokenization parameter as we need to align speakers with input tokens
    tokenizer.do_basic_tokenize = False

    subset = args['--subset'] if args['--subset'] else 'test'
    full_name = f"{protocol_name}.{subset}"
    # get oracle accuracy for protocol subset
    uris, accuracies, n_tokens = [], [], []
    for uri, accuracy, n_token in oracle(tokenizer, protocol, mapping, subset):
        uris.append(uri)
        accuracies.append(accuracy)
        n_tokens.append(n_token)
    n_tokens = f"{np.mean(n_tokens):.2f} $\\pm$ {np.std(n_tokens):.2f}"
    uris.append(full_name)
    accuracies.append(np.mean(accuracies))
    caption = (f"Oracle accuracy (file-level), protocol {full_name}, "
               f"Average \\# of words: {n_tokens}.")
    # print oracle accuracy
    print(tabulate(zip(uris, accuracies), headers=('uri', 'accuracy'), tablefmt='latex'))
    print("\\caption{%s}" % caption)