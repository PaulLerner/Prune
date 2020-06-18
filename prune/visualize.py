#!/usr/bin/env python
# encoding: utf-8
"""Usage:
  visualize.py gecko (<hypotheses_path>|<database.task.protocol>) <uri> [--map  --tag_na --database.task.protocol=<database.task.protocol> --embeddings=<embeddings>]
  visualize.py speakers (<hypotheses_path>|<database.task.protocol>) <uri>
  visualize.py update_distances <json_path> <uri> <database.task.protocol>
  visualize.py stats <database.task.protocol> [--set=<set> --filter_unk --crop=<crop> --hist --verbose]
  visualize.py -h | --help

gecko options:
    <hypotheses_path>                   Path to the hypotheses (rttm file) you want to convert to gecko-json
    <database.task.protocol>            Experimental protocol (e.g. "Etape.SpeakerDiarization.TV")
    <uri>                               Uri of the hypothesis you want to convert to gecko-json
    --embeddings=<embeddings>           Path to precomputed embeddings
    --database.task.protocol=<d.t.p>    Experimental protocol (e.g. "Etape.SpeakerDiarization.TV")
    --map                               Map hypothesis label with reference
    --tag_na                            Tag not annotated parts of the hypothesis as "#not_annotated#"
                                        Only available if annotated is provided

stats options:
  <database.task.protocol>              Experimental protocol (e.g. "Etape.SpeakerDiarization.TV")
"""

import os
from pathlib import Path
import json
from datetime import datetime

from docopt import docopt
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import seaborn as sns

sns.set_style("whitegrid", {'axes.grid': False})

import re
import numpy as np

from pyannote.core import Annotation, Segment
from pyannote.audio.features import Precomputed
from pyannote.database.util import load_rttm
from pyannote.database import get_protocol, get_annotated
from pyannote.metrics.diarization import DiarizationErrorRate
import pyannote.database
from Plumcot import Plumcot
import Plumcot as PC

from prune.convert import *
from prune.features import *

DATA_PATH = Path(PC.__file__).parent / "data"


def color_gen():
    cm = get_cmap('Set1')
    while True:
        x = np.random.rand()
        r, g, b, alpha = cm(x, bytes=True)
        color = f'#{r:02x}{g:02x}{b:02x}'
        yield color


def update_distances(args):
    """Loads user annotation from json path, converts it to pyannote `Annotation`
    using regions timings.

    From the annotation uri and precomputed embeddings, it computes the
    in-cluster distances between every speech turns

    Dumps the updated (with correct distances) JSON file to a timestamped file.
    """
    json_path = Path(args['<json_path>'])
    uri = args['<uri>']
    with open(json_path, 'r') as file:
        gecko_json = json.load(file)
    hypothesis, _, _, _ = gecko_JSON_to_Annotation(gecko_json, uri, 'speaker')

    colors = get_colors(uri)

    precomputed = Precomputed(embeddings)
    protocol = args['<database.task.protocol>']
    protocol = get_protocol(protocol)
    for reference in getattr(protocol, 'test')():
        if reference['uri'] == uri:
            features = precomputed(reference)
            break
    distances_per_speaker = get_distances_per_speaker(features, hypothesis)
    gecko_json = annotation_to_GeckoJSON(hypothesis, distances_per_speaker, colors)
    time = datetime.today().strftime('%Y%m%d-%H%M%S')
    name = f"{json_path.stem}.{time}.json"
    updated_path = Path(json_path.parent, name)
    with open(updated_path, 'w') as file:
        json.dump(gecko_json, file)
    print(f"succefully dumped {updated_path}")


def get_colors(uri):
    db = Plumcot()

    serie_uri = uri.split(".")[0]
    if serie_uri not in db.get_protocols("Collection"):
        # non PLUMCOT -> non-persistent colors for now
        return {}

    colors_dir = Path(DATA_PATH, serie_uri, 'colors')
    colors_dir.mkdir(exist_ok=True)
    colors_path = Path(colors_dir, f'{uri}.json')

    if colors_path.exists():
        with open(colors_path, "r") as file:
            colors = json.load(file)
        return colors

    # else: extract from gecko_json or generate with matplotlib
    fa = Path(DATA_PATH, serie_uri, 'forced-alignment')
    # get manual annotation if exists else falls back to raw forced-alignment
    annotation_json = Path(fa, f"{uri}.manual.json") if Path(fa,
                                                             f"{uri}.manual.json").exists() else Path(
        fa, f"{uri}.json")
    colors = {}
    if annotation_json.exists():
        # get colors
        with open(annotation_json, 'r') as file:
            annotation_json = json.load(file)
        for monologue in annotation_json["monologues"]:
            if not isinstance(monologue, dict):
                continue
            color = monologue["speaker"].get("color", next(color_gen()))
            colors[monologue["speaker"]["id"]] = color
    else:  # no annotation -> falls back to character list
        characters = db.get_characters(serie_uri)[uri]
        colors = {character: next(color_gen()) for character in characters}
    with open(colors_path, 'w') as file:
        json.dump(colors, file)
    return colors


def get_file(protocol, uri, embeddings=None):
    for reference in protocol.files():
        if reference['uri'] == uri:
            if embeddings:
                precomputed = Precomputed(embeddings)
                features = precomputed(reference)
                return reference, features
            return reference
    raise ValueError(f'{uri} is not in {protocol}')


def na():
    while True:
        yield "#not_annotated#"


def gecko(args):
    hypotheses_path = args['<hypotheses_path>']
    uri = args['<uri>']
    colors = get_colors(uri)
    distances = {}
    if Path(hypotheses_path).exists():
        hypotheses = load_rttm(hypotheses_path)
        hypothesis = hypotheses[uri]
    else:  # protocol
        protocol = get_protocol(args['<hypotheses_path>'])
        reference = get_file(protocol, uri)
        hypothesis = reference['annotation']
        annotated = get_annotated(reference)
    hypotheses_path = Path(hypotheses_path)
    protocol = args['--database.task.protocol']
    features = None
    if protocol:
        protocol = get_protocol(protocol)
        embeddings = args['--embeddings']
        reference, features = get_file(protocol, uri, embeddings=embeddings)
        if args['--map']:
            print(f"mapping {uri} with {protocol}")
            diarizationErrorRate = DiarizationErrorRate()
            annotated = get_annotated(reference)
            optimal_mapping = diarizationErrorRate.optimal_mapping(
                reference['annotation'], hypothesis, annotated)
            hypothesis = hypothesis.rename_labels(mapping=optimal_mapping)

    hypothesis = update_labels(hypothesis, distances)  # tag unsure clusters

    distances_per_speaker = get_distances_per_speaker(features,
                                                      hypothesis) if features else {}

    if args['--tag_na']:
        whole_file = Segment(0., annotated.segments_boundaries_[-1])
        not_annotated = annotated.gaps(whole_file).to_annotation(na())
        hypothesis = hypothesis.crop(annotated).update(not_annotated)

    gecko_json = annotation_to_GeckoJSON(hypothesis, distances_per_speaker, colors)

    if hypotheses_path.exists():
        dir_path = hypotheses_path.parent
    else:
        dir_path = Path(".")

    json_path = os.path.join(dir_path, f'{uri}.json')
    with open(json_path, 'w') as file:
        json.dump(gecko_json, file)
    print(f"succefully dumped {json_path}")


def speakers(args):
    hypotheses_path = args['<hypotheses_path>']
    uri = args['<uri>']
    if Path(hypotheses_path).exists():
        hypotheses = load_rttm(hypotheses_path)
        hypothesis = hypotheses[uri]
    else:  # protocol
        distances = {}
        protocol = get_protocol(args['<hypotheses_path>'])
        reference = get_file(protocol, uri)
        hypothesis = reference['annotation']
        annotated = get_annotated(reference)
    print(uri)
    print(f"Number of speakers: {len(hypothesis.labels())}")
    print(f"Chart:\n{hypothesis.chart()}")


if __name__ == '__main__':
    args = docopt(__doc__)
    if args['gecko']:
        gecko(args)
    if args['speakers']:
        speakers(args)
    if args['update_distances']:
        update_distances(args)
    if args['stats']:
        from .stats import main as stats

        stats(args)
