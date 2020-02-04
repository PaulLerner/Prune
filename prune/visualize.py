#!/usr/bin/env python
# encoding: utf-8
"""Usage:
  visualize.py gecko <hypotheses_path> <uri> <database.task.protocol>
  visualize.py distances <hypotheses_path> <uri> <database.task.protocol>
  visualize.py stats <database.task.protocol> [--set=<set> --filter_unk --crop=<crop> --hist --verbose]
  visualize.py -h | --help

gecko options:
    <hypotheses_path>                   Path to the hypotheses (rttm file) you want to convert to gecko-json
    <uri>                               Uri of the hypothesis you want to convert to gecko-json
    <database.task.protocol>            Experimental protocol (e.g. "Etape.SpeakerDiarization.TV")
    --map                               Map hypothesis label with reference

stats options:
  <database.task.protocol>              Experimental protocol (e.g. "Etape.SpeakerDiarization.TV")
"""

import os
from pathlib import Path
import json
from docopt import docopt
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False})

import re
import numpy as np

from pyannote.core import Annotation
from pyannote.audio.features import Precomputed
from pyannote.database.util import load_rttm, load_id
from pyannote.database import get_protocol,get_annotated
from pyannote.metrics.diarization import DiarizationErrorRate

from prune.convert import *

DATA_PATH=Path('/vol', 'work', 'lerner', 'pyannote-db-plumcot', 'Plumcot', 'data')
embeddings="/vol/work/lerner/baseline/emb/train/Plumcot-Friends.SpeakerDiarization.UEM.train/validate/Friends.SpeakerDiarization.FA-UEM.development/apply/1345"

def distances(args):
    hypotheses_path=args['<hypotheses_path>']
    hypotheses, distances=load_id(hypotheses_path)
    uri=args['<uri>']
    hypothesis, distances = hypotheses[uri], distances[uri]
    protocol=args['<database.task.protocol>']
    protocol = get_protocol(protocol)
    for reference in getattr(protocol, 'test')():
        if reference['uri']==uri:
            break
    if reference['uri']!=uri:
        raise ValueError(f"{uri} is not in {protocol}.test")
    annotated=get_annotated(reference)
    annotation=reference['annotation'].crop(annotated)
    true_distances,false_distances=[],[]
    tp,fp=0,0
    close_duration,far_duration=[],[]
    for segment, track, label in hypothesis.itertracks(yield_label=True):
        distance=distances.get(segment)
        distance=distance.get(label) if distance else None
        distance= distance if distance !="<NA>" else None
        annotation_segment=annotation.crop(segment)
        if distance:
            if distance < DISTANCE_THRESHOLD:
                close_duration.append(segment.duration)
            else:
                far_duration.append(segment.duration)
        if label in annotation_segment.labels():
            true_distances.append(distance)
            tp+=segment.duration
        else:
            false_distances.append(distance)
            fp+=segment.duration
    print("debug precision %:",tp/(tp+fp))

    density=False
    true_label=f"True Positives ({len(true_distances)} #, {tp:.0f} s)"
    n, bins, patches=plt.hist(true_distances,30,density=density,alpha=0.5,label=true_label)
    false_label=f"False Positives ({len(false_distances)} #, {fp:.0f} s)"
    plt.hist(false_distances,bins,density=density,alpha=0.5, label=false_label)
    plt.axvline(x=DISTANCE_THRESHOLD,color="black", linestyle='dashed')
    close_text=f"{len(close_duration)} #, {sum(close_duration):.0f} s"
    plt.text(DISTANCE_THRESHOLD-0.2, np.max(n)/2, close_text)
    far_text=f"{len(far_duration)} #, {sum(far_duration):.0f} s"
    plt.text(DISTANCE_THRESHOLD+0.2, np.max(n)/2, far_text)
    plt.legend()
    title="FA: distance between hypothesis and reference in identification pipeline\n"
    #title+="without clustering"
    title+="clustering @DER"
    print(re.sub("\s","_",title))
    plt.title(title)
    plt.ylabel("number of segments")
    plt.xlabel("angular distance")
    plt.show()

def gecko(args):
    hypotheses_path=args['<hypotheses_path>']
    hypotheses, distances=load_id(hypotheses_path)
    uri=args['<uri>']
    serie_uri=uri.split(".")[0]
    fa=Path(DATA_PATH,serie_uri,'forced-alignment')
    #get manual annotation if exists else falls back to raw forced-alignment
    annotation_json=Path(fa,f"{uri}.manual.json") if Path(fa,f"{uri}.manual.json").exists() else Path(fa,f"{uri}.json")
    colors={}
    if annotation_json.exists():
        #get colors
        with open(annotation_json,'r') as file:
            annotation_json=json.load(file)
        for monologue in annotation_json["monologues"]:
            colors[monologue["speaker"]["id"]]=monologue["speaker"].get("color")
    hypothesis, distances = hypotheses[uri], distances[uri]
    precomputed = Precomputed(embeddings)
    protocol=args['<database.task.protocol>']
    protocol = get_protocol(protocol)
    for reference in getattr(protocol, 'test')():
        if reference['uri']==uri:
            features = precomputed(reference)
            break
        if args['--map']:
            print(f"mapping {uri} with {protocol}")
            diarizationErrorRate=DiarizationErrorRate()
            annotated=get_annotated(reference)
            optimal_mapping=diarizationErrorRate.optimal_mapping(reference['annotation'], hypothesis,annotated)
            hypothesis=hypothesis.rename_labels(mapping=optimal_mapping)

    hypothesis=update_labels(hypothesis, distances)#tag unsure clusters
    distances_per_speaker=get_distances_per_speaker(features, hypothesis)
    gecko_json=annotation_to_GeckoJSON(hypothesis, distances_per_speaker, colors)
    dir_path=os.path.dirname(hypotheses_path)
    json_path=os.path.join(dir_path,f'{uri}.json')
    with open(json_path,'w') as file:
        json.dump(gecko_json,file)
    print(f"succefully dumped {json_path}")

if __name__ == '__main__':
    args = docopt(__doc__)
    if args['gecko']:
        gecko(args)
    if args['distances']:
        distances(args)
    if args['stats']:
        from stats import main as stats
        stats(args)
