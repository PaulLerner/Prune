#!/usr/bin/env python
# encoding: utf-8
"""Usage:
  visualize.py gecko <hypotheses_path> <uri> [--protocol=<database.task.protocol>]
  visualize.py stats <database.task.protocol> [--set=<set> --filter_unk --crop=<crop> --hist --verbose]
  visualize.py -h | --help

gecko options:
    <hypotheses_path>                   Path to the hypotheses (rttm file) you want to convert to gecko-json
    <uri>                               Uri of the hypothesis you want to convert to gecko-json
    --protocol=<database.task.protocol> Experimental protocol (e.g. "Etape.SpeakerDiarization.TV")

stats options:
  <database.task.protocol>              Experimental protocol (e.g. "Etape.SpeakerDiarization.TV")
"""

import os
from pathlib import Path
import json
from docopt import docopt
from pyannote.core import Annotation
from pyannote.database.util import load_rttm, load_id
from convert import annotation_to_GeckoJSON
from pyannote.database import get_protocol
from pyannote.metrics.diarization import DiarizationErrorRate

DATA_PATH=Path('/vol', 'work', 'lerner', 'pyannote-db-plumcot', 'Plumcot', 'data')

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
    protocol=args['--protocol']
    if protocol:
        print(f"mapping {uri} with {protocol}")
        protocol = get_protocol(protocol)
        #optimal mapping
        for reference in getattr(protocol, 'test')():
            if reference['uri']==uri:
                break
        diarizationErrorRate=DiarizationErrorRate()
        optimal_mapping=diarizationErrorRate.optimal_mapping(reference['annotation'], hypothesis,reference['annotated'])
        hypothesis=hypothesis.rename_labels(mapping=optimal_mapping)

    gecko_json=annotation_to_GeckoJSON(hypothesis, distances, colors)
    dir_path=os.path.dirname(hypotheses_path)
    json_path=os.path.join(dir_path,f'{uri}.json')
    with open(json_path,'w') as file:
        json.dump(gecko_json,file)
    print(f"succefully dumped {json_path}")

if __name__ == '__main__':
    args = docopt(__doc__)
    if args['gecko']:
        gecko(args)
    if args['stats']:
        from stats import main as stats
        stats(args)
