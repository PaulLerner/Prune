#!/usr/bin/env python
# encoding: utf-8
"""Usage:
  visualize.py gecko <hypotheses_path> --uri=<uri>
  visualize.py stats <database.task.protocol> [--set=<set> --filter_unk --crop=<crop> --hist --verbose]
  visualize.py -h | --help

gecko options:
    <hypotheses_path>        Path to the hypotheses (rttm file) you want to convert to gecko-json
    --uri=<uri>              Uri of the hypothesis you want to convert to gecko-json
                             Defaults to the first in the RTTM file

stats options:
  <database.task.protocol>   Experimental protocol (e.g. "Etape.SpeakerDiarization.TV")
"""

import os
import json
from docopt import docopt
from pyannote.core import Annotation
from pyannote.database.util import load_rttm
from convert import annotation_to_GeckoJSON

def gecko(args):
    hypotheses_path=args['<hypotheses_path>']
    hypotheses=load_rttm(hypotheses_path)
    uri=args['--uri']
    hypothesis=hypotheses[uri] if uri is not None else next(iter(hypotheses))
    gecko_json=annotation_to_GeckoJSON(hypothesis)
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
