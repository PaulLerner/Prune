#!/usr/bin/env python
# encoding: utf-8
"""
Gets stats and plots stuff given a protocol

Usage:
  stats.py <database.task.protocol> [--set=<set> --min_duration=<min_duration> --filter_unk --crop=<crop> --hist --verbose]
  stats.py -h | --help

Common options:
  <database.task.protocol>   Experimental protocol (e.g. "Etape.SpeakerDiarization.TV")
"""

import os
from docopt import docopt
from allies.utils import print_stats
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False})
import numpy as np
np.set_printoptions(precision=2, suppress=True)

from pyannote.database import get_protocol

FIGURE_DIR='.'

def plot_speech_duration(values,protocol_name, set,hist=True,crop=None):
    keep_n=len(values) if crop is None else int(len(values)*crop)
    values.sort()
    values=values[-keep_n:]
    mean=np.mean(values)
    std=np.std(values)
    print(f"mean: {mean:.2f}")
    print(f"std: {std:.2f}")
    print(f"mean+std: {mean+std:.2f}")
    plt.figure(figsize=(12,10))
    title=(
        f"of the speech duration in {protocol_name}.{set} "
        f"of the {keep_n} biggest speakers"
    )
    if hist:
        sns.distplot(values,kde=False,norm_hist=True)
        plt.ylabel("density")
        plt.xlabel("speech duration (s)")
        plt.title("Normed histogram "+title)
    else:
        plt.title("Plot "+title)
        plt.ylabel("speech duration (s)")
        plt.xlabel("speaker #")
        plt.plot(values,".")
        plt.errorbar(np.arange(len(values)),[mean for _ in values],[std for _ in values])
        plt.legend()
    fig_type="hist" if hist else "plot"
    save_path=os.path.join(FIGURE_DIR,f"speech_duration.{protocol_name}.{set}.{fig_type}.{keep_n}.png")
    plt.show()
    #plt.savefig(save_path)
    #print(f"succesfully saved {save_path}")

def main(args):
    protocol_name = args['<database.task.protocol>']
    set=args['--set'] if args['--set'] else "train"
    filter_unk=args['--filter_unk']
    crop=float(args['--crop']) if args['--crop'] else None
    hist=args['--hist']
    verbose=args['--verbose']
    min_duration=float(args['--min_duration']) if args['--min_duration'] else 0.0

    protocol = get_protocol(protocol_name)
    # print("uri \t duration (s)")
    # for item in protocol.train():
    #     uri = item['uri']
    #     annotated=item['annotated']
    #     annotation=item['annotation'].crop(annotated)
    #     speech_duration = annotation.get_timeline().duration()
    #     print(f"{uri} \t {speech_duration}")
    print(f"gettings stats from {protocol_name}.{set}...")
    stats=protocol.stats(set, min_duration)
    print_stats(stats)
    if filter_unk:
        values=[value for label,value in stats['labels'].items() if '#unknown#' not in label]
    else:
        values=list(stats['labels'].values())
    print(f"n_speaking_speakers: {np.array(values).nonzero()[0].shape[0]}")
    print("quartiles:")
    print(np.quantile(values,[0.,0.25,0.5,0.75,1.0]))

    print("deciles:")
    print(np.quantile(values,np.arange(0,1.1,0.1)))

    plot_speech_duration(values,protocol_name, set, hist,crop)

if __name__=='__main__':
    args = docopt(__doc__)
    main(args)
