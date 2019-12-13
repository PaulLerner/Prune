#!/usr/bin/env python
# encoding: utf-8
"""
Gets stats and plots stuff given a protocol

Usage:
  stats.py <database.task.protocol> [--set=<set> --filter_unk --crop=<crop> --hist]
  stats.py -h | --help

Common options:
  <database.task.protocol>   Experimental protocol (e.g. "Etape.SpeakerDiarization.TV")
"""

import os
from docopt import docopt
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False})
import numpy as np
np.set_printoptions(precision=2, suppress=True)

from pyannote.database import get_protocol



def plot_speech_duration(values,protocol_name, set,hist=True,crop=None):
    keep_n=len(values) if crop is None else int(len(values)*crop)
    values.sort()
    values=values[-keep_n:]
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
    fig_type="hist" if hist else "plot"
    save_path=os.path.join(FIGURE_DIR,f"speech_duration.{protocol_name}.{set}.{fig_type}.{keep_n}.png")
    plt.savefig(save_path)
    print(f"succesfully saved {save_path}")

if __name__=='__main__':
    FIGURE_DIR='/people/lerner/Images'

    args = docopt(__doc__)

    protocol_name = args['<database.task.protocol>']
    set=args['--set'] if args['--set'] else "train"
    filter_unk=args['--filter_unk']
    crop=float(args['--crop']) if args['--crop'] else None
    hist=args['--hist']

    protocol = get_protocol(protocol_name)
    print(f"gettings stats from {protocol_name}.{set}...")
    stats=protocol.stats(set)
    for key,value in stats.items():
        if key=='labels':
            break
        print(key,value)

    print("speech duration quartiles :")
    if filter_unk:
        values=[value for label,value in stats['labels'].items() if '#unknown#' not in label]
    else:
        values=list(stats['labels'].values())
    print("n_speakers:",len(values))
    print("quartiles:")
    print(np.quantile(values,[0.,0.25,0.5,0.75,1.0]))

    print("deciles:")
    print(np.quantile(values,np.arange(0,1.1,0.1)))

    plot_speech_duration(values,protocol_name, set, hist,crop)
