#!/usr/bin/env python
# encoding: utf-8
"""
Gets stats and plots stuff given a protocol
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False})

from pyannote.database import get_protocol

DATABASE="Plumcot-Friends"
TASK="SpeakerDiarization"
PROTOCOL="UEM"
SET='train'
FIGURE_DIR='/people/lerner/Images'
protocol_str='{}.{}.{}'.format(DATABASE,TASK,PROTOCOL)

def plot_speech_duration(stats):
    sns.distplot(list(stats['labels'].values()),kde=False,norm_hist=True)
    plt.ylabel("density")
    plt.xlabel("speech duration")
    plt.title(f"Normed histogram of the speech duration in {protocol_str}.{SET}")
    plt.savefig(os.path.join(FIGURE_DIR,f"speech_duration.{protocol_str}.{SET}.png"))

if __name__=='__main__':

    protocol = get_protocol(protocol_str)
    print(f"gettings stats from {protocol_str}...")
    stats=protocol.stats(SET)
    for key,value in stats.items():
        if key=='labels':
            break
        print(key,value)
    plot_speech_duration(stats)
