#!/usr/bin/env python
# encoding: utf-8

import os
from docopt import docopt
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False})
import numpy as np
np.set_printoptions(precision=2, suppress=True)

experiments={
    "weighted_coverage":{
        "path":"Plumcot/experiments/fine-tuning/embeddings/forced-alignment/20_per_fold/weighted_coverage"
    },
    "coverage":{
        "path":"Plumcot/experiments/fine-tuning/embeddings/forced-alignment/20_per_fold/coverage_no_weights"
    }
}

for experiment in experiments:
    for file in os.listdir(experiments[experiment]["path"]):
        if "loss" in file:
            metric="loss"
        else:
            metric="coverage"
        print(metric)
        path=os.path.join(experiments[experiment]["path"],file)

        print(file)
        results=np.loadtxt(path,delimiter=",",dtype=float,skiprows=1,usecols=(1,2))
        experiments[experiment][metric]=results

plt.figure(figsize=(10,8))
metric="coverage"
for experiment in experiments:
    plt.title(f"{metric} while fine-tuning embeddings on friends FA")
    plt.xlabel("epochs")
    plt.ylabel(metric)
    results=experiments[experiment][metric]
    plt.plot(results[:,0],results[:,1],label=experiment)
    plt.legend()
