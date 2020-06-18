#!/usr/bin/env python
# encoding: utf-8

import os
from docopt import docopt
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

sns.set_style("whitegrid", {'axes.grid': False})
np.set_printoptions(precision=2, suppress=True)

EXP_PATH = Path("experiments/fine-tuning/embeddings/forced-alignment/20_per_fold/")
experiments = {
    "weighted_coverage": {
        "path": Path(EXP_PATH, "weighted_coverage")
    },
    "coverage": {
        "path": Path(EXP_PATH, "coverage_no_weights")
    }
}

for experiment in experiments:
    for file in os.listdir(experiments[experiment]["path"]):
        if "loss" in file:
            metric = "loss"
        else:
            metric = "coverage"
        path = os.path.join(experiments[experiment]["path"], file)
        results = np.loadtxt(path, delimiter=",", dtype=float, skiprows=1, usecols=(1, 2))
        if "Friends" in file:
            database = "Friends"
        else:
            database = "TBBT"
        print(file, metric, database)
        experiments[experiment][database] = results
plt.figure(figsize=(10, 8))
metric = "coverage"
for experiment in experiments:
    for database in experiments[experiment]:
        if database == "path":
            continue
        print(experiment, database)
        plt.title(f"{metric} while fine-tuning embeddings on friends FA")
        plt.xlabel("epochs")
        plt.ylabel(metric)
        results = experiments[experiment][database]
        plt.plot(results[:, 0], results[:, 1], label=f"{experiment}_{database}")
        plt.legend()
plt.show()
