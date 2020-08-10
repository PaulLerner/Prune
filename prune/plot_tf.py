#!/usr/bin/env python
# encoding: utf-8
"""Usage: plot_tf.py *.csv"""

import sys
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np


sns.set_style("whitegrid", {'axes.grid': False})
np.set_printoptions(precision=2, suppress=True)


def plot(paths):
    metrics = set()
    plt.figure(figsize=(10, 8))
    for path in paths:
        path = Path(path)
        _, name, _, metric = path.stem.split('-')
        results = np.loadtxt(path, delimiter=",",
                             dtype=float, skiprows=1, usecols=(1, 2))
        plt.plot(results[:, 0], results[:, 1], label=name)
        metrics.add(metric)
    plt.xlabel("epochs")
    plt.ylabel(' / '.join(metrics))
    plt.legend()
    plt.show()


if __name__ == '__main__':
    paths = sys.argv[1:]
    plot(paths)