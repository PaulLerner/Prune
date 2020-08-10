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
    results = {}
    metrics = set()
    for path in paths:
        path = Path(path)
        _, name, _, metric = path.stem.split('-')
        results[name] = np.loadtxt(path, delimiter=",",
                                       dtype=float, skiprows=1, usecols=(1, 2))
        metrics.add(metric)
    plt.figure(figsize=(10, 8))
    plt.xlabel("epochs")
    plt.ylabel(' / '.join(metrics))
    for name, result in results.items():
        plt.plot(result[:, 0], result[:, 1], label=name)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    paths = sys.argv[1:]
    plot(paths)