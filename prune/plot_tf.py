#!/usr/bin/env python
# encoding: utf-8
"""Usage: plot_tf.py *.csv"""

import sys
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
from pathlib import Path
import numpy as np
import warnings
from itertools import cycle

sns.set_style("whitegrid", {'axes.grid': False})
np.set_printoptions(precision=2, suppress=True)

STYLES = cycle(['-', ':', '--', '-.', ''])
COLORS = cycle(colors.TABLEAU_COLORS)


def setdefault(d, k, iterator):
    """Sets d[k] = next(iterator) if k not in d"""
    if k not in d:
        v = next(iterator)
        d[k] = v
        return v
    else:
        return d[k]


def plot(paths):
    metrics = set()
    plt.figure(figsize=(10, 8))
    styles, colors = {}, {}
    for path in paths:
        path = Path(path)
        _, name, _, metric = path.stem.split('-')
        experiment = name.split('_')
        root = experiment[-3]
        dataset = experiment[-1]
        setdefault(styles, root, STYLES)
        setdefault(colors, dataset, COLORS)
        results = np.loadtxt(path, delimiter=",",
                             dtype=float, skiprows=1, usecols=(1, 2))
        epochs, indices = np.unique(results[:, 0], return_index=True)
        if epochs.shape != results[:, 0].shape:
            warnings.warn(f"Multiple values for the same epoch were found in '{path}'.")
        plt.plot(epochs, results[indices, 1], color=colors[dataset],
                 linestyle=styles[root], label=f'{root}/{dataset}')
        metrics.add(metric)
    plt.xlabel("epochs")
    plt.ylabel(' / '.join(metrics))
    plt.legend()
    plt.show()


if __name__ == '__main__':
    paths = sys.argv[1:]
    plot(paths)