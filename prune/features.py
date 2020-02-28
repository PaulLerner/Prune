#!/usr/bin/env python
# encoding: utf-8

from pyannote.core.utils.distance import pdist
from pyannote.core import Annotation
from scipy.spatial.distance import squareform
import numpy as np

def get_embeddings_per_speaker(features, hypothesis):
    """
    Gets the average speech turn embedding for every speaker and stores it in a dict.
    If a speech turn doesn't contains strictly an embedding then the 'center' mode is used for cropping,
    then the 'loose' mode. See `features.crop`

    Parameters
    ----------
    features : `SlidingWindowFeature`
    hypothesis : `Annotation`

    Returns
    -------
    embeddings_per_speaker : `dict`
        each key is a speaker which is itself a dict where keys are segments
        each segment being the average of embeddings in the segment
    """
    embeddings_per_speaker={speaker:{} for speaker in hypothesis.labels()}
    for segment, track, label in hypothesis.itertracks(yield_label=True):
        # be more and more permissive until we have
        # at least one embedding for current speech turn
        for mode in ['strict', 'center', 'loose']:
            x = features.crop(segment, mode=mode)
            if len(x) > 0:
                break
        # skip speech turns so small we don't have any embedding for it
        if len(x) < 1:
            continue
        # average speech turn embedding
        avg=np.mean(x, axis=0)
        embeddings_per_speaker[label][segment]=avg
    return embeddings_per_speaker

def get_distances_per_speaker(features, hypothesis):
    """
    Gets the distances between every speech turn embeddings for every speaker.

    Parameters
    ----------
    features : `SlidingWindowFeature`
    hypothesis : `Annotation`

    Returns
    -------
    distances_per_speaker : `dict` : each key is a speaker and each value is a list of distances,
        each distance corresponds to the distance between a speech turn and a fictitious cluster center.
    """
    embeddings_per_speaker=get_embeddings_per_speaker(features, hypothesis)
    distances_per_speaker=embeddings_per_speaker.copy()
    for speaker, segments in embeddings_per_speaker.items():
        flat_embeddings=list(segments.values())
        distances=squareform(pdist(flat_embeddings, metric='angular'))
        distances=np.mean(distances,axis=0)
        for i, segment in enumerate(distances_per_speaker[speaker]):
            distances_per_speaker[speaker][segment]=distances[i]
    return distances_per_speaker
