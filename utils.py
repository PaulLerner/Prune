from pyannote.core.utils.distance import pdist
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
    embeddings_per_speaker : `dict` : each key is a speaker and each value is a list of embeddings,
        each embedding being the average of embeddings in the speech turn
    """
    embeddings_per_speaker={speaker:[] for speaker in hypothesis.labels()}
    for segment, track, label in hypothesis.itertracks(yield_label=True):
        # be more and more permissive until we have
        # at least one embedding for current speech turn
        for mode in ['strict', 'center', 'loose']:
            x = features.crop(segment, mode=mode)
            if len(x) > 0:
                break
        # average speech turn embedding
        avg=np.mean(x, axis=0)
        embeddings_per_speaker[label].append(avg)
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
    distances_per_speaker={}
    embeddings_per_speaker=get_embeddings_per_speaker(features, hypothesis)
    for speaker, embeddings in embeddings_per_speaker.items():
        distances=squareform(pdist(embeddings_per_speaker[speaker], metric='angular'))
        distances_per_speaker[speaker]=np.mean(distances,axis=0)
    return distances_per_speaker
