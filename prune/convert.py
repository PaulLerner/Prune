#!/usr/bin/env python
# encoding: utf-8

from pyannote.core.utils.distance import pdist
from scipy.spatial.distance import squareform
import numpy as np

import json
DISTANCE_THRESHOLD=0.5

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

def update_labels(annotation, distances):
    """Tag labels with "?" depending on their distance to the reference
    """
    for segment, track, label in annotation.itertracks(yield_label=True):
        distance=distances.get(segment)
        distance=distance.get(label) if distance else None
        distance= distance if distance !="<NA>" else None
        if distance:
            if distance > DISTANCE_THRESHOLD:
                annotation[segment,track]=f"?{label}"
    return annotation

def annotation_to_GeckoJSON(annotation, distances, colors={}):
    """
    Parameters:
    -----------
    annotation: `pyannote.core.Annotation`
        proper pyannote annotation for speaker identification/diarization
    colors: `dict`
        speaker id : consistent color

    Returns:
    --------
    gecko_json : a JSON `dict` based on the demo file of https://github.com/gong-io/gecko/blob/master/samples/demo.json
        should be written to a file using json.dump
    """

    gecko_json=json.loads("""{
      "schemaVersion" : "2.0",
      "monologues" : [  ]
    }""")
    for segment, track, label in annotation.itertracks(yield_label=True):
        distance=distances.get(segment)
        distance=distance.get(label) if distance else None
        distance= distance if distance !="<NA>" else None
        if distance:
            if distance > DISTANCE_THRESHOLD:
                label=f"?{label}"

        color=colors.get(label) if colors else None
        
        gecko_json["monologues"].append(
            {
                "speaker":{
                    "id":label,
                    "start" : segment.start,
                    "end" : segment.end,
                    "color": color
                },
                "terms":[
                    {
                        "start" : segment.start,
                        "end" : segment.end
                    }
                ]
        })
    return gecko_json
