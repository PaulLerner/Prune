#!/usr/bin/env python
# encoding: utf-8

from pyannote.core.utils.distance import pdist
from pyannote.core import Annotation, Segment, Timeline
from scipy.spatial.distance import squareform
import numpy as np

import json
import re

DISTANCE_THRESHOLD=0.5
NA_VALUES={'','<NA>'}

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

def gecko_JSON_to_Annotation(gecko_JSON, uri=None, modality='speaker'):
    """
    Parameters:
    -----------
    gecko_JSON : `dict`
        loaded from a Gecko-compliant JSON as defined in xml_to_GeckoJSON
    uri (uniform resource identifier) : `str`
        which identifies the annotation (e.g. episode number)
        Default : None
    modality : `str`
        modality of the annotation as defined in https://github.com/pyannote/pyannote-core

    Returns:
    --------
    annotation: pyannote `Annotation`
        for speaker identification/diarization as defined in https://github.com/pyannote/pyannote-core
    must_link: pyannote `Annotation`
        User annotation
    cannot_link: pyannote `Annotation`
        User annotation of parts were the labelled speakers **do not** speak
    annotated: pyannote `Timeline`
        representing the annotated parts of the gecko_JSON files (depends on confidence_threshold)
    """

    annotation = Annotation(uri, modality)
    must_link = Annotation(uri, modality)
    cannot_link = Annotation(uri, f"non-{modality}")

    for monologue in gecko_JSON["monologues"]:
        segment = Segment(monologue["start"],monologue["end"])
        # '@' defined in https://github.com/hbredin/pyannote-db-plumcot/blob/develop/CONTRIBUTING.md#idepisodetxt
        # '+' defined in https://github.com/gong-io/gecko/blob/master/app/geckoModule/constants.js#L35
        speaker_ids=re.split("@|\+",monologue["speaker"]["id"])
        speaker_ids=set(speaker_ids)-NA_VALUES
        for speaker_id in speaker_ids:#most of the time there's only one
            annotation[segment,speaker_id] = speaker_id
            if monologue["speaker"]["annotators"] > 0:
                must_link[segment,speaker_id] = speaker_id

        non_ids = monologue["speaker"]["non_id"]
        non_ids=set(non_ids)-NA_VALUES
        for speaker_id in non_ids:
            cannot_link[segment,speaker_id] = speaker_id

    annotated=Timeline([Segment(0.0,monologue["end"])],uri)
    return annotation, must_link, cannot_link, annotated

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
      "schemaVersion" : "3.1",
      "monologues" : [  ]
    }""")
    for segment, track, label in annotation.itertracks(yield_label=True):
        distance=distances.get(label)
        distance=distance.get(segment) if distance else None
        color=colors.get(label) if colors else None
        gecko_json["monologues"].append(
            {
                "speaker":{
                    "id":label,
                    "color": color,
                    "distance":distance,
                    "non_id":[],
                    "annotators":0,
                    "start" : segment.start,
                    "end" : segment.end
                },
                "start" : segment.start,
                "end" : segment.end
        })
    return gecko_json
