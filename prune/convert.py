#!/usr/bin/env python
# encoding: utf-8

from pyannote.core import Annotation, Segment, Timeline

import json
import re

DISTANCE_THRESHOLD=0.5
NA_VALUES={'','<NA>'}

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

def annotation_to_GeckoJSON(annotation, distances=None, colors={}):
    """
    Parameters:
    -----------
    annotation: `pyannote.core.Annotation`
        proper pyannote annotation for speaker identification/diarization
    distances: `dict`, optional
        in-cluster distances between speech features
        see `get_distances_per_speaker`
    colors: `dict`, optional
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
        distance=distances.get(label) if distances else None
        distance=distance.get(segment) if distance else None
        color=colors.get(label) if colors else None
        gecko_json["monologues"].append(
            {
                "speaker":{
                    "id":label,
                    "color": color,
                    "distance":distance,
                    "non_id":[],
                    "annotators":0
                },
                "start" : segment.start,
                "end" : segment.end
        })
    return gecko_json
