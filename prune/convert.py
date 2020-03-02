#!/usr/bin/env python
# encoding: utf-8
"""Usage:
  convert.py serial_speakers <input_path> <serie_uri> <annotation_path> <annotated_path>
"""
from pyannote.core import Annotation, Segment, Timeline
import pyannote.database
from Plumcot import Plumcot

import json
import numpy as np
import re
from docopt import docopt
from pathlib import Path
import affinegap
from scipy.optimize import linear_sum_assignment

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

def name_alignment(refs, hyps):
    aligned_names=[]
    size = max(len(refs), len(hyps))
    min_size = min(len(refs), len(hyps))
    dists = np.ones([size, size])
    for i, ref in enumerate(refs):
        for j, hyp in enumerate(hyps):
            dists[i, j] = affinegap.normalizedAffineGapDistance(ref, hyp)
    # We use Hungarian algorithm which solves the "assignment problem" in a
    # polynomial time.
    row_ind, col_ind = linear_sum_assignment(dists)
    # Add names ignored by Hungarian algorithm when sizes are not equal
    for i, ref in enumerate(refs):
        if col_ind[i] < len(hyps):
            aligned_names.append(hyps[col_ind[i]])
        else:
            aligned_names.append(ref)
    return aligned_names

def serial_speakers_to_RTTM(input_path, serie_uri, annotation_path, annotated_path):
    annotation_path, annotated_path = Path(annotation_path), Path(annotated_path)

    if annotation_path.exists():
        raise ValueError(f"""{annotation_path} already exists.
                        You probably don't wan't to append any more data to it.""")
    if annotated_path.exists():
        raise ValueError(f"""{annotated_path} already exists.
                        You probably don't wan't to append any more data to it.""")
    annotation_path.parent.mkdir(exist_ok=True)
    annotated_path.parent.mkdir(exist_ok=True)

    db = Plumcot()
    character_uris = db.get_characters(serie_uri, field="character_uri")
    character_names = db.get_characters(serie_uri, field="character_name")

    with open(input_path, 'r') as file:
        serial_speakers = json.load(file)

    for season in serial_speakers['seasons']:
        season_i = season['id']
        for episode in season['episodes']:
            episode_i = episode['id']
            episode_uri = f"{serie_uri}.Season{season_i:02d}.Episode{episode_i:02d}"

            print(f"processing {episode_uri}",end='\r')
            annotation, annotated = serial_speaker_to_Annotation(episode, episode_uri, 'speaker')
            
            with open(annotation_path,'a') as file:
                annotation.write_rttm(file)
            with open(annotated_path,'a') as file:
                annotated.write_uem(file)

def get_serial_speaker_names(serial_speaker):
    return {segment['speaker'] for segment in serial_speaker["data"]["speech_segments"]}

def unknown_char(char_name, id_ep):
    """Transforms character name into unknown version."""
    return f"{char_name}#unknown#{id_ep}"

def serial_speaker_to_Annotation(serial_speaker, uri=None, modality='speaker'):
    """
    Parameters:
    -----------
    serial_speaker : `dict`
        loaded from a serial speaker JSON as defined
        in https://figshare.com/articles/TV_Series_Corpus/3471839
    uri (uniform resource identifier) : `str`, optional
        which identifies the annotation (e.g. episode number)
        Default : None
    modality : `str`, optional
        modality of the annotation as defined in https://github.com/pyannote/pyannote-core

    Returns:
    --------
    annotation: pyannote `Annotation`
        for speaker identification/diarization as defined
        in https://github.com/pyannote/pyannote-core
    annotated: pyannote `Timeline`
        representing the annotated parts of the serial_speaker file
        Unknown speakers are not considered as annotated
    """

    annotation = Annotation(uri, modality)
    not_annotated = Timeline(uri=uri)

    for segment in serial_speaker["data"]["speech_segments"]:
        time = Segment(segment["start"],segment["end"])
        speaker_id = segment['speaker'].replace(" ", "_")
        annotation[time, speaker_id] = speaker_id
        if speaker_id == 'unknown':
            not_annotated.add(time)

    end=serial_speaker.get("duration",segment["end"])
    annotated=not_annotated.gaps(support=Segment(0.0,end))
    return annotation, annotated

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
        representing the annotated parts of the gecko_JSON file
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

def annotation_to_GeckoJSON(annotation, distances={}, colors={}):
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
        distance=distances.get(label, {}).get(segment)
        color=colors.get(label)
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

if __name__ == '__main__':
    args = docopt(__doc__)
    input_path = args['<input_path>']
    serie_uri = args['<serie_uri>']
    annotation_path = args['<annotation_path>']
    annotated_path = args['<annotated_path>']
    serial_speakers_to_RTTM(input_path, serie_uri, annotation_path, annotated_path)
