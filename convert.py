#!/usr/bin/env python
# encoding: utf-8

import json

def annotation_to_GeckoJSON(annotation):
    """
    Parameters:
    -----------
    annotation: proper pyannote annotation for speaker identification/diarization as defined in https://github.com/pyannote/pyannote-core

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
        gecko_json["monologues"].append(
            {
                "speaker":{
                    "id":label,
                    "start" : segment.start,
                    "end" : segment.end
                },
                "terms":[
                    {
                        "start" : segment.start,
                        "end" : segment.end
                    }
                ]
        })
    return gecko_json
