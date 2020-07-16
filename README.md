# Prune
Code for Plumcot project which doesn't fit in https://github.com/pyannote, https://github.com/PaulLerner/pyannote-db-plumcot/ nor https://github.com/PaulLerner/gecko

See also https://github.com/PaulLerner/Forced-Alignment which unfortunately contains closed source dependencies.

## Named-identification (`named_id.py` and `sidnet.py`)

:warning: Beware not to use `torch 1.3`, see https://github.com/pytorch/pytorch/issues/28272

## Visualization (`visualize.py`)

```
Usage:
  visualize.py gecko (<hypotheses_path>|<database.task.protocol>) <uri> [--map  --tag_na --database.task.protocol=<database.task.protocol> --embeddings=<embeddings>]
  visualize.py update_distances <json_path> <uri> <database.task.protocol>
  visualize.py distances <hypotheses_path> <uri> <database.task.protocol>
  visualize.py stats <database.task.protocol> [--set=<set> --filter_unk --crop=<crop> --hist --verbose]
  visualize.py -h | --help

gecko options:
    <hypotheses_path>                   Path to the hypotheses (rttm file) you want to convert to gecko-json
    <database.task.protocol>            Experimental protocol (e.g. "Etape.SpeakerDiarization.TV")
    <uri>                               Uri of the hypothesis you want to convert to gecko-json
    --embeddings=<embeddings>           Path to precomputed embeddings
    --database.task.protocol=<d.t.p>    Experimental protocol (e.g. "Etape.SpeakerDiarization.TV")
    --map                               Map hypothesis label with reference
    --tag_na                            Tag not annotated parts of the hypothesis as "#not_annotated#"
                                        Only available if annotated is provided

stats options:
  <database.task.protocol>              Experimental protocol (e.g. "Etape.SpeakerDiarization.TV")
```

### Stats
Get stats from a [pyannote.database protocol](https://github.com/pyannote/pyannote-database#custom-protocols) subset (defaults to 'train') and plots either an histogram or a scatter-plot of the speaker durations.

Alternatively you can use `stats.py` directly

### Distances
Study the distribution of distances between the embeddings of a speech segment and the reference (i.e. identity) it was assigned to w.r.t. whether the segment is a True or False positive.

### Gecko

Convert a `pyannote.Annotation` (provided either by a pyannote protocol or directly loaded from a RTTM file) to [gecko](https://github.com/gong-io/gecko) compliant-JSON.

### Update distances
Another gecko-centric feature:
Loads user annotation from JSON path, converts it to pyannote `Annotation`
  using regions timings.

  From the annotation uri and precomputed embeddings, it computes the
  in-cluster distances between every speech turns

  Dumps the updated (with correct distances) JSON file to a timestamped file.

## Convert
```
Usage:
  convert.py serial_speakers <input_path> <serie_uri> <annotation_path> <annotated_path>
```
Convert the [Serial Speakers dataset](https://figshare.com/articles/TV_Series_Corpus/3471839) from Bost et al. to RTTM.

#### References

Bost, X., Labatut, V., Linares, G., 2020. Serial speakers: a dataset of tv series. arXivpreprint arXiv:2002.06923
