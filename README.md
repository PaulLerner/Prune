# Prune
Code for Plumcot project which doesn't fit in https://github.com/pyannote, https://github.com/PaulLerner/pyannote-db-plumcot/ nor https://github.com/PaulLerner/gecko

See also https://github.com/PaulLerner/Forced-Alignment which unfortunately contains closed source dependencies.

## Named-identification
### `named_id.py`

Please have a look at [pyannote.db.plumcot](https://github.com/PaulLerner/pyannote-db-plumcot/) documentation first.

File structure and usage is very similar to `pyannote.audio`:

```
<experiment_dir>
└───config.yml
│   <train_dir>
│   └────weights
│   │   └───*.tar
│   │   <validate_dir>
│   │   └───params.yml
│   │   │   <test_dir>
│   │   │   └───params.yml
│   │   │   │   eval
```

`config.yml` is optional to set additional parameters (e.g. change the default model architecture)

- `train` will train the model for `epochs` epochs, starting from `--from=<epoch>` (defaults to 0), dropping checkpoints every `save_every`.
  You can also customize learning rate, gradient clipping and freeze modules.
- `validate` will evaluate each checkpoint of the trained model and store the best metric in `params.yml`. 
  You can pick the validation order with `--evergreen` and visualize examples with `--interactive`
- `test` will evaluate only the best model according to the previous validation step, print and write the metrics in `eval`
- `oracle` will run and evaluate an oracle that knows who the speaker is if it's name (case-insensitive) is mentioned in the input
- `visualize` will apply a t-SNE over speaker name embeddings, either from raw BERT or fine-tuned model
 
Have a look at the `named_id.py` docstring for further details.

### `sidnet.py`

Named-Speaker Identification Network (inherits from `torch.nn.Module`), the docstring tells it all.

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
