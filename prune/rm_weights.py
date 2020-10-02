"""Usage: rm_weights.py [--yes] <train_dir>..."""

from docopt import docopt
from pathlib import Path
import yaml
from tqdm import tqdm
import warnings

from prune.named_id import EPOCH_FORMAT


def rm_weights(train_dir, yes=False):
    weights_path = train_dir / 'weights'
    if not weights_path.exists():
        warnings.warn(f"'{weights_path}' doesn't exist")
        return []

    keep = set()
    for validation_dir in train_dir.iterdir():
        if not validation_dir.is_dir():
            continue
        params = validation_dir / 'params.yml'
        if not params.exists():
            continue
        with open(params) as file:
            epoch = yaml.safe_load(file)['epoch']
            keep.add(EPOCH_FORMAT.format(epoch))
    if not keep and yes:
        warnings.warn(f"Skipping '{train_dir}' as it doesn't have any validation steps. "
                      f"Launch again without '--yes' and confirm manually if you want to delete all weights")
        return []
    todo = []
    for weight in sorted(weights_path.iterdir()):
        if weight.name not in keep:
            todo.append(weight)
    msg = f'About to delete {todo}\nKeeping {keep}\n'
    if yes:
        print(msg)
    else:
        answer = input(msg+"Proceed? [y/N]\n")
        if answer.lower() != 'y':
            return []
    for weight in todo:
        weight.unlink()


if __name__ == '__main__':
    args = docopt(__doc__)
    yes = args['--yes']
    for train_dir in tqdm(args['<train_dir>']):
        rm_weights(Path(train_dir), yes)