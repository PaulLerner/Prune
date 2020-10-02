"""Usage: rm_weights.py <train_dir>..."""

from docopt import docopt
from pathlib import Path
import yaml
from tqdm import tqdm

from prune.named_id import EPOCH_FORMAT


def rm_weights(train_dir):
    weights_path = train_dir / 'weights'
    keep = set()
    for validation_dir in train_dir.iterdir():
        if not validation_dir.is_dir():
            continue
        params = validation_dir / 'params.yml'
        if not params.exists():
            continue
        with open(params) as file:
            epoch = yaml.load(file)['epoch']
            keep.add(EPOCH_FORMAT.format(epoch))
    todo = []
    for weight in sorted(weights_path.iterdir()):
        if weight.name not in keep:
            todo.append(weight)
    msg = f'About to delete {todo}\nKeeping {keep}\n Proceed? [y/N]'
    answer = input(msg)
    if answer.lower() != 'y':
        return []
    for weight in todo:
        weight.unlink()


if __name__ == '__main__':
    args = docopt(__doc__)
    for train_dir in tqdm(args['<train_dir>']):
        rm_weights(Path(train_dir))