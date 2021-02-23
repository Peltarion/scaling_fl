import os
import re
from contextlib import contextmanager
from itertools import chain
from typing import Generator, IO, Tuple, Union

import fsspec

ROUND_EXPRESSION = re.compile(r'(?:^|/)round-(\d+)\.pt$')

CheckpointSpecType = Union[Tuple[IO, int], Tuple[None, None]]


@contextmanager
def last_checkpoint(
        checkpoints_dir: str,
        mode: str = 'rb') -> Generator[CheckpointSpecType, None, None]:
    """Return the last checkpoint or None."""
    checkpoint_glob = os.path.join(checkpoints_dir, '*.pt')
    files = fsspec.open_files(checkpoint_glob, 'rb')
    if files:
        round_id = chain.from_iterable(ROUND_EXPRESSION.findall(f.path) for f in files)
        round_number, i = max((int(r), i) for i, r in enumerate(round_id))
        with files[i] as checkpoint_file:
            yield checkpoint_file, round_number
    else:
        yield None, None


@contextmanager
def open_checkpoint(checkpoints_dir: str,
                    round_: int,
                    mode: str = 'rb') -> Generator[IO, None, None]:
    """Return file like object of checkpoint file."""
    checkpoint_path = os.path.join(checkpoints_dir, f'round-{round_}.pt')
    with fsspec.open(checkpoint_path, mode) as checkpoint_file:
        yield checkpoint_file
