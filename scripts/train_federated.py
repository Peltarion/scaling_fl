import os
from collections import namedtuple
from functools import partial
from itertools import starmap
from typing import Callable, Dict, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
from sacred import Experiment
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from scaling_fl.data import agnews, imdb, spooky_author, yelp
from scaling_fl.data.split import split_data
from scaling_fl.data.text_classification import (
    TextClassificationCollator,
    TextClassificationDataset,
)
from scaling_fl.model.checkpoint import last_checkpoint, open_checkpoint
from scaling_fl.sacred import observer_from_env
from scaling_fl.train import Site, train_sites_parallel

TaskDataSpecType = Tuple[TextClassificationDataset, TextClassificationDataset, int]


def create_sites(site_ids, dataloaders, loss, optimiser, make_model, state_dict=None):
    devices = [f'cuda:{i}' for i in site_ids]
    models = [make_model().to(d) for d in devices]
    optimisers = (optimiser(m.parameters()) for m in models)
    losses = (loss for _ in site_ids)
    params = zip(site_ids, dataloaders, models, optimisers, losses)

    # Create sites and load previous state
    sites = list(starmap(Site, params))
    if state_dict is not None:
        for site in sites:
            site.load_state_dict(state_dict)

    return sites


def task_train_test(task: str, train_path: str, test_path: str) -> TaskDataSpecType:
    """Return training data, validation data and number of classes for a given task.

    Notes:
        The imdb data will ignore the train_path and test_path parameters.
    """
    task_train_test: Dict[str, Callable[[], TaskDataSpecType]] = {
        'agnews': lambda: agnews.train_test(train_path, test_path),
        'imdb': imdb.train_test,
        'spooky_author': lambda: spooky_author.train_test(train_path, test_path),
        'yelp': lambda: yelp.train_test(train_path, test_path)
    }
    train_test = task_train_test.get(task)
    if train_test is None:
        raise ValueError(f'Task {task} not implemented, must be one of:',
                         ', '.join(task_train_test.keys()))
    return train_test()


ex = Experiment('scaling_fl')


@ex.config
def config():
    learning_rate = 2e-5
    max_seq_len = 128
    batch_size = 32
    val_batch_size = 512
    num_rounds = 3
    num_local_epochs = 2
    site_ids = [0]
    loss = F.cross_entropy
    optimiser = optim.SGD
    task_name = 'spooky_author'
    train_path = None
    test_path = None
    model = 'albert'
    data_seed = 58923  # Required for correct splitting with multiple nodes
    iid_split = True  # Create IID or non-IID sites
    job_name = None  # Name of corresponding job
    world_size = dist.get_world_size()  # Number of nodes participating
    author = os.environ.get('USER_NAME')
    checkpoints_dir = None


# Allow the most common models to be specified by their nickname
MODEL_NICKNAMES = {
    'albert': 'albert-base-v1',
    'bert': 'bert-base-cased',
    'distilbert': 'distilbert-base-cased'
}


@ex.main
def main(_config, _run):
    Args = namedtuple('Args', _config.keys())
    args = Args(*_config.values())
    model_name = MODEL_NICKNAMES.get(args.model, args.model)

    # Get resume weights
    start_round = 1
    resume_state_dict = None
    with last_checkpoint(args.checkpoints_dir, 'rb') as (f, round_):
        if f is not None:
            resume_state_dict = torch.load(f)
            start_round = round_ + 1  # Start from next round
            print('Loaded checkpoint:', f.path)

    # Define the function here to include parameters such as the number of labels
    def make_model():
        return AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels)

    # Get datasets for task
    trainset, testset, num_labels = task_train_test(
        task=args.task_name, train_path=args.train_path, test_path=args.test_path)

    optimiser = partial(args.optimiser, lr=args.learning_rate)
    tokenizer = AutoTokenizer.from_pretrained(model_name, max_len=args.max_seq_len)
    collate_fn = TextClassificationCollator(tokenizer, args.max_seq_len)
    site_loaders = split_data(
        trainset,
        num_loaders=len(args.site_ids),
        batch_size=args.batch_size,
        seed=args.data_seed,
        iid_split=args.iid_split,
        drop_last=True,
        collate_fn=collate_fn
    )
    sites = create_sites(
        args.site_ids,
        site_loaders,
        args.loss,
        optimiser,
        make_model,
        resume_state_dict
    )
    resume_state_dict = None  # Hopefully trigger garbage collection early
    eval_loaders = split_data(
        testset,
        num_loaders=len(args.site_ids),
        batch_size=args.val_batch_size,
        seed=args.data_seed,
        iid_split=args.iid_split,
        drop_last=False,
        collate_fn=collate_fn
    )

    def train_callback(round_, metrics, sites):
        for name, value in metrics.items():
            _run.log_scalar(f'train.{name}', value, round_)

    def eval_callback(round_, metrics, sites):
        for name, value in metrics.items():
            _run.log_scalar(f'validation.{name}', value, round_)

        # Store model
        if args.checkpoints_dir is not None:
            with open_checkpoint(args.checkpoints_dir, round_, 'wb') as f:
                print(f'Saving checkpoint of round {round_} to:', f.path)
                torch.save(sites[0].state_dict(), f)

    train_sites_parallel(
        sites=sites,
        num_rounds=args.num_rounds,
        num_local_epochs=args.num_local_epochs,
        eval_dataloaders=eval_loaders,
        train_callback=train_callback,
        eval_callback=eval_callback,
        start_round=start_round
    )


if __name__ == '__main__':
    print('Initing process group')
    dist.init_process_group(backend='nccl', init_method='env://')
    if dist.get_rank() == 0:  # Only master reports results
        ex.observers.append(observer_from_env('job_name'))
    ex.run_commandline()
