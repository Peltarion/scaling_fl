from functools import partial
from typing import Callable, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

from ..metrics import label_entropy

CollateFnInput = Sequence[Tuple[str, int]]
CollateFnOutput = Tuple[torch.Tensor, torch.Tensor]
CollateFnType = Callable[[CollateFnInput], CollateFnOutput]


def _get_labels(dataset):
    """Return the labels of the dataset."""
    if isinstance(dataset, TensorDataset):
        return dataset.tensors[-1]
    else:
        return torch.tensor([y for (x, y) in dataset])


def _split_randomly(dataset: Dataset,
                    num_parts: int,
                    seed: int = None) -> List[Subset]:
    """Return randomly chosen splits of the dataset."""
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    n = len(dataset) // num_parts
    lengths = [n for _ in range(num_parts)]
    subset = Subset(dataset, range(sum(lengths)))
    return torch.utils.data.random_split(subset, lengths, generator=generator)


def _split_ordered(dataset: Dataset, num_parts: int) -> List[Subset]:
    """Return splits made by first ordering by label and splitting in order."""
    n = len(dataset) // num_parts
    labels = _get_labels(dataset)
    sorted_indices = torch.argsort(labels).tolist()
    indices = [sorted_indices[(i * n):((i + 1) * n)] for i in range(num_parts)]
    return [Subset(dataset, index) for index in indices]


def split_data(dataset: Dataset,
               batch_size: int,
               num_loaders: int,
               seed: int,
               iid_split: bool,
               drop_last: bool,
               collate_fn: Optional[CollateFnType] = None) -> List[DataLoader]:
    """Split dataset into multiple data loaders.

    Note:
        Requires default distributed process group to be initialized.

    Args:
        dataset: Dataset with items like (x, y) to separate into loaders
        batch_size: Batch size for each data loader
        num_loaders: Number of splits per node in the world
        seed: Seed used when randomly splitting
        iid_split: Split randomly, otherwise order by label and split in order
        drop_last: Drop the last batch if smaller than batch size

    Returns:
        List of data loaders
    """
    rank, world_size = dist.get_rank(), dist.get_world_size()
    num_world_loaders = num_loaders * world_size
    if drop_last and len(dataset) // num_world_loaders < batch_size:
        raise ValueError('Dataset contains too few examples and would not'
                         'generate any full batches.')

    if iid_split:
        datasets = _split_randomly(dataset, num_world_loaders)
    else:
        datasets = _split_ordered(dataset, num_world_loaders)

    # Debug output for split
    print('Full dataset entropy:', label_entropy(_get_labels(dataset)).item())
    subset_entropies = [label_entropy(_get_labels(d)).item() for d in datasets]
    print('Subset entropies:', subset_entropies)

    dataloader = partial(DataLoader,
                         batch_size=batch_size,
                         drop_last=drop_last,
                         pin_memory=True)
    if collate_fn is not None:
        dataloader = partial(dataloader, collate_fn=collate_fn)
    selection = slice(rank * num_loaders, (rank + 1) * num_loaders)
    return list(map(dataloader, datasets[selection]))
