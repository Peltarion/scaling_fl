import torch


def label_entropy(labels: torch.Tensor, base: float = 2) -> torch.tensor:
    if len(labels) < 1:
        raise ValueError('Entropy for empty set undefined')

    _, counts = torch.unique(labels, return_counts=True)
    if len(counts) == 1:
        return torch.zeros([1], dtype=torch.float)

    p = counts / float(len(labels))
    base = torch.tensor(base, dtype=torch.float)
    return -torch.sum(p * torch.log(p) / torch.log(base))
