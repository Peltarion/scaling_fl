from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, Iterable, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor
from torch.multiprocessing import Pool, set_start_method
from torch.optim import Optimizer
from torch.utils.data import DataLoader


@dataclass
class Site:
    """A site in a federation."""

    id: int
    data: DataLoader
    model: nn.Module
    optimiser: Optimizer
    loss: Callable[[Tensor, Tensor], Tensor]

    def state_dict(self) -> Dict:
        """Return a state dict for restoring the site."""
        return {
            'model': self.model.state_dict()
        }

    def load_state_dict(self, state_dict: Dict):
        """Restore the site to the given state."""
        self.model.load_state_dict(state_dict['model'])


def eval_site(site: Site, dataloader: DataLoader) -> Dict[str, float]:
    """Evaluate the performance of the site model provided data."""
    site.model.eval()
    device = next(site.model.parameters()).device
    avg_loss = torch.zeros(1, dtype=torch.float, device=device)
    avg_acc = torch.zeros(1, dtype=torch.float, device=device)
    count = 0  # length of a dataloader may not always be correct
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            y_hat = site.model(x)
            if isinstance(y_hat, tuple):
                y_hat = y_hat[0]

            count += 1
            avg_loss += (site.loss(y_hat, y) - avg_loss) / count
            avg_acc += (y_hat.argmax(1).eq(y).float().mean() - avg_acc) / count
        return {'loss': avg_loss.item(), 'accuracy': avg_acc.item()}


def train_site(site: Site, num_epochs: int) -> Dict[str, float]:
    def train_epoch(data):
        avg_loss = 0
        avg_acc = 0
        count = 0  # length of a dataloader may not always be correct
        for x, y in data:
            x, y = x.to(device), y.to(device)
            y_hat = site.model(x)
            if isinstance(y_hat, tuple):
                y_hat = y_hat[0]

            loss = site.loss(y_hat, y)
            acc = y_hat.argmax(1).eq(y).float().mean()
            count += 1
            avg_loss += (loss.item() - avg_loss) / count
            avg_acc += (acc.item() - avg_acc) / count

            loss.backward()
            site.optimiser.step()
            site.optimiser.zero_grad()

        return avg_loss, avg_acc

    site.model.train()
    device = next(site.model.parameters()).device
    running_epoch_loss, running_epoch_acc = 0, 0
    for epoch in range(num_epochs):
        avg_loss, avg_acc = train_epoch(site.data)
        running_epoch_loss += avg_loss
        running_epoch_acc += avg_acc

    avg_epoch_loss = running_epoch_loss / num_epochs
    avg_epoch_acc = running_epoch_acc / num_epochs
    return {'loss': avg_epoch_loss, 'accuracy': avg_epoch_acc}


def federated_average(sites: Sequence[Site]):
    """Update all site models to the federated average."""
    num_all_sites = dist.get_world_size() * len(sites)
    with torch.no_grad():
        for parameters in zip(*(site.model.parameters() for site in sites)):
            # Skip parameters that do not update
            if not parameters[0].requires_grad:
                continue

            # Combine all into 1 tensor / GPU since required by reduce_multigpu
            device_avg: Dict[torch.device, torch.nn.parameter.Parameter] = {}
            for parameter in parameters:
                parameter.data /= num_all_sites
                if parameter.device in device_avg:
                    device_avg[parameter.device].data += parameter
                else:
                    device_avg[parameter.device] = parameter

            # Synchronize all parameters in device_mean
            dist.all_reduce_multigpu(
                list(device_avg.values()), op=dist.ReduceOp.SUM)

            # Disseminate average parameters stored on device to all
            for parameter in parameters:
                target_parameter = device_avg[parameter.device]
                if parameter is not target_parameter:
                    parameter.data.copy_(target_parameter.data)


def train_sites_parallel(
        sites: Sequence[Site],
        num_rounds: int,
        num_local_epochs: int,
        eval_dataloaders: Sequence[DataLoader],
        train_callback: Callable[[int, Dict[str, float], Sequence[Site]], None],
        eval_callback: Callable[[int, Dict[str, float], Sequence[Site]], None],
        start_round: int = 1):
    assert start_round >= 1
    set_start_method('spawn', force=True)
    for round_ in range(start_round, num_rounds + 1):
        print(f"Round {round_} of {num_rounds}")
        print("Training models on silos...")
        metrics = train_one_round(sites, num_local_epochs)
        if dist.get_rank() == 0:
            train_callback(round_, metrics, sites)
        print("Averaging models...")
        federated_average(sites)
        print("Evaluating...")
        metrics = eval_sites(sites, eval_dataloaders, round_)
        print('Done with eval')
        if dist.get_rank() == 0:
            eval_callback(round_, metrics, sites)
        print("Round completed")
    # Without this barrier some nodes may exit early, causing NCCL tensors to
    # lock when read.
    dist.barrier()


def synchronize_metrics(sites: Sequence[Site],
                        site_metrics: Iterable[Dict]) -> Dict[str, float]:
    it = iter(site_metrics)
    average_metrics = next(it).copy()
    for metrics in it:
        # Fail if not all sites produce the same metrics
        for name, value in metrics.items():
            average_metrics[name] += value

    device = next(sites[0].model.parameters()).device
    num_all_sites = dist.get_world_size() * len(sites)
    for name, value in average_metrics.items():
        value = torch.tensor(value / num_all_sites, device=device)
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        average_metrics[name] = value.item()
    return average_metrics


def train_one_round(
        sites: Sequence[Site],
        num_epochs: int) -> Dict[str, float]:
    train = partial(train_site, num_epochs=num_epochs)
    with Pool(processes=len(sites)) as pool:
        results = pool.map_async(train, sites)
        per_site_metrics = results.get()
    return synchronize_metrics(sites, per_site_metrics)


def eval_sites(
        sites: Sequence[Site],
        data_loaders: Sequence[DataLoader],
        round_: int) -> Dict[str, float]:
    with Pool(processes=len(sites)) as pool:
        results = pool.starmap_async(eval_site, zip(sites, data_loaders))
        per_site_metrics = results.get()
    return synchronize_metrics(sites, per_site_metrics)
