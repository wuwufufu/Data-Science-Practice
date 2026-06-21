import torch
from torch.utils.data import DataLoader


def worker_init_fn(worker_id):
    """Avoid OpenMP/thread oversubscription when many DataLoader workers run."""
    torch.set_num_threads(1)


def build_dataloader(
    dataset,
    *,
    batch_size,
    shuffle=False,
    sampler=None,
    num_workers=0,
    pin_memory=True,
    persistent_workers=None,
    prefetch_factor=None,
    drop_last=False,
):
    loader_kwargs = dict(
        batch_size=batch_size,
        pin_memory=pin_memory,
        drop_last=drop_last,
        worker_init_fn=worker_init_fn,
    )

    if sampler is not None:
        loader_kwargs["sampler"] = sampler
        loader_kwargs["shuffle"] = False
    else:
        loader_kwargs["shuffle"] = shuffle

    loader_kwargs["num_workers"] = num_workers
    if num_workers > 0:
        if persistent_workers is None:
            persistent_workers = True
        if prefetch_factor is None:
            prefetch_factor = 2
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(dataset, **loader_kwargs)
