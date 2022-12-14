# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# From DomainBed https://github.com/facebookresearch/DomainBed/blob/25f173caa689f20828629b2e42f90193f203fdfa/domainbed/lib/fast_data_loader.py

import torch

class _InfiniteSampler(torch.utils.data.Sampler):
    """Wraps another Sampler to yield an infinite stream."""
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch

class InfiniteDataLoader:
    def __init__(self, dataset, batch_size, collate_fn=None, weights=None, num_workers=1):
        super().__init__()

        if batch_size > len(dataset):
            batch_size = len(dataset)        
        
        if weights is not None:
            sampler = torch.utils.data.WeightedRandomSampler(weights,
                replacement=True,
                num_samples=batch_size)
        else:
            sampler = torch.utils.data.RandomSampler(dataset,
                replacement=True)

        if weights == None:
            weights = torch.ones(len(dataset))

        batch_sampler = torch.utils.data.BatchSampler(
            sampler,
            batch_size=batch_size,
            drop_last=True)

        self._infinite_iterator = iter(torch.utils.data.DataLoader(
            dataset,
            collate_fn=collate_fn,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler)
        ))

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def __len__(self):
        raise ValueError
