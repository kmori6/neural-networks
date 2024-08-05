import random
from itertools import chain
from typing import Iterator

from torch.utils.data.sampler import Sampler


class LengthBucketSampler(Sampler):
    def __init__(self, num_samples: int, batch_size: int, shuffle: bool):
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[int]:
        indices = list(range(self.num_samples))
        chunks = [indices[i : i + self.batch_size] for i in range(0, self.num_samples, self.batch_size)]
        if self.shuffle:
            chunks = chunks[:1] + random.sample(chunks[1:-1], len(chunks) - 2) + chunks[-1:]
        return iter(chain(*chunks))

    def __len__(self) -> int:
        return self.num_samples
