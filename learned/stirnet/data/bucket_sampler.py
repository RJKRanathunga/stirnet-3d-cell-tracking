from __future__ import annotations

from collections import defaultdict
import random
from torch.utils.data import Sampler


class ShapeBucketBatchSampler(Sampler[list[int]]):
    """Batch indices by dataset-provided shape keys.

    The dataset must expose `shape_key(index)` returning a hashable spacing/patch-shape key.
    """
    def __init__(self,dataset,batch_size:int,shuffle:bool=True,drop_last:bool=False):
        self.dataset=dataset;self.batch_size=batch_size;self.shuffle=shuffle;self.drop_last=drop_last
        buckets=defaultdict(list)
        for i in range(len(dataset)): buckets[dataset.shape_key(i)].append(i)
        self.buckets=dict(buckets)

    def __iter__(self):
        batches=[]
        for ids in self.buckets.values():
            ids=list(ids)
            if self.shuffle: random.shuffle(ids)
            for i in range(0,len(ids),self.batch_size):
                b=ids[i:i+self.batch_size]
                if len(b)==self.batch_size or not self.drop_last:batches.append(b)
        if self.shuffle: random.shuffle(batches)
        yield from batches

    def __len__(self):
        total=0
        for ids in self.buckets.values():
            total += len(ids)//self.batch_size if self.drop_last else (len(ids)+self.batch_size-1)//self.batch_size
        return total
