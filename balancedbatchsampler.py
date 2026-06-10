import torch
import numpy as np
from torch.utils.data import Sampler


class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, num_replicas=None, rank=None):

        assert batch_size % 2 == 0, "batch_size 必须是偶数"

        self.dataset = dataset
        self.batch_size = batch_size
        self.half = batch_size // 2

        self.num_replicas = num_replicas
        self.rank = rank

        # 分离 real / fake 索引
        self.real_indices = [i for i, (_, label) in enumerate(dataset.samples) if label == 0]
        self.fake_indices = [i for i, (_, label) in enumerate(dataset.samples) if label == 1]

        self.num_samples = min(len(self.real_indices), len(self.fake_indices))

        self.num_batches = self.num_samples // self.half

    def __iter__(self):

        real = np.random.permutation(self.real_indices)
        fake = np.random.permutation(self.fake_indices)

        for i in range(self.num_batches):

            r = real[i*self.half:(i+1)*self.half]
            f = fake[i*self.half:(i+1)*self.half]

            batch = np.concatenate([r, f])
            np.random.shuffle(batch)

            yield batch.tolist()

    def __len__(self):
        return self.num_batches
    
    def set_epoch(self, epoch):
        np.random.seed(epoch)