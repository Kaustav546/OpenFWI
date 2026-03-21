# © 2022. Triad National Security, LLC. All rights reserved.

# This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos

# National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.

# Department of Energy/National Nuclear Security Administration. All rights in the program are

# reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear

# Security Administration. The Government is granted for itself and others acting on its behalf a

# nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare

# derivative works, distribute copies to the public, perform publicly and display publicly, and to permit

# others to do so.

import logging
import os
import numpy as np
import torch
import psutil
import warnings
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from tqdm import tqdm
import transforms as T

logger = logging.getLogger(__name__)


class FWIDataset(Dataset):
    ''' FWI dataset
    For convenience, in this class, a batch refers to a npy file
    instead of the batch used during training.

    Args:
        anno:             path to annotation file
        preload:          whether to preload batches into memory at init time
        sample_ratio:     downsample ratio for seismic data
        file_size:        number of samples in each npy file
        transform_data:   transformation applied to seismic data
        transform_label:  transformation applied to velocity labels
        memory_limit:     maximum RAM usage percentage before preloading stops
                          (default 90).  Batches that didn't fit at init time
                          are loaded on demand and the oldest cached batch is
                          evicted when memory is tight.  Even if a single
                          batch is larger than the remaining headroom, the
                          eviction loop is guaranteed to exit because the
                          dataset always keeps at least one batch resident in
                          the cache (see ``__getitem__`` for details).

    Performance notes
    -----------------
    Transforms are applied **once** inside ``_load_chunk`` (not per sample
    inside ``__getitem__``).  Loaded batches are stored as PyTorch
    shared-memory tensors so that DataLoader workers can read them via a
    direct pointer rather than copying data through IPC pipes.
    '''

    def __init__(self, anno, preload=True, sample_ratio=1, file_size=500,
                 transform_data=None, transform_label=None, memory_limit=90):
        if not os.path.exists(anno):
            raise FileNotFoundError(f'Annotation file {anno} not found.')
        self.preload = preload
        self.sample_ratio = sample_ratio
        self.file_size = file_size
        self.transform_data = transform_data
        self.transform_label = transform_label
        self.memory_limit = memory_limit
        with open(anno, 'r') as f:
            self.batches = [line.strip() for line in f if line.strip()]

        # LRU cache: batch_idx -> shared-memory tensor (transforms already applied)
        self.cache_data = {}
        self.cache_label = {}
        self.cache_order = []   # insertion-ordered list for LRU eviction
        # Suppress duplicate ResourceWarning for oversized batches (emitted at
        # most once per dataset instance, not once per cache miss).
        self._large_batch_warned = False

        # preload_complete guards against accessing data before init finishes.
        # For preload=False there is nothing to wait for, so it is True immediately.
        self.preload_complete = not preload
        if preload:
            self._preload_until_limit()
            self.preload_complete = True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_mem_usage(self):
        '''Return current RAM usage as a percentage (0-100).'''
        return psutil.virtual_memory().percent

    def _load_chunk(self, batch_idx):
        '''Load one .npy batch from disk, apply transforms once, and return as
        shared-memory PyTorch tensors.  Transforms are applied here so that
        DataLoader workers never need to recompute them.'''
        batch_line = self.batches[batch_idx]
        parts = [p.strip() for p in batch_line.split('\t')]
        data = np.load(parts[0])[:, :, ::self.sample_ratio, :].astype('float32')
        label = np.load(parts[1]).astype('float32') if len(parts) > 1 else None

        # Apply transforms once per chunk load – O(file_size) not O(epochs * file_size).
        if self.transform_data:
            buf = np.empty_like(data)
            for i in range(len(data)):
                buf[i] = self.transform_data(data[i])
            data = buf
        if self.transform_label and label is not None:
            lbuf = np.empty_like(label)
            for i in range(len(label)):
                lbuf[i] = self.transform_label(label[i])
            label = lbuf

        # Convert to shared-memory tensors: workers read via pointer, no IPC copy.
        data_t = torch.from_numpy(np.ascontiguousarray(data)).share_memory_()
        label_t = (torch.from_numpy(np.ascontiguousarray(label)).share_memory_()
                   if label is not None else None)
        return data_t, label_t

    def _preload_until_limit(self):
        '''Preload batches sequentially until RAM usage reaches
        ``memory_limit * 0.95`` percent.'''
        for i in tqdm(range(len(self.batches)), desc='Preloading batches'):
            if self._get_mem_usage() >= self.memory_limit * 0.95:
                break
            if i not in self.cache_data:
                data_t, label_t = self._load_chunk(i)
                self.cache_data[i] = data_t
                if label_t is not None:
                    self.cache_label[i] = label_t
                self.cache_order.append(i)

    def _unload_oldest(self):
        '''Evict the oldest cached batch to free memory.'''
        if not self.cache_order:
            return
        oldest = self.cache_order.pop(0)
        del self.cache_data[oldest]
        if oldest in self.cache_label:
            del self.cache_label[oldest]

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __getitem__(self, idx):
        if not self.preload_complete:
            raise RuntimeError('Attempted to access data before preload is complete.')
        batch_idx, sample_idx = idx // self.file_size, idx % self.file_size

        if batch_idx not in self.cache_data:
            # Evict the oldest cached batches until RAM usage drops below the
            # threshold, then load the required batch.
            #
            # Why "len(self.cache_order) > 1"?
            # ─────────────────────────────────
            # If a single .npy batch file occupies more RAM than the available
            # headroom (i.e. memory usage stays above `memory_limit * 0.95`
            # even after evicting everything), the while-loop condition would
            # never become False on its own.  That is a livelock: the loop
            # would spin forever calling `_unload_oldest`, which becomes a
            # no-op once the cache is empty.
            #
            # The `> 1` guard breaks the cycle: once only one batch remains
            # in the cache the loop exits unconditionally, letting `_load_chunk`
            # run.  Training continues — the memory limit is temporarily
            # exceeded for that one large batch — but it never deadlocks.
            while (self._get_mem_usage() >= self.memory_limit * 0.95
                   and len(self.cache_order) > 1):
                self._unload_oldest()
            # Warn when the limit could not be satisfied (batch larger than
            # available headroom).  This is expected behaviour; it is logged
            # so that operators can tune `memory_limit` if needed.
            # The warning is emitted at most once per dataset instance to avoid
            # flooding the logs during long training runs.
            if (not self._large_batch_warned
                    and self._get_mem_usage() >= self.memory_limit * 0.95
                    and len(self.cache_order) <= 1):
                warnings.warn(
                    f'Memory usage ({self._get_mem_usage():.1f}%) is still '
                    f'above the {self.memory_limit * 0.95:.1f}% threshold '
                    f'after evicting all but one cached batch.  The batch at '
                    f'index {batch_idx} is likely larger than the available '
                    f'headroom.  Consider increasing `memory_limit` or '
                    f'reducing `file_size`.  Training continues.',
                    ResourceWarning,
                    stacklevel=2,
                )
                self._large_batch_warned = True
            data_t, label_t = self._load_chunk(batch_idx)
            self.cache_data[batch_idx] = data_t
            if label_t is not None:
                self.cache_label[batch_idx] = label_t
            self.cache_order.append(batch_idx)

        # Transforms were already applied in _load_chunk; return directly.
        data = self.cache_data[batch_idx][sample_idx]
        label = (self.cache_label[batch_idx][sample_idx]
                 if batch_idx in self.cache_label else None)
        return data, label if label is not None else np.array([])

    def __len__(self):
        return len(self.batches) * self.file_size


if __name__ == '__main__':
    transform_data = Compose([
        T.LogTransform(k=1),
        T.MinMaxNormalize(T.log_transform(-61, k=1), T.log_transform(120, k=1))
    ])
    transform_label = Compose([
        T.MinMaxNormalize(2000, 6000)
    ])
    dataset = FWIDataset('relevant_files/temp.txt', transform_data=transform_data,
                         transform_label=transform_label, file_size=1)
    data, label = dataset[0]
    print(data.shape)
    print(label is None)
