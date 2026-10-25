import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple, Dict, Any


def generate_parity_sample(
    vocab_size: int = 3,
    context_length: int = 40,
    min_length: int = 3,
    max_length: int = 40,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Standard official Parity Check task:
    Counts the modulo-2 sum of tokens in {1, 2} (where 1 -> 0 and 2 -> 1).
    Returns class 1 for even parity and class 2 for odd parity at position size - 1.
    """
    rng = np.random.default_rng(seed)
    res = np.zeros(context_length, dtype=np.int32)
    res[:-1] = rng.integers(1, vocab_size, size=context_length - 1)
    size = int(rng.integers(min_length, max_length + 1))
    prediction_mask = np.zeros(context_length, dtype=np.int32)

    res[size - 1 :] = 0
    par = int((np.sum(res[: size - 1]) - size + 1) % (vocab_size - 1) + 1)
    res[size - 1] = par
    prediction_mask[size - 1] = 1

    return res, prediction_mask


def get_controlled_parity_sequence(total_len: int = 40) -> Dict[str, Any]:
    """
    Constructs a deterministic 40-token Parity Check evaluation sequence with
    structured alternating parity flips (token 2) and stationary holds (token 1).
    """
    seq = np.zeros(total_len, dtype=np.int32)
    # Phase 1 (0..11): frequent parity flips (2, 1, 2, 1, 2, 2, 1, 2, 1, 2, 2, 1)
    seq[:12] = [2, 1, 2, 1, 2, 2, 1, 2, 1, 2, 2, 1]
    # Phase 2 (12..27): stationary plateau of 1s (no parity change)
    seq[12:28] = [1] * 16
    # Phase 3 (28..38): active flips
    seq[28:39] = [2, 2, 1, 2, 1, 2, 2, 1, 2, 1, 2]

    par = int((np.sum(seq[:39]) - 39) % 2 + 1)
    seq[39] = par

    state_strip = []
    running = 0
    for i in range(total_len - 1):
        if seq[i] == 2:
            running = (running + 1) % 2
        state_strip.append(running)
    state_strip.append(running)

    return {
        "sequence": seq.tolist(),
        "token_names": [str(x) for x in seq[:39]],
        "state_strip": state_strip,
        "target_val": par,
        "boundary_steps": [12, 28],
    }


class ParityDataset(Dataset):
    def __init__(
        self,
        num_samples: int = 4000,
        seq_len: int = 40,
        min_seq_len: int = 3,
        seed: int = 42,
        vocab_size: int = 3,
        pregenerate: bool = False,
    ):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.min_seq_len = min_seq_len
        self.seed = seed
        self.vocab_size = vocab_size
        self.pregenerate = pregenerate
        self.data = self._generate_all() if pregenerate else None

    def _generate_single(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        res, mask = generate_parity_sample(
            vocab_size=self.vocab_size,
            context_length=self.seq_len,
            min_length=self.min_seq_len,
            max_length=self.seq_len,
            seed=self.seed + idx,
        )
        mask = mask.astype(bool)
        seq_in = np.where(mask, 0, res)
        seq_out = np.where(mask, res, -100)
        x = seq_in[:-1]
        y = seq_out[1:]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

    def _generate_all(self):
        rng = np.random.default_rng(self.seed)
        L = self.seq_len
        tokens = rng.integers(1, self.vocab_size, size=(self.num_samples, L - 1), dtype=np.int64)
        sizes = rng.integers(self.min_seq_len, L + 1, size=(self.num_samples,), dtype=np.int64)

        xs = np.zeros((self.num_samples, L - 1), dtype=np.int64)
        ys = np.full((self.num_samples, L - 1), -100, dtype=np.int64)

        for i in range(self.num_samples):
            sz = sizes[i]
            xs[i, : sz - 1] = tokens[i, : sz - 1]
            par = int((np.sum(tokens[i, : sz - 1]) - sz + 1) % (self.vocab_size - 1) + 1)
            ys[i, sz - 2] = par

        return torch.from_numpy(xs), torch.from_numpy(ys)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.data is not None:
            return self.data[0][idx], self.data[1][idx]
        return self._generate_single(idx)

