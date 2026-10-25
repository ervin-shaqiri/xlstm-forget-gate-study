import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple, Dict, Any


def generate_even_pairs_sample(
    vocab_size: int = 3,
    context_length: int = 40,
    min_length: int = 3,
    max_length: int = 40,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Standard official Even Pairs task:
    Counts alternating transitions between 1 and 2.
    """
    rng = np.random.default_rng(seed)
    res = np.zeros(context_length, dtype=np.int32)
    res[:-1] = rng.integers(vocab_size - 1, size=context_length - 1) + 1
    size = int(rng.integers(min_length, max_length + 1))
    prediction_mask = np.zeros(context_length, dtype=np.int32)

    diffs = res[1:-1] - res[:-2]
    cumdiffs = np.cumsum(diffs)

    res[size:] = 0
    prediction_mask[size - 1] = 1
    res[size - 1] = int(np.abs(cumdiffs[size - 3])) + 1

    return res, prediction_mask


def get_controlled_forgetting_sequence(total_len: int = 40) -> Dict[str, Any]:
    """
    Constructs a controlled Even Pairs sequence with distinct phases:
    - Phase 1 (steps 0 to 11): Active alternating tokens (1, 2, 1, 2, 1, 2, ...)
    - Phase 2 (steps 12 to 27): Long distractor run of identical tokens (2, 2, 2, ...)
    - Phase 3 (steps 28 to 38): Resumed alternating transitions (1, 2, 1, 2, ...)
    - Step 39: Query/prediction position.
    """
    seq = np.zeros(total_len, dtype=np.int32)
    seq[:12] = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
    seq[12:28] = [2] * 16
    seq[28:39] = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1]

    diffs = seq[1:39] - seq[:38]
    cumdiffs = np.cumsum(diffs)
    target_val = int(np.abs(cumdiffs[39 - 3])) + 1
    seq[39] = target_val

    state_strip = []
    curr_diff = 0
    for i in range(total_len):
        if i == 0:
            state_strip.append(0)
        elif i < 39:
            d = abs(seq[i] - seq[i - 1])
            curr_diff = (curr_diff + d) % 2
            state_strip.append(curr_diff)
        else:
            state_strip.append(target_val % 2)

    phase_labels = []
    for i in range(total_len):
        if i < 12:
            phase_labels.append("Phase 1: Active")
        elif i < 28:
            phase_labels.append("Phase 2: Distractor (Forget)")
        elif i < 39:
            phase_labels.append("Phase 3: Active Target")
        else:
            phase_labels.append("Evaluation")

    return {
        "sequence": seq.tolist(),
        "token_names": [str(x) for x in seq[:39]],
        "state_strip": state_strip,
        "phase_labels": phase_labels,
        "target_val": target_val,
        "boundary_steps": [12, 28],
    }


class EvenPairsDataset(Dataset):
    def __init__(
        self,
        num_samples: int = 4000,
        seq_len: int = 40,
        min_seq_len: int = 3,
        seed: int = 42,
        vocab_size: int = 3,
    ):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.min_seq_len = min_seq_len
        self.seed = seed
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        res, mask = generate_even_pairs_sample(
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

