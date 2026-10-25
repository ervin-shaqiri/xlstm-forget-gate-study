import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, Any

TOKEN_MAP = {0: "PAD", 1: "R", 2: "+1", 3: "-1"}


def get_controlled_cycle_nav_sequence() -> Dict[str, Any]:
    """
    Returns the 40-token Custom Cycle Navigation evaluation sequence with explicit
    Reset tokens (R = 1) at timesteps 10 and 24.
    """
    seq = [
        2, 2, 2, 2, 2, 3, 3, 2, 2, 3,
        1,
        2, 2, 2, 2, 3, 2, 2, 3, 3, 2, 2, 2, 3,
        1,
        2, 2, 3, 2, 2, 2, 3, 3, 2, 2, 3, 2, 2, 3,
        0,
    ]
    token_names = [TOKEN_MAP[t] for t in seq]
    ground_truth = []
    curr = 0
    for t in seq[:-1]:
        if t == 1:
            curr = 0
        elif t == 2:
            curr = (curr + 1) % 6
        elif t == 3:
            curr = (curr - 1) % 6
        ground_truth.append(curr)
    ground_truth.append(curr)

    return {
        "sequence": seq,
        "token_names": token_names,
        "state_strip": ground_truth,
        "target_val": curr + 5,
        "boundary_steps": [10, 24],
    }


class CustomCycleNavDataset(Dataset):
    """
    Custom Cycle Navigation with Reset Token and NO Stay action.

    Alphabet:
      0: [PAD] - sequence padding / query trigger
      1: Reset Token (R) - instantaneously resets ring automaton state to 0
      2: Step Clockwise (+1 mod 6)
      3: Step Counter-Clockwise (-1 mod 6)
      4: Unused / Reserved
      5..10: Target state classes for ring positions 0..5 (mod 6)

    Total Vocab Size = 11.
    """

    def __init__(
        self,
        num_samples: int = 100000,
        seq_len: int = 80,
        min_seq_len: int = 10,
        cycle_size: int = 6,
        reset_prob: float = 0.15,
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.min_seq_len = min_seq_len
        self.cycle_size = cycle_size
        self.reset_prob = reset_prob
        self.seed = seed
        self.data = self._generate_data()

    def _generate_data(self):
        rng = np.random.default_rng(self.seed)
        samples = []

        p_reset = self.reset_prob
        p_dir = (1.0 - p_reset) / 2.0
        choices = [1, 2, 3]
        probs = [p_reset, p_dir, p_dir]

        for _ in range(self.num_samples):
            L = int(rng.integers(self.min_seq_len, self.seq_len + 1))
            seq = np.zeros(self.seq_len, dtype=np.int64)
            target = np.full(self.seq_len, -100, dtype=np.int64)

            tokens = rng.choice(choices, size=L - 1, p=probs)
            seq[: L - 1] = tokens

            curr_state = 0
            for t in range(L - 1):
                tok = tokens[t]
                if tok == 1:
                    curr_state = 0
                elif tok == 2:
                    curr_state = (curr_state + 1) % self.cycle_size
                elif tok == 3:
                    curr_state = (curr_state - 1) % self.cycle_size

            target[L - 1] = curr_state + 5
            seq[L - 1] = 0

            samples.append((torch.tensor(seq, dtype=torch.long), torch.tensor(target, dtype=torch.long)))

        return samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

