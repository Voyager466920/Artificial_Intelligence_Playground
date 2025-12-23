import torch
from torch.utils.data import Dataset

class AdditionDataset(Dataset):
    def __init__(self, max_n: int = 112):
        super().__init__()
        self.max_n = max_n
        self.mod = max_n + 1

        self.eq_id = self.mod
        self.plus_id = self.mod + 1
        self.vocab_size = self.mod + 2  # numbers + '=' + '+'

        self.pairs = [(a, b) for a in range(self.mod) for b in range(self.mod)]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        a, b = self.pairs[idx]
        x_ids = torch.tensor([a, self.plus_id, b, self.eq_id], dtype=torch.long)
        y = torch.tensor((a + b) % self.mod, dtype=torch.long)
        return x_ids, y
