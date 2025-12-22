import torch
from torch.utils.data import Dataset

class AdditionDataset(Dataset):
    def __init__(self, max_n: int = 112):
        super().__init__()
        self.max_n = max_n
        self.eq_id = max_n + 1
        self.vocab_size = self.eq_id + 1
        self.mod = max_n + 1

        pairs = []
        for a in range(self.mod):
            for b in range(self.mod):
                pairs.append((a, b))
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        a, b = self.pairs[idx]
        x_ids = torch.tensor([a, b, self.eq_id], dtype=torch.long)
        y_ids = torch.tensor([b, self.eq_id, (a + b) % self.mod], dtype=torch.long)
        x_oh = torch.nn.functional.one_hot(x_ids, num_classes=self.vocab_size).float()
        return x_oh, y_ids
