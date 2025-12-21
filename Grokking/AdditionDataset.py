import torch
from torch.utils.data import Dataset

class AdditionDataset(Dataset):
    def __init__(self, n_samples: int = 200000, max_n: int = 112):
        super().__init__()
        self.n_samples = n_samples
        self.max_n = max_n
        self.eq_id = max_n + 1  # 113
        self.vocab_size = self.eq_id + 1  # 114

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        a = torch.randint(0, self.max_n + 1, (1,)).item()
        b = torch.randint(0, self.max_n + 1, (1,)).item()
        if a + b > self.max_n:
            b = torch.randint(0, self.max_n - a + 1, (1,)).item()

        x_ids = torch.tensor([a, b, self.eq_id], dtype=torch.long)
        y_ids = torch.tensor([b, self.eq_id, a + b], dtype=torch.long)

        x_oh = torch.nn.functional.one_hot(x_ids, num_classes=self.vocab_size).float()
        return x_oh, y_ids
