import torch
from torch.utils.data import Dataset


class ToyNextTokenDataset(Dataset):
    def __init__(self, n_samples: int, seq_len: int, vocab_size: int, pad_id: int = 0, device: str = "cpu"):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.device = device

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = torch.randint(1, self.vocab_size, (self.seq_len,), dtype=torch.long)
        y = torch.roll(x, shifts=-1, dims=0)
        y[-1] = self.pad_id
        return x, y