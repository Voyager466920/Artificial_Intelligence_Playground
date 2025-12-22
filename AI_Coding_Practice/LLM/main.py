import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from Transformer import TransformerDecoderBlock
from Train_Step import train_step
from Test_Step import test_step
from Dataset import ToyNextTokenDataset


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = 5
    batch_size = 32
    seq_len = 64
    vocab_size = 1000
    pad_id = 0

    d_model = 256
    n_heads = 8
    d_ff = 1024
    n_layers = 4
    dropout = 0.1

    lr = 3e-4
    weight_decay = 0.01

    train_dataset = ToyNextTokenDataset(
        n_samples=4096,
        seq_len=seq_len,
        vocab_size=vocab_size,
        pad_id=pad_id,
    )

    test_dataset = ToyNextTokenDataset(
        n_samples=1024,
        seq_len=seq_len,
        vocab_size=vocab_size,
        pad_id=pad_id,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = TransformerDecoderBlock(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_layers=n_layers,
        max_len=seq_len,
        dropout=dropout,
        pad_id=pad_id,
    ).to(device)

    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(1, epochs + 1):
        train_loss, train_last_acc = train_step(
            model, train_loader, loss_fn, optimizer, device
        )

        test_loss, test_last_acc = test_step(
            test_loader, model, loss_fn, device
        )

        train_ppl = math.exp(train_loss) if train_loss < 50 else float("inf")
        test_ppl = math.exp(test_loss) if test_loss < 50 else float("inf")

        print(
            f"epoch {epoch:03d} | "
            f"train loss {train_loss:.4f} ppl {train_ppl:.2f} last_acc {train_last_acc:.4f} | "
            f"test loss {test_loss:.4f} ppl {test_ppl:.2f} last_acc {test_last_acc:.4f}"
        )


if __name__ == "__main__":
    main()
