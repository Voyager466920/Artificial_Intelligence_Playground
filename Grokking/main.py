from math import ceil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from Transformer import GPTDecoder
from Train_Step import train_step
from Test_Step import test_step
from AdditionDataset import AdditionDataset


def build_loaders(batch_size=256, training_fraction=0.4, max_n=112):
    ds = AdditionDataset(max_n=max_n)

    train_size = int(len(ds) * training_fraction)
    test_size = len(ds) - train_size
    train_ds, test_ds = random_split(ds, [train_size, test_size])

    batch_size = min(batch_size, ceil(len(ds) / 2))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, test_loader, ds.vocab_size


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    max_n = 112
    training_fraction = 0.4
    batch_size = 256

    num_steps = 20000
    lr = 1e-3
    weight_decay = 1e-1

    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []

    train_loader, test_loader, vocab_size = build_loaders(
        batch_size=batch_size,
        training_fraction=training_fraction,
        max_n=max_n
    )

    model = GPTDecoder(
        vocab_size=vocab_size,
        seq_len=4,
        embedding_dim=128,
        num_heads=4,
        mlp_size=512,
        dropout=0.0,
        num_layers=4
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.98),
        weight_decay=weight_decay
    )

    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=50
    )

    steps_done = 0
    epoch = 0

    while steps_done < num_steps:
        epoch += 1
        remaining = num_steps - steps_done
        steps_this_epoch = min(len(train_loader), remaining)

        train_loss, train_acc = train_step(
            model,
            train_loader,
            loss_fn,
            optimizer,
            scheduler,
            device,
            max_steps=steps_this_epoch
        )

        steps_done += steps_this_epoch

        test_loss, test_acc = test_step(
            test_loader,
            model,
            loss_fn,
            device
        )

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        print(
            f"Epoch {epoch:5d} | steps {steps_done:7d}/{num_steps} | "
            f"Train Loss {train_loss:.6f} Acc {train_acc:.4f} | "
            f"Test Loss {test_loss:.6f} Acc {test_acc:.4f}"
        )

    torch.save(
        {
            "model": model.state_dict(),
            "vocab_size": vocab_size,
            "max_n": max_n
        },
        "adder_grokking.pt"
    )

    xs = list(range(1, len(train_losses) + 1))
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(xs, train_losses, label="Train Loss")
    plt.plot(xs, test_losses, label="Test Loss")
    plt.xlabel("Eval step")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(xs, train_accs, label="Train Acc")
    plt.plot(xs, test_accs, label="Test Acc")
    plt.xlabel("Eval step")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
