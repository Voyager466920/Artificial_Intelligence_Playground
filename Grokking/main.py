import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

import matplotlib.pyplot as plt

from Transformer import GPTDecoder
from Train_Step import train_step
from Test_Step import test_step
from AdditionDataset import AdditionDataset

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    batch_size = 256
    epochs = 100_000
    lr = 3e-4

    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []

    train_loader, test_loader, vocab_size = build_loaders(batch_size=batch_size, max_n=112)

    model = GPTDecoder(vocab_size=vocab_size, seq_len=3, embedding_dim=128, num_heads=4, dropout=0.0).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9,0.95), weight_decay=1e-1)

    for epoch in range(epochs):
        train_loss, train_acc = train_step(model, train_loader, loss_fn, optimizer, device)
        test_loss, test_acc = test_step(test_loader, model, loss_fn, device)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f}, Acc: {train_acc:.4f} | Test Loss: {test_loss:.6f}, Acc: {test_acc:.4f}")

    torch.save({"model": model.state_dict(), "vocab_size": vocab_size}, "Checkpoint/adder_gpt1.pt")
    print("saved: adder_gpt1.pt")

    epochs_range = range(1, epochs + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label="Train Loss")
    plt.plot(epochs_range, test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accs, label="Train Acc")
    plt.plot(epochs_range, test_accs, label="Test Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=200)
    plt.close()

    print("Saved training_curves.png")


def build_loaders(batch_size=256, split=0.05, max_n=112):
    ds = AdditionDataset(max_n=max_n)
    n_train = int(len(ds) * split)
    n_test = len(ds) - n_train
    train_ds, test_ds = random_split(ds, [n_train, n_test])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, test_loader, ds.vocab_size



if __name__ == "__main__":
    main()
