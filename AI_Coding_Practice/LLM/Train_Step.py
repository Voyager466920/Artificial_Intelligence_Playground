import torch

def train_step(model, dataloader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    total_tokens = 0

    last_correct = 0
    last_total = 0

    for tokens, targets in dataloader:
        tokens, targets = tokens.to(device), targets.to(device)
        logits = model(tokens)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds = logits.argmax(-1)
            total_loss += loss.item() * targets.numel()
            total_tokens += targets.numel()

            last_correct += (preds[:, -1] == targets[:, -1]).sum().item()
            last_total += targets.size(0)

    return total_loss / total_tokens, last_correct / last_total
