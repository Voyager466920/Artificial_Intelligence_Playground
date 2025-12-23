import torch

def train_step(model, dataloader, loss_fn, optimizer, scheduler, device, max_steps=None):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0

    for batch_idx, (x_ids, y) in enumerate(dataloader):
        x_ids, y = x_ids.to(device), y.to(device)

        logits_last = model(x_ids)            # (B,V)
        loss = loss_fn(logits_last, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            preds = logits_last.argmax(dim=-1)
            total_correct += (preds == y).sum().item()
            total += y.size(0)
            total_loss += loss.item() * y.size(0)

        if max_steps is not None and (batch_idx + 1) >= max_steps:
            break

    return total_loss / max(total, 1), total_correct / max(total, 1)
