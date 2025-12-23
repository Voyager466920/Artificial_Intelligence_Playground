import torch

def test_step(dataloader, model, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0

    with torch.no_grad():
        for x_ids, y in dataloader:
            x_ids, y = x_ids.to(device), y.to(device)
            logits_last = model(x_ids)
            loss = loss_fn(logits_last, y)

            preds = logits_last.argmax(dim=-1)
            total_correct += (preds == y).sum().item()
            total += y.size(0)
            total_loss += loss.item() * y.size(0)

    return total_loss / max(total, 1), total_correct / max(total, 1)
