import torch

def test_step(dataloader, model, loss_fn, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for tokens, targets in dataloader:
            tokens, targets = tokens.to(device), targets.to(device)
            logits = model(tokens)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            preds = logits.argmax(-1)

            total_correct += (preds == targets).sum().item()
            total_tokens += targets.numel()
            total_loss += loss.item() * targets.numel()

    avg_loss = total_loss / total_tokens
    accuracy = total_correct / total_tokens
    return avg_loss, accuracy