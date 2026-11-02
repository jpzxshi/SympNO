import math
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader


def learning_rate(epoch, epochs, lr, min_lr, warmup_epochs):
    if warmup_epochs:
        factor = (
            epoch / warmup_epochs
            if epoch <= warmup_epochs
            else (1 + math.cos(math.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs))) / 2
        )
    else:
        factor = (1 + math.cos(math.pi * (epoch - 1) / epochs)) / 2
    return min_lr + (lr - min_lr) * factor


@torch.no_grad()
def evaluate(net, indices, batch, batch_size):
    net.eval()
    total, elements = 0.0, 0
    for start in range(0, len(indices), batch_size):
        x, y = batch(indices[start : start + batch_size])
        total += F.mse_loss(net(x), y, reduction="sum").item()
        elements += y.numel()
    mse = total / elements
    if not math.isfinite(mse):
        raise FloatingPointError("Nonfinite MSE")
    return mse


def train(
    net,
    data,
    output,
    epochs=500,
    batch_size=8,
    lr=1e-3,
    min_lr=1e-5,
    warmup_epochs=0,
    betas=(0.9, 0.999),
    eps=1e-12,
    weight_decay=0.0,
    fused=True,
    eval_every=5,
    eval_batch_size=256,
    seed=42,
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    x, y = data
    device = next(net.parameters()).device
    resident = len(net.grid) == 1
    if resident:
        x, y = (torch.from_numpy(np.array(a)).to(device) for a in (x, y))

    def batch(ids):
        if resident:
            ids = torch.as_tensor(ids, device=device)
            return x[ids], y[ids]
        return tuple(torch.from_numpy(np.array(a[np.asarray(ids)])).to(device) for a in (x, y))

    n = len(x)
    indices = np.random.RandomState(seed).permutation(n)
    train_ids, val_ids, test_ids = indices[n // 5 :], indices[n // 10 : n // 5], indices[: n // 10]
    train_ids = torch.as_tensor(train_ids, device=device if resident else "cpu")
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(
        net.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
        fused=fused and device.type == "cuda",
    )
    best, best_epoch, history = math.inf, 0, []

    for epoch in range(epochs + 1):
        rate, train_mse, val_mse = lr, np.nan, np.nan
        if epoch > 0:
            rate = learning_rate(epoch, epochs, lr, min_lr, warmup_epochs)
            for group in optimizer.param_groups:
                group["lr"] = rate
            if resident:
                order = torch.randperm(len(train_ids), device=device)
            else:
                order = torch.cat(
                    list(
                        DataLoader(
                            range(len(train_ids)),
                            batch_size=batch_size,
                            shuffle=True,
                            generator=generator,
                        )
                    )
                )
            net.train()
            total, elements = 0.0, 0
            for start in range(0, len(order), batch_size):
                xb, yb = batch(train_ids[order[start : start + batch_size]])
                optimizer.zero_grad(set_to_none=True)
                loss = F.mse_loss(net(xb), yb)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Nonfinite training loss")
                loss.backward()
                optimizer.step()
                total += loss.item() * yb.numel()
                elements += yb.numel()
            train_mse = total / elements

        if epoch % eval_every == 0 or epoch == epochs:
            val_mse = evaluate(net, val_ids, batch, eval_batch_size)
            if val_mse < best:
                best, best_epoch = val_mse, epoch
                torch.save(net.state_dict(), output / "best.pt")
        history.append((epoch, rate, train_mse, val_mse))
        np.savetxt(output / "loss.txt", history, header="epoch lr train_mse validation_mse")
        print(f"Epoch {epoch:4d}  train {train_mse:.6e}  validation {val_mse:.6e}", flush=True)

    net.load_state_dict(torch.load(output / "best.pt", map_location=device, weights_only=True))
    test_mse = evaluate(net, test_ids, batch, eval_batch_size)
    report = {"best_epoch": best_epoch, "validation_mse": best, "test_mse": test_mse}
    (output / "test.txt").write_text(
        "\n".join(f"{key}: {value}" for key, value in report.items()) + "\n", encoding="utf-8"
    )
    print("Test MSE:", test_mse, flush=True)
    return report
