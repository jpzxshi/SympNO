import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sympno import SympNO
from training import train
from wave2d.data import load_data


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    path = Path(__file__).resolve().parent
    data_path = path / "dataset"
    output_path = path / "outputs"

    # Data
    data_config = {
        "samples": 10000, "grid": (128, 128), "dt": 0.01,
        "seed": 42, "c": 0.05, "cubic": 1.0, "substeps": 4,
        "pairs_per_trajectory": 50, "sample_stride": 20,
    }

    # Network
    model_config = {
        "grid_shape": data_config["grid"], "dt": data_config["dt"],
        "layers": 4, "width": 256, "fouriermodes": (16, 16),
        "mlp_width": 32, "activation": "gelu",
    }

    # Training
    train_config = {
        "epochs": 500, "batch_size": 8,
        "lr": 1e-3, "min_lr": 1e-5, "warmup_epochs": 5,
        "betas": (0.9, 0.999), "eps": 1e-12, "weight_decay": 0.0, "fused": True,
        "eval_every": 5, "eval_batch_size": 128, "seed": 42,
    }

    torch.manual_seed(train_config["seed"])
    net = SympNO(**model_config).to(device)
    data = load_data(data_path, **data_config)
    return train(net, data, output_path, **train_config)


if __name__ == "__main__":
    main()
