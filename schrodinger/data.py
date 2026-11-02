from pathlib import Path
import numpy as np


def random_field(n, rng):
    x = np.arange(n) / n
    modes = np.arange(1, 13, dtype=np.float64)
    phase = 2 * np.pi * modes[:, None] * x[None, :]
    decay = np.exp(-((modes / 6) ** 4))
    field = rng.normal() + (
        (rng.normal(size=12) * decay) @ np.cos(phase)
        + (rng.normal(size=12) * decay) @ np.sin(phase)
    )
    return field * (rng.uniform(0.25, 1.0) / max(np.max(np.abs(field)), np.finfo(float).tiny))


def step(state, dt, alpha):
    n = state.shape[-1] // 2
    q, p = state[..., :n], state[..., n:]
    k = 2 * np.pi * np.fft.fftfreq(n, d=1 / n)
    psi = np.fft.ifftn(
        np.fft.fftn(q + 1j * p, axes=(-1,)) * np.exp(-1j * alpha * (k * k) * dt), axes=(-1,)
    )
    return np.concatenate((psi.real, psi.imag), axis=-1)


def generate_data(folder, samples=100000, grid=(100,), dt=0.001, seed=42, alpha=0.1):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    inputs = np.empty((samples, 2 * grid[0]), dtype=np.float64)
    targets = np.empty_like(inputs)
    for i in range(samples):
        inputs[i] = np.concatenate((random_field(grid[0], rng), random_field(grid[0], rng)))
    for start in range(0, samples, 64):
        targets[start : start + 64] = step(inputs[start : start + 64], dt, alpha)
        if start % 1280 == 0 or start + 64 >= samples:
            print(f"Data: {min(start + 64, samples)}/{samples}", flush=True)
    np.save(folder / "inputs.npy", inputs)
    np.save(folder / "targets.npy", targets)


def load_data(folder, **parameters):
    folder = Path(folder)
    if not (folder / "inputs.npy").exists() or not (folder / "targets.npy").exists():
        generate_data(folder, **parameters)
    return np.load(folder / "inputs.npy", mmap_mode="r"), np.load(
        folder / "targets.npy", mmap_mode="r"
    )


if __name__ == "__main__":
    generate_data(Path(__file__).resolve().parent / "dataset")
