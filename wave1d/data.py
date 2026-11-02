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


def step(state, dt, c):
    n = state.shape[-1] // 2
    q, p = state[..., :n], state[..., n:]
    k = 2 * np.pi * np.fft.fftfreq(n, d=1 / n)
    w2 = c**2 * (k * k)
    omega = np.sqrt(w2)
    qh, ph = np.fft.fftn(q, axes=(-1,)), np.fft.fftn(p, axes=(-1,))
    co = np.cos(omega * dt)
    si = dt * np.sinc(omega * dt / np.pi)
    q = np.fft.ifftn(co * qh + si * ph, axes=(-1,)).real
    p = np.fft.ifftn(-w2 * si * qh + co * ph, axes=(-1,)).real
    return np.concatenate((q, p), axis=-1)


def generate_data(folder, samples=100000, grid=(100,), dt=0.001, seed=42, c=0.05):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    inputs = np.empty((samples, 2 * grid[0]), dtype=np.float64)
    targets = np.empty_like(inputs)
    for i in range(samples):
        inputs[i] = np.concatenate((random_field(grid[0], rng), random_field(grid[0], rng)))
    for start in range(0, samples, 64):
        targets[start : start + 64] = step(inputs[start : start + 64], dt, c)
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
