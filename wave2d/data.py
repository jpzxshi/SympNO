from pathlib import Path
import numpy as np


def random_field(grid, rng):
    ny, nx = grid
    ky, kx = np.fft.fftfreq(ny, d=1 / ny), np.fft.rfftfreq(nx, d=1 / nx)
    radius = np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    filt = np.exp(-((radius / 4) ** 4)) * (radius <= 8)
    field = np.fft.irfft2(np.fft.rfft2(rng.normal(size=grid)) * filt, s=grid)
    return field * (rng.uniform(0.25, 1.0) / max(np.max(np.abs(field)), np.finfo(float).tiny))


def step(state, grid, dt, c, cubic, substeps):
    fields = state.reshape(-1, 2, *grid)
    q, p = fields[:, 0], fields[:, 1]
    ky, kx = np.meshgrid(*[2 * np.pi * np.fft.fftfreq(n, d=1 / n) for n in grid], indexing="ij")
    w2 = c**2 * (ky * ky + kx * kx)
    omega = np.sqrt(w2)
    coefficients = [1.0]
    for order in (2, 4, 6):
        a = 1 / (2 - 2 ** (1 / (order + 1)))
        coefficients = [s * v for s in (a, 1 - 2 * a, a) for v in coefficients]
    for _ in range(substeps):
        for coefficient in coefficients:
            h = dt / substeps * coefficient
            p = p - 0.5 * h * cubic * q**3
            qh, ph = np.fft.fftn(q, axes=(-2, -1)), np.fft.fftn(p, axes=(-2, -1))
            co = np.cos(omega * h)
            si = h * np.sinc(omega * h / np.pi)
            q = np.fft.ifftn(co * qh + si * ph, axes=(-2, -1)).real
            p = np.fft.ifftn(-w2 * si * qh + co * ph, axes=(-2, -1)).real
            p = p - 0.5 * h * cubic * q**3
    return np.stack((q, p), axis=1).reshape(state.shape)


def generate_data(
    folder,
    samples=10000,
    grid=(128, 128),
    dt=0.01,
    seed=42,
    c=0.05,
    cubic=1.0,
    substeps=4,
    pairs_per_trajectory=50,
    sample_stride=20,
):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    shape = (samples, 2 * int(np.prod(grid)))
    # Write the 5.24 GB dataset a pair at a time.
    inputs = np.lib.format.open_memmap(
        folder / "inputs.npy", mode="w+", dtype=np.float64, shape=shape
    )
    targets = np.lib.format.open_memmap(
        folder / "targets.npy", mode="w+", dtype=np.float64, shape=shape
    )
    index = 0
    while index < samples:
        state = np.concatenate((random_field(grid, rng).ravel(), random_field(grid, rng).ravel()))[
            None
        ]
        for t in range((pairs_per_trajectory - 1) * sample_stride + 1):
            if t % sample_stride == 0:
                inputs[index] = state[0]
            state = step(state, grid, dt, c, cubic, substeps)
            if t % sample_stride == 0:
                targets[index] = state[0]
                index += 1
                if index == samples:
                    break
        print(f"Data: {index}/{samples}", flush=True)
    inputs.flush()
    targets.flush()


def load_data(folder, **parameters):
    folder = Path(folder)
    if not (folder / "inputs.npy").exists() or not (folder / "targets.npy").exists():
        generate_data(folder, **parameters)
    return np.load(folder / "inputs.npy", mmap_mode="r"), np.load(
        folder / "targets.npy", mmap_mode="r"
    )


if __name__ == "__main__":
    generate_data(Path(__file__).resolve().parent / "dataset")
