"""Symplectic neural operator for periodic 1D and 2D fields."""

import math
import torch
from torch import nn


class GradientModule(nn.Module):
    """Symplectic shear with gradient K* [a(x) activation(Ku + b(x))]."""

    def __init__(self, grid_shape, width, fouriermodes, mlp_width, activation, mode):
        super().__init__()
        self.grid = tuple(grid_shape)
        self.width = width
        self.modes = fouriermodes
        self.mode = mode
        self.act = {"gelu": nn.GELU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid, "relu": nn.ReLU}[
            activation
        ]()
        dtype = torch.float64

        params = {"w": nn.Parameter(0.001 * torch.randn(width, 1, dtype=dtype))}
        if len(self.grid) == 1:
            shapes = {"R_re": (fouriermodes, width), "R_im": (fouriermodes, width)}
        else:
            ny, nx = self.grid
            my = min(fouriermodes[0], (ny + 1) // 2)
            mx = min(fouriermodes[1], nx // 2 + 1)
            shapes = {
                "R_positive_re": (my, mx, width),
                "R_positive_im": (my, mx, width),
                "R_negative_re": (my - 1, mx, width),
                "R_negative_im": (my - 1, mx, width),
            }
        for name, shape in shapes.items():
            params[name] = nn.Parameter(0.001 * torch.randn(*shape, dtype=dtype))
        self.ps = nn.ParameterDict(params)

        features = 4 * len(self.grid)
        self.a_mlp = nn.Sequential(
            nn.Linear(features, mlp_width, dtype=dtype),
            nn.GELU(),
            nn.Linear(mlp_width, width, dtype=dtype),
        )
        self.b_mlp = nn.Sequential(
            nn.Linear(features, mlp_width, dtype=dtype),
            nn.GELU(),
            nn.Linear(mlp_width, width, dtype=dtype),
        )

    def forward(self, qph):
        q, p, h = qph
        if self.mode == "up":
            return q + self.gradient(p) * h, p
        return q, p + self.gradient(q) * h

    def gradient(self, x):
        shape = x.shape
        x = x.reshape(-1, *self.grid)
        axes = [torch.arange(n, dtype=x.dtype, device=x.device) / n for n in self.grid]
        coordinates = torch.meshgrid(*axes, indexing="ij")[::-1]
        features = torch.stack(
            [
                f(k * torch.pi * coordinate)
                for coordinate in coordinates
                for k in (2, 4)
                for f in (torch.sin, torch.cos)
            ],
            dim=-1,
        )
        a = self.a_mlp(features).unsqueeze(0)
        b = self.b_mlp(features).unsqueeze(0)

        if len(self.grid) == 1:
            local = x.unsqueeze(-1) @ self.ps["w"].t()
        else:
            local = x.unsqueeze(-1) * self.ps["w"].view(1, 1, 1, self.width)
        z = self.spectral(x) + local + b
        g = a * self.act(z)
        if len(self.grid) == 1:
            grad = self.spectral(g, adjoint=True) + (g @ self.ps["w"]).squeeze(-1)
        else:
            local_adjoint = (g * self.ps["w"].view(1, 1, 1, self.width)).sum(dim=-1)
            grad = self.spectral(g, adjoint=True) + local_adjoint
        return grad.reshape(shape)

    def spectral(self, x, adjoint=False):
        if len(self.grid) == 1:
            n = self.grid[0]
            x_ft = torch.fft.rfft(x, dim=1)
            m = min(self.modes, n // 2 + 1)
            R = self.spectrum_1d(m, n)
            shape = (len(x), n // 2 + 1) if adjoint else (len(x), n // 2 + 1, self.width)
            out = x_ft.new_zeros(shape)
            if adjoint:
                out[:, :m] = torch.einsum("bmc,mc->bm", x_ft[:, :m, :], R.conj())
            else:
                out[:, :m, :] = torch.einsum("bm,mc->bmc", x_ft[:, :m], R)
            return torch.fft.irfft(out, n=n, dim=1)

        x_ft = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        R = self.spectrum_2d()
        if adjoint:
            out = (x_ft * R.conj().unsqueeze(0)).sum(dim=-1)
        else:
            out = x_ft.unsqueeze(-1) * R.unsqueeze(0)
        return torch.fft.irfft2(out, s=self.grid, dim=(1, 2), norm="ortho")

    def spectrum_1d(self, m, n):
        real, imag = self.ps["R_re"][:m], self.ps["R_im"][:m]
        mask = torch.ones_like(imag)
        mask[0] = 0
        if n % 2 == 0 and m == n // 2 + 1:
            mask[-1] = 0
        return torch.complex(real, imag * mask)

    def spectrum_2d(self):
        ny, nx = self.grid
        positive = torch.complex(self.ps["R_positive_re"], self.ps["R_positive_im"])
        negative = torch.complex(self.ps["R_negative_re"], self.ps["R_negative_im"])
        my, mx = positive.shape[:2]
        R = positive.new_zeros((ny, nx // 2 + 1, self.width))
        nyquist = nx % 2 == 0 and mx == nx // 2 + 1
        stop = mx - 1 if nyquist else mx
        if stop > 1:
            R[:my, 1:stop] = positive[:, 1:stop]
            if my > 1:
                R[-(my - 1) :, 1:stop] = negative[:, 1:stop]

        # Real FFT boundary columns require Hermitian symmetry along y.
        for column in [0, nx // 2] if nyquist else [0]:
            real = positive[0, column].real
            R[0, column] = torch.complex(real, torch.zeros_like(real))
            if my > 1:
                R[1:my, column] = positive[1:my, column]
                R[-(my - 1) :, column] = positive[1:my, column].conj().flip(0)
        return R


class SympNO(nn.Module):
    """Alternating q/p shears. Input and output: flattened [q, p]."""

    def __init__(
        self, grid_shape, layers, width, fouriermodes, mlp_width, activation="gelu", dt=0.001
    ):
        super().__init__()
        self.grid = tuple(grid_shape)
        self.dim = 2 * math.prod(self.grid)
        self.dt = dt
        if len(self.grid) not in (1, 2) or min(self.grid) < 2:
            raise ValueError("grid_shape must describe a 1D or 2D grid.")
        if min(layers, width, mlp_width) < 1:
            raise ValueError("layers, width and mlp_width must be positive.")
        modes = (fouriermodes,) if isinstance(fouriermodes, int) else tuple(fouriermodes)
        if len(modes) != len(self.grid) or min(modes) < 1:
            raise ValueError("Specify a positive Fourier mode count for each grid axis.")
        fouriermodes = modes[0] if len(self.grid) == 1 else modes
        self.ms = nn.ModuleDict(
            {
                f"GradM{i + 1}": GradientModule(
                    self.grid,
                    width,
                    fouriermodes,
                    mlp_width,
                    activation,
                    "up" if i % 2 == 0 else "low",
                )
                for i in range(layers)
            }
        )

    def forward(self, state):
        state, h = state if isinstance(state, (tuple, list)) else (state, self.dt)
        if state.ndim not in (1, 2) or state.shape[-1] != self.dim:
            raise ValueError(f"Expected ({self.dim},) or (batch, {self.dim}).")
        h = torch.as_tensor(h, dtype=state.dtype, device=state.device)
        if state.ndim == 2 and h.ndim == 1:
            h = h[:, None]
        q, p = state[..., : self.dim // 2], state[..., self.dim // 2 :]
        for layer in self.ms.values():
            q, p = layer((q, p, h))
        return torch.cat((q, p), dim=-1)

    @torch.no_grad()
    def predict(self, state, steps=1, dt=None, include_initial=True):
        """The penultimate output axis is time."""
        if steps < 0 or (steps == 0 and not include_initial):
            raise ValueError("Use positive steps, or include the initial state.")
        parameter = next(self.parameters())
        state = torch.as_tensor(state, dtype=parameter.dtype, device=parameter.device)
        trajectory = [state] if include_initial else []
        for _ in range(steps):
            state = self(state if dt is None else (state, dt))
            trajectory.append(state)
        return torch.stack(trajectory, dim=-2)
