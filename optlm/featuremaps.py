"""Feature maps A-H of protocol 3.4, implemented in torch (GPU) for the experiment loops.

Input h: (n, d) float32, already standardized with train-pool statistics.
Output r: (n, D) float32 (standardized afterwards by the caller).

| ID | name       | definition                                        | knob (4 values)            |
|----|------------|---------------------------------------------------|----------------------------|
| A  | identity   | r = h (D = d)                                     | none                       |
| B  | linear     | r = R h, R_ij ~ N(0, 1/d)                         | none                       |
| C  | quadratic  | r = (R h)^2                                       | none                       |
| D  | tanh       | r = tanh(s R h)                                   | s in {0.25, 0.5, 1, 2}     |
| E  | cintensity | r = |(R1 + i R2) h|^2, R1, R2 ~ N(0, 1/(2d))       | none                       |
| F  | F-scatter  | r = |U exp(i phi)|^2, U ~ CN(0, 1/d)               | beta in {pi/4,pi/2,pi,2pi} |
| F  | F-fourier  | SLM grid -> unitary 2-D FFT -> D camera pixels    | beta in {pi/4,pi/2,pi,2pi} |
| H  | rff        | r = sqrt(2/D) cos(W h + b), W ~ N(0, sigma^-2)    | sigma in {0.5,1,2,4} sqrt(d)|
| F2 | F2-scatter / F2-fourier: protocol v2 encoding phi = 2 pi (tanh(h/tau)+1)/2 | tau in {1,4,16,64} |
(G, the trainable MLP, lives in optlm.heads.)

Seeding: every random matrix comes from np.random.default_rng([map_code, map_seed]).
- B, C and D share one R per map seed (map code 1), so they differ only in the nonlinearity.
- F-scatter / F-fourier take U / the pixel set from the tested numpy classes in optics.maps.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from optics.maps import FFourier, FScatter

# F2-* (protocol amendment v2) share F-*'s map codes, so v1 and v2 use identical optics per seed.
MAP_CODES = {"B": 1, "C": 1, "D": 1, "E": 4, "F-scatter": 6, "F-fourier": 7, "H": 8,
             "F2-scatter": 6, "F2-fourier": 7}
KNOB_NAMES = {"D": "s", "F-scatter": "beta", "F-fourier": "beta", "H": "sigma_over_sqrt_d",
              "F2-scatter": "tau", "F2-fourier": "tau"}
F2_BETA = 2 * math.pi  # v2: full SLM phase range


def _gauss(rng, shape, std):
    return torch.from_numpy((rng.standard_normal(shape) * std).astype(np.float32))


class FeatureMap:
    """Random matrices are drawn once per (map, map seed); the knob is applied at call time."""

    def __init__(self, name: str, d: int, D: int, seed: int, device: str = "cpu", chunk: int = 4096):
        self.name, self.d, self.D, self.seed, self.device, self.chunk = name, d, D, seed, device, chunk
        if name == "A":
            if D != d:
                raise ValueError("identity map requires D = d")
            return
        rng = np.random.default_rng([MAP_CODES[name], seed])
        if name in ("B", "C", "D"):
            self.R = _gauss(rng, (D, d), 1 / math.sqrt(d)).to(device)
        elif name == "E":
            self.R1 = _gauss(rng, (D, d), 1 / math.sqrt(2 * d)).to(device)
            self.R2 = _gauss(rng, (D, d), 1 / math.sqrt(2 * d)).to(device)
        elif name in ("F-scatter", "F2-scatter"):
            m = FScatter(d, D, seed=[MAP_CODES[name], seed], complex_dtype=np.complex128)
            self.Ur = torch.from_numpy(m.U.real.astype(np.float32)).to(device)
            self.Ui = torch.from_numpy(m.U.imag.astype(np.float32)).to(device)
        elif name in ("F-fourier", "F2-fourier"):
            # 64 x 64 far-field window (4096 camera pixels); larger D needs a finer camera grid (128 x 128)
            m = FFourier(d, D, seed=[MAP_CODES[name], seed], fft_size=64 if D <= 4096 else 128)
            self.G, self.M, self.offset = m.G, m.M, m.offset
            self.pad_amplitude, self.pad_phase = m.pad_amplitude, m.pad_phase
            self.pixels = torch.from_numpy(m.pixels.astype(np.int64)).to(device)
        elif name == "H":
            self.W0 = _gauss(rng, (D, d), 1.0).to(device)
            self.b = torch.from_numpy(rng.uniform(0, 2 * math.pi, D).astype(np.float32)).to(device)
        else:
            raise ValueError(f"unknown map {name!r}")

    # ------------------------------------------------------------------ per-chunk transforms
    def _phase(self, h, knob):
        """v1 (F-*): knob = beta, phi = beta (tanh h + 1)/2.  v2 (F2-*): knob = tau, phi = 2 pi (tanh(h/tau) + 1)/2."""
        if self.name.startswith("F2-"):
            return F2_BETA * (torch.tanh(h / float(knob)) + 1.0) / 2.0
        return float(knob) * (torch.tanh(h) + 1.0) / 2.0

    def _apply(self, h: torch.Tensor, knob) -> torch.Tensor:
        n = self.name
        if n == "A":
            return h
        if n == "B":
            return h @ self.R.T
        if n == "C":
            return (h @ self.R.T) ** 2
        if n == "D":
            return torch.tanh(float(knob) * (h @ self.R.T))
        if n == "E":
            return (h @ self.R1.T) ** 2 + (h @ self.R2.T) ** 2
        if n in ("F-scatter", "F2-scatter"):
            phi = self._phase(h, knob)
            c, s = torch.cos(phi), torch.sin(phi)
            ur = c @ self.Ur.T - s @ self.Ui.T
            ui = c @ self.Ui.T + s @ self.Ur.T
            return ur * ur + ui * ui
        if n in ("F-fourier", "F2-fourier"):
            phi = self._phase(h, knob)
            k = h.shape[0]
            g2 = self.G * self.G
            field = torch.empty((k, g2), dtype=torch.complex64, device=h.device)
            field[:, : self.d] = torch.polar(torch.ones_like(phi), phi)
            pad = self.pad_amplitude * complex(math.cos(self.pad_phase), math.sin(self.pad_phase))
            field[:, self.d :] = pad
            win = torch.zeros((k, self.M, self.M), dtype=torch.complex64, device=h.device)
            o = self.offset
            win[:, o : o + self.G, o : o + self.G] = field.reshape(k, self.G, self.G)
            far = torch.fft.fft2(win, norm="ortho").reshape(k, -1)[:, self.pixels]
            return far.real ** 2 + far.imag ** 2
        if n == "H":
            sigma = float(knob) * math.sqrt(self.d)
            return math.sqrt(2.0 / self.D) * torch.cos((h @ self.W0.T) / sigma + self.b)
        raise ValueError(n)

    @torch.no_grad()
    def __call__(self, h: torch.Tensor, knob=None) -> torch.Tensor:
        h = h.to(self.device, torch.float32)
        out = torch.empty((h.shape[0], self.D), dtype=torch.float32, device=self.device)
        for s in range(0, h.shape[0], self.chunk):
            out[s : s + self.chunk] = self._apply(h[s : s + self.chunk], knob)
        return out


def standardize_fit(X: torch.Tensor):
    """Per-dimension mean / SD over all rows of a (rows, D) tensor; constant columns -> 0."""
    mean = X.mean(0)
    sd = X.std(0, unbiased=False)
    constant = sd <= 1e-6 * torch.clamp(mean.abs(), min=1.0)
    sd = torch.where(constant, torch.full_like(sd, float("inf")), sd)
    return mean, sd
