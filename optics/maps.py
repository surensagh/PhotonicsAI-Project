"""Phase-encoded optical feature map F (protocol 3.4).

    phi = beta * (tanh(h) + 1) / 2           phase written on the SLM, in [0, beta]
    r   = | U exp(i phi) |^2                 intensity read by the camera

Two physical variants of U:

F-scatter  U_jk ~ CN(0, 1/d) i.i.d.  (diffuser / multimode fibre between SLM and camera).

F-fourier  The d phases drive the first d macro-pixels of a G x G SLM grid (G = ceil(sqrt(d)),
           row-major; 28 x 28 = 784 for d = 768). The remaining G^2 - d pixels are padding:
           illuminated with amplitude `pad_amplitude` (default 1, a real illuminated SLM pixel)
           and phase `pad_phase` (default 0). The G x G field sits in the centre of an M x M
           window (default M = 64) that is dark outside the SLM aperture. A lens performs a
           unitary 2-D FFT of the M x M window; the camera reads D intensities at fixed,
           randomly chosen far-field pixels. M > G means the camera samples the far field more
           finely than the SLM pitch, which is what makes D = 4d = 3072 > G^2 = 784 possible.
"""
from __future__ import annotations

import math

import numpy as np


def phase_encode(h: np.ndarray, beta: float, tau: float = 1.0) -> np.ndarray:
    """phi = beta * (tanh(h / tau) + 1) / 2.  tau = 1 is the v1 encoding; v2 uses beta = 2 pi, tau knob."""
    return beta * (np.tanh(h / tau) + 1.0) / 2.0


def analytic_kernel_scatter(phi1: np.ndarray, phi2: np.ndarray) -> np.ndarray:
    """(1/D) E_U[r(h)^T r(h')] = 1 + |mean_k exp(i (phi1_k - phi2_k))|^2 for U ~ CN(0, 1/d)."""
    c = np.exp(1j * (np.asarray(phi1) - np.asarray(phi2))).mean(-1)
    return 1.0 + np.abs(c) ** 2


def _rows(x: np.ndarray, d: int) -> tuple[np.ndarray, tuple]:
    if x.shape[-1] != d:
        raise ValueError(f"last dimension must be d={d}, got {x.shape[-1]}")
    lead = x.shape[:-1]
    return x.reshape(-1, d), lead


class FScatter:
    """F-scatter: i.i.d. complex Gaussian transmission matrix."""

    name = "F-scatter"

    def __init__(self, d: int, D: int, seed: int, complex_dtype=np.complex64, chunk: int = 4096):
        self.d, self.D, self.seed, self.chunk = d, D, seed, chunk
        rng = np.random.default_rng(seed)
        U = (rng.standard_normal((D, d)) + 1j * rng.standard_normal((D, d))) / math.sqrt(2 * d)
        self.U = U.astype(complex_dtype)
        self.UT = np.ascontiguousarray(self.U.T)

    def intensity_from_phase(self, phi: np.ndarray) -> np.ndarray:
        flat, lead = _rows(phi, self.d)
        out = np.empty((flat.shape[0], self.D), dtype=np.float64 if self.U.dtype == np.complex128 else np.float32)
        for s in range(0, flat.shape[0], self.chunk):
            field = np.exp(1j * flat[s : s + self.chunk]).astype(self.U.dtype)
            u = field @ self.UT
            out[s : s + self.chunk] = (u.real ** 2 + u.imag ** 2)
        return out.reshape(*lead, self.D)

    def __call__(self, h: np.ndarray, beta: float, tau: float = 1.0) -> np.ndarray:
        return self.intensity_from_phase(phase_encode(h, beta, tau))


class FFourier:
    """F-fourier: SLM macro-pixel grid -> lens (unitary 2-D FFT) -> D camera pixels."""

    name = "F-fourier"

    def __init__(
        self,
        d: int,
        D: int,
        seed: int,
        fft_size: int = 64,
        grid: int | None = None,
        pad_amplitude: float = 1.0,
        pad_phase: float = 0.0,
        complex_dtype=np.complex64,
        chunk: int = 1024,
    ):
        self.d, self.D, self.seed, self.chunk = d, D, seed, chunk
        self.G = grid or math.ceil(math.sqrt(d))
        self.M = fft_size
        if self.G * self.G < d:
            raise ValueError("grid too small for d")
        if self.M < self.G:
            raise ValueError("fft_size must be >= grid")
        if D > self.M * self.M:
            raise ValueError(f"D={D} exceeds the {self.M}x{self.M} far-field pixels")
        self.pad_amplitude, self.pad_phase = pad_amplitude, pad_phase
        self.complex_dtype = complex_dtype
        self.offset = (self.M - self.G) // 2
        rng = np.random.default_rng(seed)
        if D == self.M * self.M:
            self.pixels = np.arange(D)
        else:
            self.pixels = np.sort(rng.choice(self.M * self.M, size=D, replace=False))

    def slm_field(self, phi: np.ndarray, pad_phase: float | None = None) -> np.ndarray:
        """Complex field on the G x G SLM for a batch of phase vectors, shape (n, G, G)."""
        pp = self.pad_phase if pad_phase is None else pad_phase
        n = phi.shape[0]
        g2 = self.G * self.G
        full = np.empty((n, g2), dtype=self.complex_dtype)
        full[:, : self.d] = np.exp(1j * phi)
        full[:, self.d :] = self.pad_amplitude * np.exp(1j * pp)
        return full.reshape(n, self.G, self.G)

    def intensity_from_phase(self, phi: np.ndarray, pad_phase: float | None = None) -> np.ndarray:
        flat, lead = _rows(phi, self.d)
        real_dtype = np.float64 if self.complex_dtype == np.complex128 else np.float32
        out = np.empty((flat.shape[0], self.D), dtype=real_dtype)
        o, G, M = self.offset, self.G, self.M
        for s in range(0, flat.shape[0], self.chunk):
            f = self.slm_field(flat[s : s + self.chunk], pad_phase)
            window = np.zeros((f.shape[0], M, M), dtype=self.complex_dtype)
            window[:, o : o + G, o : o + G] = f
            far = np.fft.fft2(window, norm="ortho").reshape(f.shape[0], M * M)
            far = far[:, self.pixels]
            out[s : s + self.chunk] = far.real ** 2 + far.imag ** 2
        return out.reshape(*lead, self.D)

    def __call__(self, h: np.ndarray, beta: float, tau: float = 1.0) -> np.ndarray:
        return self.intensity_from_phase(phase_encode(h, beta, tau))
