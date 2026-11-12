"""E0 optical-map unit tests (protocol E0 table). Run: pytest tests/test_optics.py -v"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from optics.maps import FFourier, FScatter, analytic_kernel_scatter, phase_encode  # noqa: E402

RNG = np.random.default_rng(1234)


def _loop_scatter(U, phi):
    n, d = phi.shape
    D = U.shape[0]
    out = np.zeros((n, D))
    for a in range(n):
        for j in range(D):
            s = 0j
            for k in range(d):
                s += U[j, k] * complex(math.cos(phi[a, k]), math.sin(phi[a, k]))
            out[a, j] = abs(s) ** 2
    return out


def _loop_fourier(m: FFourier, phi):
    """Explicit 2-D DFT sum, no numpy FFT: F[u,v] = (1/M) sum_xy f[x,y] exp(-2 pi i (u x + v y) / M)."""
    n = phi.shape[0]
    M, G, o = m.M, m.G, m.offset
    out = np.zeros((n, m.D))
    for a in range(n):
        f = np.zeros((M, M), dtype=complex)
        for k in range(G * G):
            x, y = divmod(k, G)
            if k < m.d:
                val = complex(math.cos(phi[a, k]), math.sin(phi[a, k]))
            else:
                val = m.pad_amplitude * complex(math.cos(m.pad_phase), math.sin(m.pad_phase))
            f[o + x, o + y] = val
        for j, p in enumerate(m.pixels):
            u, v = divmod(int(p), M)
            s = 0j
            for x in range(M):
                for y in range(M):
                    if f[x, y] != 0:
                        ang = -2 * math.pi * (u * x + v * y) / M
                        s += f[x, y] * complex(math.cos(ang), math.sin(ang))
            out[a, j] = abs(s / M) ** 2
    return out


# ----------------------------------------------------------------- 1. vectorized vs brute force
def test_scatter_vectorized_matches_loop():
    m = FScatter(d=12, D=20, seed=3, complex_dtype=np.complex128)
    phi = phase_encode(RNG.standard_normal((5, 12)), math.pi)
    assert np.abs(m.intensity_from_phase(phi) - _loop_scatter(m.U, phi)).max() < 1e-6


@pytest.mark.parametrize("pad_amplitude", [1.0, 0.0])
def test_fourier_vectorized_matches_loop(pad_amplitude):
    m = FFourier(d=12, D=30, seed=3, fft_size=8, pad_amplitude=pad_amplitude, complex_dtype=np.complex128)
    phi = phase_encode(RNG.standard_normal((4, 12)), math.pi)
    assert np.abs(m.intensity_from_phase(phi) - _loop_fourier(m, phi)).max() < 1e-6


# ----------------------------------------------------------------- 2. energy conservation
@pytest.mark.parametrize(
    "pad_amplitude,fft_size,expected",
    [
        (0.0, 28, 768.0),   # protocol wording: full unitary FFT, D = d (dark padding) -> sum r = d
        (1.0, 28, 784.0),   # our default: 16 illuminated padding pixels -> sum r = G^2 = 784
        (1.0, 64, 784.0),   # oversampled 64x64 window, all pixels read: energy still 784
    ],
)
def test_fourier_energy_conservation(pad_amplitude, fft_size, expected):
    m = FFourier(d=768, D=fft_size * fft_size, seed=0, fft_size=fft_size,
                 pad_amplitude=pad_amplitude, complex_dtype=np.complex128)
    phi = phase_encode(RNG.standard_normal((3, 768)), 2 * math.pi)
    sums = m.intensity_from_phase(phi).sum(-1)
    np.testing.assert_allclose(sums, expected, rtol=1e-9)


# ----------------------------------------------------------------- 3. beta -> 0
@pytest.mark.parametrize("make", [
    lambda: FScatter(d=64, D=128, seed=0, complex_dtype=np.complex128),
    lambda: FFourier(d=64, D=128, seed=0, fft_size=16, complex_dtype=np.complex128),
])
def test_beta_zero_gives_constant_features(make):
    m = make()
    h = RNG.standard_normal((50, 64)) * 3
    r = m(h, beta=1e-10)
    spread = (r.max(0) - r.min(0)) / np.maximum(r.mean(0), 1e-12)
    assert spread.max() < 1e-8
    r0 = m(h, beta=0.0)
    assert np.all(r0 == r0[0])


# ----------------------------------------------------------------- 4. global phase offset
def test_scatter_global_phase_invariance():
    m = FScatter(d=32, D=64, seed=1, complex_dtype=np.complex128)
    phi = phase_encode(RNG.standard_normal((6, 32)), math.pi)
    np.testing.assert_allclose(m.intensity_from_phase(phi + 0.7), m.intensity_from_phase(phi), atol=1e-10)


def test_fourier_global_phase_invariance():
    """A global offset applies to every illuminated SLM pixel, padding included."""
    m = FFourier(d=30, D=100, seed=1, fft_size=12, complex_dtype=np.complex128)
    phi = phase_encode(RNG.standard_normal((6, 30)), math.pi)
    shifted = m.intensity_from_phase(phi + 0.7, pad_phase=m.pad_phase + 0.7)
    np.testing.assert_allclose(shifted, m.intensity_from_phase(phi), atol=1e-10)


def test_fourier_offset_on_data_pixels_only_changes_output():
    """Documented property: with illuminated padding, shifting only the data pixels is NOT a global
    offset (the padding acts as a phase reference), so the intensities change."""
    m = FFourier(d=30, D=100, seed=1, fft_size=12, complex_dtype=np.complex128)
    phi = phase_encode(RNG.standard_normal((6, 30)), math.pi)
    assert np.abs(m.intensity_from_phase(phi + 0.7) - m.intensity_from_phase(phi)).max() > 1e-3


# ----------------------------------------------------------------- 5. phase wrap
@pytest.mark.parametrize("make", [
    lambda: FScatter(d=32, D=64, seed=2, complex_dtype=np.complex128),
    lambda: FFourier(d=32, D=64, seed=2, fft_size=12, complex_dtype=np.complex128),
])
def test_phase_wrap_invariance(make):
    m = make()
    phi = phase_encode(RNG.standard_normal((6, 32)), math.pi)
    kwargs = {"pad_phase": m.pad_phase + 2 * math.pi} if isinstance(m, FFourier) else {}
    np.testing.assert_allclose(m.intensity_from_phase(phi + 2 * math.pi, **kwargs),
                               m.intensity_from_phase(phi), atol=1e-9)


# ----------------------------------------------------------------- 6. empirical vs analytic kernel
def test_empirical_kernel_matches_analytic():
    """Mean of r(h).r(h')/D over 10^4 random U vs 1 + |mean exp(i dphi)|^2, within 2%."""
    d, D, n_draws, chunk = 16, 64, 10_000, 500
    rng = np.random.default_rng(7)
    hs = rng.standard_normal((4, d))
    phis = phase_encode(hs, math.pi)
    pairs = [(0, 0), (0, 1), (1, 2), (2, 3)]
    fields = np.exp(1j * phis)  # (4, d)
    acc = np.zeros(len(pairs))
    for _ in range(n_draws // chunk):
        U = (rng.standard_normal((chunk, D, d)) + 1j * rng.standard_normal((chunk, D, d))) / math.sqrt(2 * d)
        r = np.abs(np.einsum("bjk,nk->bnj", U, fields)) ** 2  # (chunk, 4, D)
        for p, (a, b) in enumerate(pairs):
            acc[p] += (r[:, a] * r[:, b]).sum(-1).sum() / D
    emp = acc / n_draws
    ana = np.array([analytic_kernel_scatter(phis[a], phis[b]) for a, b in pairs])
    np.testing.assert_allclose(emp, ana, rtol=0.02)


# ----------------------------------------------------------------- extra: distribution and seeds
def test_scatter_matrix_variance():
    m = FScatter(d=256, D=512, seed=0, complex_dtype=np.complex128)
    assert abs(np.mean(np.abs(m.U) ** 2) * 256 - 1.0) < 0.02


def test_fourier_pixels_depend_on_seed_only():
    a = FFourier(d=768, D=3072, seed=0)
    b = FFourier(d=768, D=3072, seed=0)
    c = FFourier(d=768, D=3072, seed=1)
    assert np.array_equal(a.pixels, b.pixels) and not np.array_equal(a.pixels, c.pixels)
    assert len(np.unique(a.pixels)) == 3072
