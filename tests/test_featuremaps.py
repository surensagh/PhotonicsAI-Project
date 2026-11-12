"""Torch feature maps: agreement with the tested numpy optics, definitions and seeding."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from optics.maps import FFourier, FScatter  # noqa: E402
from optlm.featuremaps import FeatureMap  # noqa: E402

d, D = 64, 256
H = torch.from_numpy(np.random.default_rng(0).standard_normal((40, d)).astype(np.float32))


def test_fscatter_matches_numpy():
    t = FeatureMap("F-scatter", d, D, seed=2)
    n = FScatter(d, D, seed=[6, 2], complex_dtype=np.complex128)
    np.testing.assert_allclose(t(H, math.pi).numpy(), n(H.numpy().astype(np.float64), math.pi), rtol=1e-4, atol=1e-4)


def test_ffourier_matches_numpy():
    t = FeatureMap("F-fourier", d, D, seed=2)
    n = FFourier(d, D, seed=[7, 2], complex_dtype=np.complex128)
    np.testing.assert_allclose(t(H, math.pi / 2).numpy(), n(H.numpy().astype(np.float64), math.pi / 2),
                               rtol=1e-4, atol=1e-5)


def test_definitions_relations():
    B = FeatureMap("B", d, D, 0)(H)
    C = FeatureMap("C", d, D, 0)(H)
    Dm = FeatureMap("D", d, D, 0)(H, 0.001)
    torch.testing.assert_close(C, B ** 2)                  # C = (Rh)^2, same R as B
    torch.testing.assert_close(Dm / 0.001, B, rtol=1e-3, atol=1e-4)  # tanh(sx) ~ sx for small s
    assert torch.equal(FeatureMap("A", d, d, 0)(H), H)


def test_rff_range_and_kernel():
    r = FeatureMap("H", d, 4096, 0)(H, 1.0)
    assert r.abs().max() <= math.sqrt(2 / 4096) + 1e-6
    # RFF approximates the Gaussian kernel exp(-||h-h'||^2 / (2 sigma^2)), sigma = sqrt(d)
    approx = (r[0] * r[1]).sum().item()
    exact = math.exp(-((H[0] - H[1]) ** 2).sum().item() / (2 * d))
    assert abs(approx - exact) < 0.05


def test_seeds_reproducible_and_distinct():
    a = FeatureMap("F-scatter", d, d, 3)(H, math.pi)
    b = FeatureMap("F-scatter", d, d, 3)(H, math.pi)
    c = FeatureMap("F-scatter", d, d, 4)(H, math.pi)
    assert torch.equal(a, b) and not torch.allclose(a, c)


@pytest.mark.parametrize("name,knob", [("B", None), ("C", None), ("D", 1.0), ("E", None),
                                       ("F-scatter", math.pi), ("F-fourier", math.pi), ("H", 2.0)])
def test_shapes_and_chunking(name, knob):
    m = FeatureMap(name, d, D, 1, chunk=7)
    big_chunk = FeatureMap(name, d, D, 1, chunk=4096)
    r = m(H, knob)
    assert r.shape == (40, D) and torch.isfinite(r).all()
    torch.testing.assert_close(r, big_chunk(H, knob))
