"""Simulated optical feature maps (protocol 3.4, map F)."""
from .maps import FFourier, FScatter, analytic_kernel_scatter, phase_encode

__all__ = ["FScatter", "FFourier", "phase_encode", "analytic_kernel_scatter"]
