"""Fitted insertion loss of the T191134 fused SMF 3 dB coupler.

This file is self-contained. It depends only on NumPy.

The coefficients are the least-squares fit to time-gated Luna OVA data
(Kaiser gate 0.143 ns, 1300–1320 nm), referenced to 1310 nm. Bar paths are
white-white and blue-red. Cross paths are white-red and blue-white.
"""

from __future__ import annotations

import numpy as np


def coupler_insertion_loss_db(wavelength_nm: float | np.ndarray) -> list[float] | list[np.ndarray]:
    """Insertion loss of the fitted fused SMF 2×2 coupler.

    ``wavelength_nm`` is one wavelength or an array of wavelengths, in nm.
    The return value is always a list of four insertion losses, in dB, in
    this order: white-white, white-red, blue-white, blue-red. A single
    wavelength returns four floats; an array returns four arrays of the same
    shape. Insertion loss is the positive-loss convention, ``IL = -10 log10(T)``.
    """
    wavelength = np.asarray(wavelength_nm, dtype=float)
    scalar = wavelength.ndim == 0
    delta_nm = wavelength - 1310.0

    # Effective input throughput at 1310 nm, and its natural-log slope per nm.
    g_white = 0.9744008942504716 * np.exp(0.0005427807875566681 * delta_nm)
    g_blue = 0.9887745271735727 * np.exp(6.091144868185035e-05 * delta_nm)
    # Output factor and its slope. The red output is the reciprocal gauge.
    o_white = 0.9875815343377361 * np.exp(7.6130021562314284e-06 * delta_nm)
    o_red = 1.0 / o_white
    # Supermode beat phase at 1310 nm, and its slope, both in radians.
    theta = 0.7985731814296588 + 0.0015318850091198548 * delta_nm
    bar = np.cos(theta) ** 2
    cross = np.sin(theta) ** 2

    power = (
        g_white * o_white * bar,
        g_white * o_red * cross,
        g_blue * o_white * cross,
        g_blue * o_red * bar,
    )
    losses = [-10.0 * np.log10(np.clip(np.asarray(value, dtype=float), 1e-30, None)) for value in power]
    if scalar:
        return [float(value) for value in losses]
    return losses
