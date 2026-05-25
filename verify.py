"""Verify an ACI2 solution's score.

Reproduces the Einstein Arena platform verifier exactly: FFT autoconvolution,
Simpson's rule for the L2 integral (piecewise-linear interpolation with
zero-padded boundaries), and pointwise L1 / Linf.

    $ python verify.py solutions/best_2031546.npy
    N        = 2,031,546
    C        = 0.9627341576
"""

import argparse
import numpy as np


def score(f):
    f = np.asarray(f, dtype=np.float64)
    n = len(f)
    nc = 2 * n - 1
    nfft = 1
    while nfft < nc:
        nfft <<= 1
    F = np.fft.rfft(f, n=nfft)
    g = np.fft.irfft(F * F, n=nfft)[:nc]
    h = 1.0 / (nc + 1)
    y = np.concatenate(([0.0], g, [0.0]))
    l2sq = (h / 3.0) * np.sum(y[:-1] ** 2 + y[:-1] * y[1:] + y[1:] ** 2)
    l1 = g.sum() * h
    linf = g.max()
    return l2sq / (l1 * linf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Path to a .npy file holding f as a 1-D float64 array")
    args = parser.parse_args()
    f = np.load(args.path)
    print(f"N        = {len(f):,}")
    print(f"C        = {score(f):.10f}")
