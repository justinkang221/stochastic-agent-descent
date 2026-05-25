# Stochastic Agent Descent — ACI2 Solutions

Companion artefact for the paper

> **Stochastic Agent Descent: Adaptive Agents for the Future of Non-Convex Optimization.**
> Justin Singh Kang. *AI Agents for Discovery in the Wild Workshop, ACM CAIS 2026.*

This repo holds the two SOTA solutions for the second autocorrelation
inequality (ACI2) discovered by SAD and quoted in the paper.

## The problem

ACI2 asks for the supremum, over non-negative compactly-supported $f$, of

$$
C(f) \;=\; \frac{\|f \star f\|_2^2}{\|f \star f\|_1 \, \|f \star f\|_\infty}.
$$

Higher is better; the ratio is scale-invariant in $f$.

## Solutions

| File | $N$ | $C$ |
| --- | ---: | ---: |
| `solutions/best_2031546.npy` | 2,031,546 | 0.9627341576 |
| `solutions/best_400000.npy`  |   400,000 | 0.9626486384 |

Both are stored as 1-D `float64` NumPy arrays. The $N{=}400{,}000$
solution exceeds the prior published lower bound at roughly
$5\times$ smaller support than our best.

## Scoring

The Einstein Arena verifier uses Simpson's rule for the $L^2$ integral
(piecewise-linear $g$ with zero-padded boundaries), trapezoidal $L^1$,
and pointwise $L^\infty$. A naive $\sum g_i^2$ disagrees in the 6th
decimal:

```python
import numpy as np
f = np.load("solutions/best_2031546.npy").astype(np.float64)
g = np.convolve(f, f)
nc = len(g)
h = 1.0 / (nc + 1)
y = np.concatenate(([0.0], g, [0.0]))
l2sq = (h / 3.0) * np.sum(y[:-1]**2 + y[:-1]*y[1:] + y[1:]**2)
l1   = g.sum() * h
linf = g.max()
print(l2sq / (l1 * linf))   # 0.9627341576...
```

This matches the Einstein Arena platform verifier exactly.

## Related repos

The full optimisation codebase (worker recipes, Dinkelbach +
$\beta$-anneal inner loop, batched L-BFGS, etc.) lives at
[`justinkang221/second-autocorrelation-inequality`](https://github.com/justinkang221/second-autocorrelation-inequality).
