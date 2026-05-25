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

```python
import numpy as np
f = np.load("solutions/best_2031546.npy").astype(np.float64)
g = np.convolve(f, f)
C = (g**2).sum() / (g.sum() * g.max())
print(C)   # 0.9627341576...
```

The numbers above match the Einstein Arena platform verifier.

## Related repos

The full optimisation codebase (worker recipes, Dinkelbach +
$\beta$-anneal inner loop, batched L-BFGS, etc.) lives at
[`justinkang221/second-autocorrelation-inequality`](https://github.com/justinkang221/second-autocorrelation-inequality).
