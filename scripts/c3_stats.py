"""Statistics fixed by experiments/C3_cycle/PREREG.md, shared by the cycle-3 scripts.

* paired bootstrap: 2,000 resamples, seed 7, 95% percentile interval;
* paired sign-flip permutation test on the per-statement differences: exact when
  at most 20 statements change, otherwise 100,000 random flips with seed 7;
* the number of statements that go up and down, always reported with the delta.
"""

from __future__ import annotations

import itertools

import numpy as np


def bootstrap_ci(diff: np.ndarray, resamples: int = 2000, seed: int = 7) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(diff)
    means = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(resamples)])
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def sign_flip_p(diff: np.ndarray, flips: int = 100_000, seed: int = 7) -> tuple[float, str]:
    """Two-sided p for the mean difference; ``(p, "exact" | "sampled")``."""
    changed = diff[diff != 0]
    m = len(changed)
    if m == 0:
        return 1.0, "exact"
    observed = abs(changed.sum())
    if m <= 20:
        hits = sum(abs(sum(s * x for s, x in zip(signs, changed))) >= observed - 1e-12
                   for signs in itertools.product((1, -1), repeat=m))
        return hits / 2 ** m, "exact"
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(flips, m))
    stats = np.abs(signs @ changed)
    return float(((stats >= observed - 1e-12).sum() + 1) / (flips + 1)), "sampled"


def compare(control: np.ndarray, treatment: np.ndarray) -> dict:
    diff = np.asarray(treatment, dtype=float) - np.asarray(control, dtype=float)
    low, high = bootstrap_ci(diff)
    p, kind = sign_flip_p(diff)
    return {"control": round(float(np.mean(control)), 5), "treatment": round(float(np.mean(treatment)), 5),
            "delta": round(float(diff.mean()), 5), "ci95": [round(low, 5), round(high, 5)],
            "sign_flip_p": round(p, 5), "sign_flip": kind, "up": int((diff > 0).sum()),
            "down": int((diff < 0).sum()), "n": int(len(diff))}
