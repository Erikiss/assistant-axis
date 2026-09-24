"""The statistics the study is allowed to use, and the limits written into them.

Three facts constrain everything here:

1. There are 40 controls.  A rank-based p-value against them has a floor of
   ``1/41 = 0.0244``.  Nothing in this study can produce p < 0.024.
2. There are three boundaries, one of which is the exposure boundary.  Choosing the
   exposure boundary *after* looking is a 1-in-3 selection; no correction is applied
   anywhere in this module, so every reported p is uncorrected and must be read that
   way.
3. The anchor was selected for study because it looked interesting.  Statistics that
   condition on the same property that selected it are circular; :func:`selection_note`
   exists so probes have to say which side of that line they are on.

The functions return plain dataclasses rather than printing, so a probe's decision rule
can be applied mechanically.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Sequence

import numpy as np

from . import config as C


@dataclass(frozen=True)
class ControlComparison:
    """One statistic for the anchor, referenced against the same statistic for the
    controls."""

    statistic: str
    anchor: float
    control_median: float
    control_mean: float
    control_sd: float
    control_min: float
    control_max: float
    n_controls: int
    k_at_or_above: int          # controls whose value >= anchor's
    p_one_sided: float          # (k + 1) / (n + 1)
    z_vs_controls: float        # (anchor - mean) / sd, NaN if sd == 0
    percentile: float           # anchor's percentile within the control distribution
    p_floor: float              # the smallest p this comparison could have produced
    direction: str              # "greater" or "less"

    def as_dict(self) -> dict:
        return asdict(self)


def compare_to_controls(
    statistic: str,
    anchor: float,
    controls: Sequence[float],
    *,
    direction: str = "greater",
) -> ControlComparison:
    """Rank the anchor inside the control distribution.

    ``direction="greater"`` asks "is the anchor unusually large", ``"less"`` asks
    "unusually small".  The p-value is the standard permutation-style
    ``(k + 1) / (n + 1)``, which is conservative and never zero.
    """
    arr = np.asarray([c for c in controls if np.isfinite(c)], dtype=float)
    n = int(arr.size)
    if n == 0:
        raise ValueError(f"{statistic}: no finite control values")
    if direction not in ("greater", "less"):
        raise ValueError("direction must be 'greater' or 'less'")

    if direction == "greater":
        k = int(np.sum(arr >= anchor))
    else:
        k = int(np.sum(arr <= anchor))

    sd = float(arr.std(ddof=1)) if n > 1 else 0.0
    z = float((anchor - arr.mean()) / sd) if sd > 0 else float("nan")
    pct = float((np.sum(arr < anchor) + 0.5 * np.sum(arr == anchor)) / n * 100.0)

    return ControlComparison(
        statistic=statistic,
        anchor=float(anchor),
        control_median=float(np.median(arr)),
        control_mean=float(arr.mean()),
        control_sd=sd,
        control_min=float(arr.min()),
        control_max=float(arr.max()),
        n_controls=n,
        k_at_or_above=k,
        p_one_sided=float((k + 1) / (n + 1)),
        z_vs_controls=z,
        percentile=pct,
        p_floor=float(1.0 / (n + 1)),
        direction=direction,
    )


@dataclass(frozen=True)
class BoundaryContrast:
    """The exposure boundary against the placebo boundaries, for one statistic.

    With two placebo boundaries there is no meaningful p-value; what this reports is
    whether the exposure boundary is outside the placebo range at all.  It is a
    *screen*, not a test, and :attr:`n_placebo` is carried along so a reader cannot
    forget how weak it is.
    """

    statistic: str
    exposure: float
    placebo: tuple[float, ...]
    placebo_min: float
    placebo_max: float
    outside_placebo_range: bool
    ratio_to_largest_placebo: float
    n_placebo: int

    def as_dict(self) -> dict:
        d = asdict(self)
        d["placebo"] = list(self.placebo)
        return d


def contrast_boundaries(
    statistic: str,
    per_boundary: dict[tuple[int, int], float],
    *,
    exposure: tuple[int, int] = C.EXPOSURE_BOUNDARY,
) -> BoundaryContrast:
    exp = float(per_boundary[exposure])
    plac = tuple(
        float(v) for b, v in per_boundary.items() if b != exposure and np.isfinite(v)
    )
    if not plac:
        raise ValueError(f"{statistic}: no placebo boundaries available")
    lo, hi = min(plac), max(plac)
    largest = max(abs(p) for p in plac)
    return BoundaryContrast(
        statistic=statistic,
        exposure=exp,
        placebo=plac,
        placebo_min=lo,
        placebo_max=hi,
        outside_placebo_range=bool(exp > hi or exp < lo),
        ratio_to_largest_placebo=float(exp / largest) if largest > 0 else float("nan"),
        n_placebo=len(plac),
    )


def total_variation(p: np.ndarray, q: np.ndarray, axis: int = -1) -> np.ndarray:
    """Total-variation distance between two distributions, ``0.5 * sum|p - q|``."""
    return 0.5 * np.abs(np.asarray(p) - np.asarray(q)).sum(axis=axis)


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12, axis: int = -1) -> np.ndarray:
    """Jensen-Shannon divergence in nats.  Bounded, symmetric, and unlike TV it is
    sensitive to the tail -- useful as a second opinion when TV says nothing."""
    p = np.clip(np.asarray(p, dtype=float), eps, None)
    q = np.clip(np.asarray(q, dtype=float), eps, None)
    p = p / p.sum(axis=axis, keepdims=True)
    q = q / q.sum(axis=axis, keepdims=True)
    m = 0.5 * (p + q)
    kl = lambda a, b: np.sum(a * (np.log(a) - np.log(b)), axis=axis)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def bootstrap_ci(
    values: Sequence[float],
    *,
    statistic=np.median,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for a control-distribution statistic."""
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boots = statistic(arr[idx], axis=1)
    return (
        float(np.quantile(boots, alpha / 2)),
        float(np.quantile(boots, 1 - alpha / 2)),
    )


def noise_floor(per_boundary: dict[tuple[int, int], Sequence[float]]) -> dict:
    """Summarise the placebo boundaries as the noise floor for a statistic.

    This is the number every other probe is judged against: how much does this
    quantity move between two checkpoints when *nothing happened*?
    """
    placebo = [
        np.asarray(v, dtype=float)
        for b, v in per_boundary.items()
        if b in C.PLACEBO_BOUNDARIES
    ]
    if not placebo:
        raise ValueError("no placebo boundaries present")
    pooled = np.concatenate(placebo)
    pooled = pooled[np.isfinite(pooled)]
    return {
        "n": int(pooled.size),
        "mean_abs": float(np.abs(pooled).mean()),
        "median_abs": float(np.median(np.abs(pooled))),
        "sd": float(pooled.std(ddof=1)) if pooled.size > 1 else 0.0,
        "q95_abs": float(np.quantile(np.abs(pooled), 0.95)),
        "max_abs": float(np.abs(pooled).max()),
    }


def selection_note(conditioned_on_selection: bool) -> str:
    """Every probe declares whether its statistic is the one that selected the anchor."""
    if conditioned_on_selection:
        return (
            "CIRCULAR: this statistic is (a function of) the quantity that made state "
            "60482 interesting in the first place. It cannot provide independent "
            "evidence; read it as a description only."
        )
    return (
        "Independent of the selection criterion: this statistic was not used to pick "
        "the anchor."
    )


def multiplicity_note(n_tests: int) -> str:
    return (
        f"{n_tests} comparisons were made; no multiplicity correction is applied. "
        f"The smallest attainable p against {C.N_CONTROLS} controls is "
        f"{C.MIN_ATTAINABLE_P:.4f}, so a Bonferroni-corrected significant result is "
        f"{'unreachable' if C.MIN_ATTAINABLE_P * n_tests > 0.05 else 'reachable'} "
        f"at alpha=0.05."
    )
