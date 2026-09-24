"""Does the context-NLL drop at the exposure boundary have any positional structure?

A single exposure to a passage should, if it leaves a trace at all, leave it somewhere
in particular: on the passage's distinctive content -- its proper nouns, its unusual
phrasings -- and with some spatial coherence, because neighbouring tokens share context.

The earlier token-level analysis found none of that.  The drop sat on tokens with high
baseline NLL, on punctuation and on word continuations, while proper nouns were
untouched, and neighbouring positions did not agree with each other.  That pattern is
what regression-to-the-mean looks like: whatever was hardest at checkpoint A improves
most by checkpoint B, everywhere, in every passage.

This probe re-derives that finding from the bundle so it is part of the record rather
than a remembered result, and it tests it the way it should be tested: against the
controls, which have no exposure and should show the same pattern if it is generic.
"""

from __future__ import annotations

import numpy as np

from .. import config as C
from .. import stats as S
from ..measure import Bundle
from ..registry import ProbeResult, register

RULE = (
    "Across the exposure boundary compute, per position, the change in context NLL. "
    "Test three structures: (a) correlation with baseline NLL at the earlier "
    "checkpoint, (b) lag-1 spatial autocorrelation of the change, (c) mean change on "
    "the name span versus the rest. The drop has EXPOSURE-SPECIFIC structure if any of "
    "these differs from the 40 controls' distribution at the same boundary with "
    "control-referenced p <= 0.05. SUPPORTED if so. REFUTED if all three sit inside the "
    "control range AND the baseline-NLL correlation is strongly negative "
    "(r <= -0.3), which is the regression-to-the-mean signature."
)


def _lag1(x: np.ndarray) -> float:
    if x.size < 3:
        return float("nan")
    a, b = x[:-1], x[1:]
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float((a * b).sum() / den) if den > 0 else float("nan")


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float((a * b).sum() / den) if den > 0 else float("nan")


@register(
    "ctx_nll_structure",
    paper="(no paper -- re-derivation of the token-level finding, now control-referenced)",
    hypothesis="The exposure left a trace localised on the passage's distinctive content.",
    question="Does the NLL improvement at the exposure boundary sit anywhere in particular?",
    decision_rule=RULE,
    order=35,
)
def ctx_nll_structure(bundle: Bundle) -> ProbeResult:
    lo, hi = C.EXPOSURE_BOUNDARY
    if lo not in bundle.checkpoints or hi not in bundle.checkpoints:
        return ProbeResult(
            probe="ctx_nll_structure", paper="(re-derivation)",
            hypothesis="The exposure left a localised trace.",
            question="Does the NLL improvement sit anywhere in particular?",
            verdict="NOT_RUN", decision_rule=RULE,
            summary="Exposure boundary not present.",
            cannot_conclude="Nothing; the measurement was not made.",
        )

    c_lo, c_hi = bundle.ci(lo), bundle.ci(hi)

    def stats_for(p_i: int) -> dict:
        base = bundle.ctx_nll[c_lo, p_i]
        delta = bundle.ctx_nll[c_hi, p_i] - base
        # ctx_nll[i] is the NLL of token i+1, so the name at position j is at index j-1
        name_idx = [j - 1 for j in C.NAME_POSITIONS if 0 < j <= delta.size]
        rest = np.setdiff1d(np.arange(delta.size), name_idx)
        return {
            "mean_delta": float(delta.mean()),
            "corr_with_baseline": _corr(delta, base),
            "lag1_autocorr": _lag1(delta),
            "name_span_delta": float(delta[name_idx].mean()) if name_idx else float("nan"),
            "rest_delta": float(delta[rest].mean()) if rest.size else float("nan"),
        }

    a = stats_for(bundle.pi(C.ANCHOR))
    ctrl = [stats_for(bundle.pi(n)) for n in bundle.control_names]

    comparisons = {}
    for key, direction in (
        ("corr_with_baseline", "less"),
        ("lag1_autocorr", "greater"),
        ("mean_delta", "less"),
    ):
        vals = [c[key] for c in ctrl if np.isfinite(c[key])]
        if vals and np.isfinite(a[key]):
            comparisons[key] = S.compare_to_controls(key, a[key], vals, direction=direction)

    # name-span specificity: is the name improving more than the rest, unusually so?
    a_gap = a["name_span_delta"] - a["rest_delta"]
    ctrl_gap = [c["name_span_delta"] - c["rest_delta"] for c in ctrl
                if np.isfinite(c["name_span_delta"]) and np.isfinite(c["rest_delta"])]
    if ctrl_gap and np.isfinite(a_gap):
        comparisons["name_span_minus_rest"] = S.compare_to_controls(
            "name_span_minus_rest", a_gap, ctrl_gap, direction="less"
        )

    any_outside = any(c.p_one_sided <= 0.05 for c in comparisons.values())
    rtm = np.isfinite(a["corr_with_baseline"]) and a["corr_with_baseline"] <= -0.3

    rows = [
        {
            "statistic": k,
            "anchor": c.anchor,
            "control_median": c.control_median,
            "control_min": c.control_min,
            "control_max": c.control_max,
            "percentile": c.percentile,
            f"p_{c.direction}": c.p_one_sided,
        }
        for k, c in comparisons.items()
    ]

    if any_outside:
        verdict = "SUPPORTED"
        worst = min(comparisons.values(), key=lambda c: c.p_one_sided)
        summary = (
            f"{worst.statistic} = {worst.anchor:.4f} for the anchor against a control "
            f"median of {worst.control_median:.4f} (p_{worst.direction} = "
            f"{worst.p_one_sided:.4f}); the improvement is not distributed the way an "
            "unexposed passage's is."
        )
    elif rtm:
        verdict = "REFUTED"
        summary = (
            f"The per-position improvement correlates {a['corr_with_baseline']:+.3f} "
            "with the baseline NLL -- whatever was hardest improved most -- with "
            f"lag-1 spatial autocorrelation {a['lag1_autocorr']:+.3f} and no name-span "
            f"specificity (name {a['name_span_delta']:+.4f} vs rest "
            f"{a['rest_delta']:+.4f}). Every statistic sits inside the 40 controls' "
            "range. This is regression to the mean, not a localised trace."
        )
    else:
        verdict = "INCONCLUSIVE"
        summary = (
            "No statistic falls outside the control range, but the "
            f"baseline-NLL correlation ({a['corr_with_baseline']:+.3f}) is not the "
            "regression-to-the-mean signature either."
        )

    return ProbeResult(
        probe="ctx_nll_structure",
        paper="(re-derivation of the token-level finding)",
        hypothesis="The exposure left a trace localised on the passage's distinctive content.",
        question="Does the NLL improvement at the exposure boundary sit anywhere in particular?",
        verdict=verdict,
        decision_rule=RULE,
        summary=summary,
        evidence={
            "boundary": f"{lo} -> {hi}",
            "anchor": a,
            "comparisons": {k: c.as_dict() for k, c in comparisons.items()},
            "regression_to_mean_signature": bool(rtm),
        },
        tables={"anchor vs controls, exposure boundary": rows},
        cannot_conclude=(
            "Absence of positional structure does not prove absence of a trace: a trace "
            "spread evenly over all 207 positions would look exactly like this. What it "
            "does rule out is the specific story the memorization reading told -- that "
            "the model learned this passage's distinctive content."
        ),
        caveats=[
            "The three structure tests are correlated with one another; four "
            "comparisons at a floor of "
            f"{C.MIN_ATTAINABLE_P:.4f} cannot survive correction.",
            S.multiplicity_note(4),
        ],
    )
