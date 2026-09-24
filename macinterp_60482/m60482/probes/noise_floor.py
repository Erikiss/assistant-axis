"""How big is the exposure-boundary change compared with nothing happening?

This is the probe that should have been run before any mechanism was proposed.  Two of
the three checkpoint boundaries contain no exposure of the anchor.  Whatever the target
probability does across those is the noise floor, and the exposure boundary has to
clear it before there is an effect to explain.

Two nulls are used, and they answer different questions:

*anchor-own*
    How far does the anchor's own target probability move at the placebo boundaries?
    With two placebo boundaries this is an estimate from two numbers, which is
    reported as such.

*control-referenced*
    How far do the 40 control passages' target probabilities move at this same
    boundary?  This is the better null -- 40 samples rather than two -- but it assumes
    the controls are exchangeable with the anchor, which they are only approximately.
"""

from __future__ import annotations

import numpy as np

from .. import config as C
from .. import stats as S
from ..measure import Bundle
from ..registry import ProbeResult, register

RULE = (
    "Let d_exp be the anchor's change in log p(target) across the exposure boundary and "
    "d_plac its changes across the two placebo boundaries. Let D be the 40 controls' "
    "changes across the exposure boundary. SUPPORTED requires BOTH (a) |d_exp| > "
    "max|d_plac|, and (b) |d_exp| above the 95th percentile of |D|, i.e. a "
    "control-referenced p <= 0.05 -- which, with 40 controls, means being the largest "
    "or second largest of 41 values. REFUTED if |d_exp| <= max|d_plac|. Anything else "
    "is INCONCLUSIVE."
)


@register(
    "noise_floor",
    paper="(no paper -- null calibration; the precondition for every other probe)",
    hypothesis="The exposure boundary moved the target probability more than nothing does.",
    question="Does the exposure-boundary change exceed the checkpoint-to-checkpoint noise floor?",
    decision_rule=RULE,
    order=15,
)
def noise_floor(bundle: Bundle) -> ProbeResult:
    a_i = bundle.pi(C.ANCHOR)
    ctrl = [bundle.pi(n) for n in bundle.control_names]

    def dlogp(p_i: int, boundary: tuple[int, int]) -> float:
        lo, hi = boundary
        return float(
            bundle.target_logp[bundle.ci(hi), p_i] - bundle.target_logp[bundle.ci(lo), p_i]
        )

    boundaries = [b for b in C.BOUNDARIES if b[0] in bundle.checkpoints and b[1] in bundle.checkpoints]
    if C.EXPOSURE_BOUNDARY not in boundaries:
        return ProbeResult(
            probe="noise_floor",
            paper="(null calibration)",
            hypothesis="The exposure boundary moved the target probability more than nothing does.",
            question="Does the exposure-boundary change exceed the noise floor?",
            verdict="NOT_RUN",
            decision_rule=RULE,
            summary="The exposure boundary is not present in this bundle's checkpoints.",
            cannot_conclude="Nothing; the measurement was not made.",
        )

    d_exp = dlogp(a_i, C.EXPOSURE_BOUNDARY)
    placebo = {b: dlogp(a_i, b) for b in boundaries if b != C.EXPOSURE_BOUNDARY}
    max_plac = max(abs(v) for v in placebo.values()) if placebo else float("nan")

    ctrl_exp = [dlogp(i, C.EXPOSURE_BOUNDARY) for i in ctrl]
    cmp_ = S.compare_to_controls(
        "abs_dlogp_target_exposure", abs(d_exp), [abs(x) for x in ctrl_exp], direction="greater"
    )

    rows = []
    for b in boundaries:
        cd = [abs(dlogp(i, b)) for i in ctrl]
        rows.append(
            {
                "boundary": f"{b[0]} -> {b[1]}",
                "exposure": b == C.EXPOSURE_BOUNDARY,
                "anchor_dlogp": dlogp(a_i, b),
                "anchor_abs": abs(dlogp(a_i, b)),
                "control_median_abs": float(np.median(cd)),
                "control_p95_abs": float(np.quantile(cd, 0.95)),
                "control_max_abs": float(np.max(cd)),
                "anchor_percentile": float(np.mean(np.asarray(cd) < abs(dlogp(a_i, b))) * 100),
            }
        )

    beats_own = abs(d_exp) > max_plac
    beats_controls = cmp_.p_one_sided <= 0.05

    if beats_own and beats_controls:
        verdict = "SUPPORTED"
    elif not beats_own:
        verdict = "REFUTED"
    else:
        verdict = "INCONCLUSIVE"

    summary = (
        f"anchor dlogp(target) at the exposure boundary = {d_exp:+.5f}; its own placebo "
        f"boundaries give {', '.join(f'{v:+.5f}' for v in placebo.values())} "
        f"(max |.| = {max_plac:.5f}). Against the 40 controls at the same boundary the "
        f"anchor sits at the {cmp_.percentile:.0f}th percentile, p = {cmp_.p_one_sided:.4f} "
        f"(floor {cmp_.p_floor:.4f})."
    )

    return ProbeResult(
        probe="noise_floor",
        paper="(null calibration)",
        hypothesis="The exposure boundary moved the target probability more than nothing does.",
        question="Does the exposure-boundary change exceed the checkpoint-to-checkpoint noise floor?",
        verdict=verdict,
        decision_rule=RULE,
        summary=summary,
        evidence={
            "d_exposure": d_exp,
            "d_placebo": {f"{k[0]}->{k[1]}": v for k, v in placebo.items()},
            "max_abs_placebo": max_plac,
            "beats_own_placebo": bool(beats_own),
            "control_comparison": cmp_.as_dict(),
            "n_placebo_boundaries": len(placebo),
        },
        tables={"change in log p(target) per boundary": rows},
        cannot_conclude=(
            "It cannot establish that a change which clears this floor was *caused* by "
            "the exposure. The anchor's own null has two samples, so 'larger than both "
            "placebo boundaries' is roughly a coin flip under the null even when nothing "
            "happened. The control-referenced p has a floor of "
            f"{C.MIN_ATTAINABLE_P:.4f} and is uncorrected for the "
            f"{C.N_BOUNDARIES} boundaries and the number of probes in this suite."
        ),
        caveats=[
            S.selection_note(True),
            S.multiplicity_note(10),
            "Controls have their own targets, drawn from different contexts. Their "
            "log-probability changes are a null for 'how much does a target move "
            "between checkpoints', not for 'how much does THIS target move'.",
        ],
    )
