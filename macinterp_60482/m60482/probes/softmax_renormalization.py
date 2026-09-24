"""Is the probability change a change in the target, or the softmax renormalising?

Probabilities are not independent.  A softmax couples every token to every other, so if
the dominant token loses mass, every other token gains -- without anything having
happened to them.  At the anchor, the dominant token is `" ph"` at p = 0.64-0.80, and
across the exposure boundary it *falls* (0.798 -> 0.744).  The target's rise from 0.0909
to 0.1046 is measured against that fall.

The clean instrument is a pairwise logit contrast.  Because
``log p_a - log p_b == logit_a - logit_b`` exactly -- the normaliser cancels -- the
contrast between the target and a reference token is computable from the stored
probabilities and is invariant to anything that shifts the whole distribution.

Schaeffer et al. (arXiv:2304.15004) make the general version of this point about
discontinuous metrics: a rank, or a probability read against a moving normaliser, can
manufacture a sharp-looking change out of a smooth underlying one.
"""

from __future__ import annotations

import numpy as np

from .. import config as C
from .. import stats as S
from ..measure import Bundle
from ..registry import ProbeResult, register

#: The exposure contrast must exceed the largest placebo contrast by this factor.
MARGIN = 1.5

RULE = (
    "For each passage take its modal top-1 token as the reference, and compute the "
    "contrast c = log p(target) - log p(reference), which equals the logit difference "
    "exactly because the normaliser cancels. Let d_exp and d_plac be the changes in c "
    "across the exposure and placebo boundaries. The probability-space effect reflects "
    f"a real change in the target iff |d_exp| >= {MARGIN} * max|d_plac|. REFUTED "
    "(the effect is largely renormalisation) if that fails. SUPPORTED if it holds AND "
    "the anchor's |d_exp| is also above the 95th percentile of the controls' at the "
    "same boundary."
)


@register(
    "softmax_renormalization",
    paper="Schaeffer et al. 2023, Are Emergent Abilities a Mirage? (arXiv:2304.15004) -- NOT on the candidate list",
    hypothesis="The probability rise is a change in the target, not the softmax renormalising around a falling competitor.",
    question="Does the effect survive being measured as a logit contrast?",
    decision_rule=RULE,
    order=18,
)
def softmax_renormalization(bundle: Bundle) -> ProbeResult:
    lo, hi = C.EXPOSURE_BOUNDARY
    if lo not in bundle.checkpoints or hi not in bundle.checkpoints:
        return ProbeResult(
            probe="softmax_renormalization", paper="arXiv:2304.15004",
            hypothesis="The probability rise is a change in the target.",
            question="Does the effect survive being measured as a logit contrast?",
            verdict="NOT_RUN", decision_rule=RULE,
            summary="Exposure boundary not present.",
            cannot_conclude="Nothing; the measurement was not made.",
        )

    vocab = bundle.meta.get("vocab", {})
    boundaries = [
        b for b in C.BOUNDARIES
        if b[0] in bundle.checkpoints and b[1] in bundle.checkpoints
    ]

    def reference_token(p_i: int) -> int:
        """The passage's modal top-1 across checkpoints -- a stable denominator."""
        ids = [int(bundle.top1_ids[bundle.ci(s), p_i]) for s in bundle.checkpoints]
        return max(set(ids), key=ids.count)

    def prob_of(c_i: int, p_i: int, tid: int) -> float:
        where = np.flatnonzero(bundle.topk_ids[c_i, p_i] == tid)
        return float(bundle.topk_probs[c_i, p_i][where[0]]) if where.size else float("nan")

    def contrast(p_i: int, step: int, ref: int) -> float:
        c_i = bundle.ci(step)
        pt = float(bundle.target_p[c_i, p_i])
        pr = prob_of(c_i, p_i, ref)
        if not (pt > 0 and pr > 0):
            return float("nan")
        return float(np.log(pt) - np.log(pr))

    def deltas(p_i: int) -> dict:
        ref = reference_token(p_i)
        return {b: contrast(p_i, b[1], ref) - contrast(p_i, b[0], ref) for b in boundaries}

    a_i = bundle.pi(C.ANCHOR)
    a_ref = reference_token(a_i)
    a_d = deltas(a_i)
    d_exp = a_d[C.EXPOSURE_BOUNDARY]
    d_plac = [v for b, v in a_d.items() if b != C.EXPOSURE_BOUNDARY and np.isfinite(v)]
    max_plac = max((abs(v) for v in d_plac), default=float("nan"))

    # the same quantity in probability space, for the side-by-side
    p_d = {
        b: float(
            np.log(max(bundle.target_p[bundle.ci(b[1]), a_i], 1e-12))
            - np.log(max(bundle.target_p[bundle.ci(b[0]), a_i], 1e-12))
        )
        for b in boundaries
    }

    rows = []
    for b in boundaries:
        rows.append(
            {
                "boundary": f"{b[0]} -> {b[1]}",
                "exposure": b == C.EXPOSURE_BOUNDARY,
                "d_log_p_target": p_d[b],
                "d_logit_contrast": a_d[b],
                "reference": vocab.get(str(a_ref), str(a_ref)),
            }
        )

    ctrl_d = []
    for n in bundle.control_names:
        v = deltas(bundle.pi(n)).get(C.EXPOSURE_BOUNDARY, float("nan"))
        if np.isfinite(v):
            ctrl_d.append(abs(v))
    cmp_ = (
        S.compare_to_controls("abs_d_contrast", abs(d_exp), ctrl_d, direction="greater")
        if ctrl_d else None
    )

    ratio = abs(d_exp) / max_plac if max_plac and np.isfinite(max_plac) and max_plac > 0 else float("nan")
    p_ratio = (
        abs(p_d[C.EXPOSURE_BOUNDARY])
        / max((abs(v) for b, v in p_d.items() if b != C.EXPOSURE_BOUNDARY), default=float("nan"))
    )

    clears_margin = np.isfinite(ratio) and ratio >= MARGIN
    clears_controls = cmp_ is not None and cmp_.p_one_sided <= 0.05

    if clears_margin and clears_controls:
        verdict = "SUPPORTED"
        summary = (
            f"The contrast against {vocab.get(str(a_ref), a_ref)!r} moves {d_exp:+.5f} "
            f"at the exposure boundary, {ratio:.1f}x the largest placebo, and the "
            f"anchor is at the {cmp_.percentile:.0f}th control percentile "
            f"(p = {cmp_.p_one_sided:.4f}). The effect is not renormalisation."
        )
    elif not clears_margin:
        verdict = "REFUTED"
        summary = (
            f"In probability space the exposure boundary looks "
            f"{p_ratio:.1f}x the largest placebo boundary. Measured as a logit contrast "
            f"against {vocab.get(str(a_ref), a_ref)!r} -- which cancels the normaliser -- "
            f"it is only {ratio:.1f}x ({d_exp:+.5f} against a largest placebo of "
            f"{max_plac:.5f}). Most of the apparent effect is the softmax "
            "redistributing mass as the dominant token falls, not a change in the "
            "target."
        )
    else:
        verdict = "INCONCLUSIVE"
        summary = (
            f"The contrast clears the placebo margin ({ratio:.1f}x) but the anchor is "
            f"not unusual among the controls "
            f"(p = {cmp_.p_one_sided:.4f} at the {cmp_.percentile:.0f}th percentile)."
            if cmp_ else
            f"The contrast clears the placebo margin ({ratio:.1f}x) but no control "
            "comparison was available."
        )

    return ProbeResult(
        probe="softmax_renormalization",
        paper="Schaeffer et al. 2023 (arXiv:2304.15004)",
        hypothesis="The probability rise is a change in the target, not the softmax renormalising around a falling competitor.",
        question="Does the effect survive being measured as a logit contrast?",
        verdict=verdict,
        decision_rule=RULE,
        summary=summary,
        evidence={
            "reference_token_id": int(a_ref),
            "reference_token": vocab.get(str(a_ref), str(a_ref)),
            "d_contrast_exposure": d_exp,
            "d_contrast_placebo": d_plac,
            "max_abs_placebo": max_plac,
            "ratio_contrast_space": ratio,
            "ratio_probability_space": p_ratio,
            "margin_required": MARGIN,
            "vs_controls": cmp_.as_dict() if cmp_ else None,
        },
        tables={"probability space vs logit-contrast space": rows},
        cannot_conclude=(
            "A contrast is relative to its reference. This one says the target did not "
            "move much *relative to the dominant token*; it cannot say which of the two "
            "moved in absolute terms, because a softmax has no absolute scale. Nothing "
            "here rules out the whole distribution having shifted for reasons connected "
            "to the exposure -- only that the target did not move against its main "
            "competitor in a way the placebo boundaries do not also show."
        ),
        caveats=[
            "The reference is the passage's modal top-1 across checkpoints. If a "
            "passage's argmax is unstable, the contrast mixes two references and the "
            "control comparison is noisier than it looks.",
            "Controls have different targets and different dominant tokens, so their "
            "contrasts are a null for 'how much does a target move against its own "
            "competitor', which is the right null here but not an exact match.",
            S.selection_note(True),
        ],
    )
