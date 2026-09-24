"""How much of the 207-token context does the prediction actually use?

The target is " per" after "... wind speeds could reach approximately 74 km".  In
English text "km" is followed by "per" (as in "km per hour"), by "/" (as in "km/h"),
or by "ph" (as in "kmph").  Those three are exactly the top three continuations, in
that order of competition, at every checkpoint.

That raises a question the whole study depends on: does p(" per") here reflect
anything about *this passage* at all, or is it the model's continuation prior for the
bigram "74 km"?  The truncation ladder answers it directly -- re-run the same target
with only the last k tokens of context, for k from 1 to the full 207.  If the value at
k=2 already matches the full-context value, the remaining 205 tokens, the name, and the
training exposure are all irrelevant to the number being studied.
"""

from __future__ import annotations

import numpy as np

from .. import config as C
from .. import stats as S
from ..measure import Bundle
from ..registry import ProbeResult, register

#: Relative tolerance at which the short-context value counts as "already there".
TOL = 0.20

RULE = (
    "Using the truncation ladder at the post-exposure checkpoint, let p_full be "
    "p(target | 207 tokens) and p_k be p(target | last k tokens). The prediction is a "
    "LOCAL continuation prior -- carrying no passage-specific information -- if "
    f"|p_2 - p_full| / p_full <= {TOL}. If so the verdict is SCOPE_FAILED for every "
    "passage-level hypothesis. It is CONTEXT-DEPENDENT, and the passage-level "
    "hypotheses stay live, if that ratio exceeds "
    f"{TOL} and p_k rises monotonically with k over at least half the ladder."
)


@register(
    "context_dependence",
    paper="(no paper -- tokenizer/bigram prior check; the gap the candidate list leaves)",
    hypothesis="p(target) is a local continuation prior for '74 km', carrying no passage-specific information.",
    question="How short can the context be before p(target) changes?",
    decision_rule=RULE,
    order=30,
)
def context_dependence(bundle: Bundle) -> ProbeResult:
    aux = bundle.aux or {}
    ladder = aux.get("ladder")
    if ladder is None:
        return ProbeResult(
            probe="context_dependence",
            paper="(bigram prior check)",
            hypothesis="p(target) is a local continuation prior, not passage-specific.",
            question="How short can the context be before p(target) changes?",
            verdict="NOT_RUN",
            decision_rule=RULE,
            summary=(
                "The truncation ladder was not measured. Run with auxiliary "
                "measurements enabled (m60482.aux.measure_aux)."
            ),
            cannot_conclude=(
                "Nothing. Without this probe the suite cannot distinguish a "
                "passage-level effect from the model's continuation prior for '74 km', "
                "which is the single most likely explanation of the whole finding."
            ),
        )

    ladder = np.asarray(ladder)
    ks = list(aux.get("ladder_k", []))
    names = list(aux.get("ladder_names", bundle.names))
    step = C.EXPOSURE_BOUNDARY[1] if C.EXPOSURE_BOUNDARY[1] in bundle.checkpoints else bundle.checkpoints[-1]
    c_i = bundle.ci(step)
    a_i = names.index(C.ANCHOR)

    curve = ladder[c_i, a_i]
    p_full = float(curve[-1])
    k2_i = ks.index(2) if 2 in ks else 0
    p_k2 = float(curve[k2_i])
    ratio = abs(p_k2 - p_full) / p_full if p_full > 0 else float("inf")

    diffs = np.diff(curve)
    monotone_frac = float(np.mean(diffs >= 0)) if diffs.size else 0.0

    rows = [
        {
            "k": k,
            "p_target": float(curve[i]),
            "relative_to_full": float(curve[i] / p_full) if p_full > 0 else float("nan"),
        }
        for i, k in enumerate(ks)
    ]

    # Do the controls behave the same way? If every passage's target is already set by
    # k=2, the ladder is telling us about the measurement, not about the anchor.
    ctrl_idx = [names.index(n) for n in bundle.control_names if n in names]
    ctrl_ratio = []
    for i in ctrl_idx:
        c = ladder[c_i, i]
        if c[-1] > 0:
            ctrl_ratio.append(abs(float(c[k2_i]) - float(c[-1])) / float(c[-1]))
    cmp_ = (
        S.compare_to_controls("ladder_k2_ratio", ratio, ctrl_ratio, direction="greater")
        if ctrl_ratio
        else None
    )

    if ratio <= TOL:
        verdict = "SUPPORTED"
        summary = (
            f"At step{step}, p(target) from the last 2 tokens alone is {p_k2:.5f} against "
            f"{p_full:.5f} from the full 207 ({ratio * 100:.1f}% apart). The other 205 "
            "tokens -- the name, the topic, the training exposure -- contribute nothing "
            "measurable. The quantity under study is a local continuation prior, and "
            "every passage-level hypothesis -- memorization included -- is explaining "
            "something that is not there."
        )
    elif monotone_frac >= 0.5:
        verdict = "REFUTED"
        summary = (
            f"At step{step}, p(target) rises from {p_k2:.5f} at k=2 to {p_full:.5f} at "
            f"k=207 ({ratio * 100:.1f}% apart), monotone over {monotone_frac * 100:.0f}% "
            "of the ladder. The context is doing work, so passage-level hypotheses "
            "remain live."
        )
    else:
        verdict = "INCONCLUSIVE"
        summary = (
            f"p(target) differs by {ratio * 100:.1f}% between k=2 and full context but "
            f"the ladder is not monotone (only {monotone_frac * 100:.0f}% of steps "
            "increase); the context changes the prediction without accumulating into it."
        )

    return ProbeResult(
        probe="context_dependence",
        paper="(bigram prior check)",
        hypothesis="p(target) is a local continuation prior for '74 km', carrying no passage-specific information.",
        question="How short can the context be before p(target) changes?",
        verdict=verdict,
        decision_rule=RULE,
        summary=summary,
        evidence={
            "checkpoint": step,
            "p_full_context": p_full,
            "p_two_tokens": p_k2,
            "relative_gap": ratio,
            "monotone_fraction": monotone_frac,
            "control_comparison": cmp_.as_dict() if cmp_ else None,
        },
        tables={f"truncation ladder, anchor, step{step}": rows},
        cannot_conclude=(
            "A flat ladder rules out the *measured quantity* depending on the passage. "
            "It does not rule out the exposure having changed something else about the "
            "model -- only that whatever changed is not visible in p(target) given this "
            "context. Nor does it identify which short-context feature carries the "
            "prediction: ' km' alone, ' 74 km', or the numeric pattern."
        ),
        caveats=[
            "Truncating a context also removes the BOS-like first position, which is a "
            "strong attention sink here. Short-context values therefore differ from "
            "long-context ones for a reason unrelated to content, and that is a "
            "confound this probe cannot separate. Compare against the control "
            "distribution, not against zero.",
            S.selection_note(False),
        ],
    )
