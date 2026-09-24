"""Is the attention profile frozen, and can 1000 steps encode a single exposure?

The strangest number in the original measurement is not the sink -- it is the *sign* of
the shift. The anchor's attention profile moves LESS across every boundary than 40 out
of 40 controls do (TV 0.017 against a control median of 0.033, p = 1.000 at the
exposure boundary). A passage that had just been trained on moved less than passages
that had not.

Gu et al. (arXiv:2410.10781) predicts exactly that, and predicts it as a non-finding:
the attention sink emerges within the first few thousand optimizer steps and is
essentially fixed thereafter. By step130000 of Pythia's 143000, the profile has been
saturated for something like 128000 steps. Two checkpoints 1000 steps apart then cannot
encode a single exposure, because they cannot encode much of anything -- and the
attention null follows from which checkpoints were measured, not from the absence of an
effect.

The probe distinguishes that from the alternative reading ("attention was measured and
showed nothing") by asking whether the profiles are frozen for *everything*.
"""

from __future__ import annotations

import numpy as np

from .. import config as C
from .. import stats as S
from ..measure import Bundle
from ..registry import ProbeResult, register

#: A profile counts as frozen when its across-boundary TV distance is below this.
FROZEN_TV = 0.05

RULE = (
    "For every passage and every boundary, compute the total-variation distance between "
    "the head-averaged final-query profiles, averaged over layers. The profile is FROZEN "
    f"if the median TV across all passages and all boundaries is below {FROZEN_TV}. "
    "SUPPORTED (the attention null is a consequence of the checkpoint spacing) if the "
    "profile is frozen AND the exposure boundary's TV distribution is inside the "
    "placebo boundaries' range. REFUTED if the profile is not frozen -- attention does "
    "move at this spacing, so its null at the exposure boundary is a real null. "
    "INCONCLUSIVE if frozen but the exposure boundary stands out."
)


@register(
    "sink_stability",
    paper="Gu et al. 2024, When Attention Sink Emerges in Language Models (arXiv:2410.10781) -- NOT on the candidate list",
    hypothesis="The profile froze early in training, so 1000-step boundaries cannot encode a single exposure.",
    question="Does the attention profile move at all between adjacent checkpoints, for any passage?",
    decision_rule=RULE,
    order=42,
)
def sink_stability(bundle: Bundle) -> ProbeResult:
    if not bundle.meta.get("attention_captured"):
        return ProbeResult(
            probe="sink_stability", paper="arXiv:2410.10781",
            hypothesis="The attention profile froze early in training.",
            question="Does the profile move between adjacent checkpoints?",
            verdict="NOT_RUN", decision_rule=RULE,
            summary="Attention was not captured in this bundle.",
            cannot_conclude="Nothing; the measurement was not made.",
        )

    boundaries = [
        b for b in C.BOUNDARIES
        if b[0] in bundle.checkpoints and b[1] in bundle.checkpoints
    ]
    if not boundaries:
        return ProbeResult(
            probe="sink_stability", paper="arXiv:2410.10781",
            hypothesis="The attention profile froze early in training.",
            question="Does the profile move between adjacent checkpoints?",
            verdict="NOT_RUN", decision_rule=RULE,
            summary="No complete boundary in this bundle.",
            cannot_conclude="Nothing; the measurement was not made.",
        )

    def tv(p_i: int, boundary: tuple[int, int]) -> float:
        lo, hi = boundary
        a = bundle.attn_last[bundle.ci(lo), p_i]
        b = bundle.attn_last[bundle.ci(hi), p_i]
        return float(S.total_variation(a, b, axis=-1).mean())   # mean over layers

    a_i = bundle.pi(C.ANCHOR)
    ctrl_idx = [bundle.pi(n) for n in bundle.control_names]

    per_boundary = {}
    rows = []
    for b in boundaries:
        anchor_tv = tv(a_i, b)
        ctrl_tv = [tv(i, b) for i in ctrl_idx]
        cmp_ = S.compare_to_controls("tv", anchor_tv, ctrl_tv, direction="greater")
        per_boundary[b] = {"anchor": anchor_tv, "controls": ctrl_tv, "cmp": cmp_}
        rows.append(
            {
                "boundary": f"{b[0]} -> {b[1]}",
                "exposure": b == C.EXPOSURE_BOUNDARY,
                "anchor_tv": anchor_tv,
                "control_median": cmp_.control_median,
                "control_min": cmp_.control_min,
                "control_max": cmp_.control_max,
                "k_controls_above": cmp_.k_at_or_above,
                "p_greater": cmp_.p_one_sided,
            }
        )

    all_tv = np.asarray(
        [v["anchor"] for v in per_boundary.values()]
        + [t for v in per_boundary.values() for t in v["controls"]]
    )
    median_tv = float(np.median(all_tv))
    frozen = median_tv < FROZEN_TV

    exposure_present = C.EXPOSURE_BOUNDARY in per_boundary
    inside = True
    if exposure_present and len(per_boundary) > 1:
        exp = np.asarray(per_boundary[C.EXPOSURE_BOUNDARY]["controls"])
        plac = np.concatenate([
            np.asarray(v["controls"])
            for b, v in per_boundary.items() if b != C.EXPOSURE_BOUNDARY
        ])
        # do the exposure boundary's control TVs look like the placebo boundaries'?
        inside = bool(
            np.median(exp) <= np.quantile(plac, 0.95)
            and np.median(exp) >= np.quantile(plac, 0.05)
        )

    anchor_below_all = all(
        v["cmp"].k_at_or_above >= v["cmp"].n_controls for v in per_boundary.values()
    )

    if not frozen:
        verdict = "REFUTED"
        summary = (
            f"Profiles do move at this spacing: median TV across all passages and "
            f"boundaries is {median_tv:.4f}, above the {FROZEN_TV} frozen threshold. "
            "Attention is not saturated here, so its null at the exposure boundary is a "
            "real null rather than an artefact of checkpoint spacing."
        )
    elif inside:
        verdict = "SUPPORTED"
        summary = (
            f"The profile is effectively frozen: median TV across all passages and all "
            f"boundaries is {median_tv:.4f} (< {FROZEN_TV}), and the exposure boundary's "
            "spread is inside the placebo boundaries'. Two checkpoints 1000 steps apart "
            "at step~130000 of 143000 cannot encode a single exposure, so the attention "
            "null is a consequence of which checkpoints were measured."
            + (
                " The anchor moves less than every control at every boundary, which is "
                "what a frozen profile plus a low-entropy passage looks like, not a "
                "trace of training."
                if anchor_below_all else ""
            )
        )
    else:
        verdict = "INCONCLUSIVE"
        summary = (
            f"The profile is frozen (median TV {median_tv:.4f}), but the exposure "
            "boundary's spread sits outside the placebo boundaries', so something "
            "distinguishes it that this probe does not identify."
        )

    return ProbeResult(
        probe="sink_stability",
        paper="Gu et al. 2024 (arXiv:2410.10781)",
        hypothesis="The profile froze early in training, so 1000-step boundaries cannot encode a single exposure.",
        question="Does the attention profile move at all between adjacent checkpoints, for any passage?",
        verdict=verdict,
        decision_rule=RULE,
        summary=summary,
        evidence={
            "median_tv_all_passages_all_boundaries": median_tv,
            "frozen_threshold": FROZEN_TV,
            "frozen": bool(frozen),
            "exposure_spread_inside_placebo": bool(inside),
            "anchor_below_every_control_at_every_boundary": bool(anchor_below_all),
            "per_boundary": {
                f"{b[0]}->{b[1]}": {
                    "anchor_tv": v["anchor"],
                    "control_median": v["cmp"].control_median,
                    "p_greater": v["cmp"].p_one_sided,
                }
                for b, v in per_boundary.items()
            },
        },
        tables={"attention profile shift per boundary": rows},
        cannot_conclude=(
            "The frozen reading is an inference from the measured checkpoints, not a "
            "test of the emergence claim. Confirming it means measuring early "
            "checkpoints -- step1000 through step10000 -- where the sink is supposed to "
            "form, and showing the TV distances there are large. This suite measures "
            "four checkpoints near step130000 and cannot see that."
        ),
        caveats=[
            "TV is computed on head-averaged profiles; a head that moves while others "
            "compensate is invisible here.",
            "This probe reads raw attention weight. See norm_attribution.",
            S.selection_note(False),
            S.multiplicity_note(len(boundaries)),
        ],
    )
