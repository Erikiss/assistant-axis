"""Focus Directions: do any heads read the span the memorization story needs?

Tang et al., *Focus Directions Make Your Language Models Pay More Attention to Relevant
Contexts* (arXiv:2503.23306), identifies "contextual heads" -- heads whose query/key
directions route attention onto the span that actually answers the question -- and
shows that steering along those directions improves retrieval from long contexts.

For that machinery to be the explanation here, some head somewhere has to be reading
the relevant span in the first place.  The span the memorization account depends on is
the name at positions 124-127.  This probe asks the prior question: does *any* head, at
*any* layer, put more than a uniform share of its attention there?

Head-averaged attention would hide a single head doing the work, so this probe uses the
per-head tensor, which is stored for the anchor family.
"""

from __future__ import annotations

import numpy as np

from .. import config as C
from ..measure import Bundle
from ..registry import ProbeResult, register

#: A head must beat the uniform share by this factor to count as reading the span.
ENRICHMENT = 3.0

RULE = (
    "Using per-head final-query attention, compute each head's mass on the name span "
    f"(positions {C.NAME_POSITIONS[0]}-{C.NAME_POSITIONS[-1]}). The uniform share is "
    "4/T. A head READS the span if its mass exceeds "
    f"{ENRICHMENT}x uniform. SUPPORTED if at least one head reads the span AND its mass "
    "changes across the exposure boundary by more than the largest change it shows "
    "across the placebo boundaries. REFUTED if no head anywhere exceeds "
    f"{ENRICHMENT}x uniform. INCONCLUSIVE if a head reads the span but its change at "
    "the exposure boundary does not exceed its placebo changes."
)


@register(
    "focus_directions",
    paper="Tang et al. 2025, Focus Directions (arXiv:2503.23306)",
    hypothesis="Contextual heads route attention onto the relevant span; the effect lives there.",
    question="Does any head read the name span at all, and does the exposure change that?",
    decision_rule=RULE,
    order=50,
)
def focus_directions(bundle: Bundle) -> ProbeResult:
    if bundle.attn_heads.size == 0 or C.ANCHOR not in bundle.attn_head_names:
        return ProbeResult(
            probe="focus_directions", paper="arXiv:2503.23306",
            hypothesis="Contextual heads route attention onto the relevant span.",
            question="Does any head read the name span?",
            verdict="NOT_RUN", decision_rule=RULE,
            summary="Per-head attention was not stored for the anchor.",
            cannot_conclude="Nothing; the measurement was not made.",
        )

    f_i = bundle.attn_head_names.index(C.ANCHOR)
    T = bundle.attn_heads.shape[-1]
    pos = list(C.NAME_POSITIONS)
    uniform = len(pos) / T
    threshold = ENRICHMENT * uniform

    # [ckpt, layer, head] -- mass on the name span
    mass = bundle.attn_heads[:, f_i][:, :, :, pos].sum(axis=-1)

    step = C.EXPOSURE_BOUNDARY[1] if C.EXPOSURE_BOUNDARY[1] in bundle.checkpoints else bundle.checkpoints[-1]
    c_i = bundle.ci(step)
    at_step = mass[c_i]                          # [layer, head]
    best = np.unravel_index(int(at_step.argmax()), at_step.shape)
    best_mass = float(at_step[best])
    readers = [
        {"layer": int(l), "head": int(h), "mass": float(at_step[l, h]),
         "enrichment": float(at_step[l, h] / uniform)}
        for l, h in zip(*np.where(at_step > threshold))
    ]
    readers.sort(key=lambda r: -r["mass"])

    # change across boundaries, per head
    def delta(boundary: tuple[int, int]) -> np.ndarray:
        lo, hi = boundary
        return mass[bundle.ci(hi)] - mass[bundle.ci(lo)]

    boundaries = [b for b in C.BOUNDARIES if b[0] in bundle.checkpoints and b[1] in bundle.checkpoints]
    d_exp = delta(C.EXPOSURE_BOUNDARY) if C.EXPOSURE_BOUNDARY in boundaries else None
    d_plac = [delta(b) for b in boundaries if b != C.EXPOSURE_BOUNDARY]

    exposure_exceeds = False
    reader_detail = []
    for r in readers[:10]:
        l, h = r["layer"], r["head"]
        de = float(d_exp[l, h]) if d_exp is not None else float("nan")
        dp = [float(d[l, h]) for d in d_plac]
        exceeds = bool(np.isfinite(de) and dp and abs(de) > max(abs(x) for x in dp))
        exposure_exceeds = exposure_exceeds or exceeds
        reader_detail.append(
            {**r, "delta_exposure": de,
             "delta_placebo_max_abs": max((abs(x) for x in dp), default=float("nan")),
             "exposure_exceeds_placebo": exceeds}
        )

    per_layer_max = [
        {"layer": int(l), "best_head": int(at_step[l].argmax()),
         "mass": float(at_step[l].max()), "enrichment": float(at_step[l].max() / uniform)}
        for l in range(at_step.shape[0])
    ]

    if not readers:
        verdict = "REFUTED"
        summary = (
            f"At step{step} the strongest head on the name span is layer {best[0]} "
            f"head {best[1]} with {best_mass:.5f} of its mass there, against a uniform "
            f"share of {uniform:.5f} -- an enrichment of "
            f"{best_mass / uniform:.2f}x, below the {ENRICHMENT}x threshold. No head at "
            "any layer reads the name. There is no contextual head here for focus "
            "directions to steer."
        )
    elif exposure_exceeds:
        verdict = "SUPPORTED"
        summary = (
            f"{len(readers)} head(s) read the name span at step{step} (best: layer "
            f"{readers[0]['layer']} head {readers[0]['head']}, "
            f"{readers[0]['enrichment']:.1f}x uniform), and at least one moves more "
            "across the exposure boundary than across either placebo boundary."
        )
    else:
        verdict = "INCONCLUSIVE"
        summary = (
            f"{len(readers)} head(s) read the name span at step{step} (best "
            f"{readers[0]['enrichment']:.1f}x uniform), but none moves more at the "
            "exposure boundary than at the placebo boundaries."
        )

    return ProbeResult(
        probe="focus_directions",
        paper="Tang et al. 2025, Focus Directions (arXiv:2503.23306)",
        hypothesis="Contextual heads route attention onto the relevant span; the effect lives there.",
        question="Does any head read the name span at all, and does the exposure change that?",
        verdict=verdict,
        decision_rule=RULE,
        summary=summary,
        evidence={
            "checkpoint": step,
            "uniform_share": uniform,
            "threshold": threshold,
            "best_head": {"layer": int(best[0]), "head": int(best[1]), "mass": best_mass,
                          "enrichment": best_mass / uniform},
            "n_reading_heads": len(readers),
            "readers": reader_detail,
        },
        tables={
            f"strongest head per layer on the name span, step{step}": per_layer_max,
            **({"heads above threshold": reader_detail} if reader_detail else {}),
        },
        cannot_conclude=(
            "The name span is one choice of 'relevant context'. This probe does not test "
            "whether some other span is read -- and in the original paper's setting the "
            "relevant span is defined by a question, which this passage does not have. "
            "A negative result here says the memorization story's own span is unread; it "
            "does not say the model attends to nothing meaningful."
        ),
        caveats=[
            "Attention mass is not information flow. A head could read the span through "
            "a small weight and still carry it; only ablation would settle that.",
            "Per-head tensors are stored for the anchor family only, so there is no "
            "control distribution for this statistic -- the comparison is against the "
            "uniform share, which is weaker.",
        ],
    )
