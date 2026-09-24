"""Does the sink survive when attention is weighted by what it actually carries?

The headline attention result is a null: the anchor's attention profile shifts *less*
across the exposure boundary than 40 out of 40 controls do.  That was read as "no
exposure signal in attention".

Kobayashi et al. (arXiv:2004.10102) is the reason that reading is not yet available.
Raw attention weight is not attribution.  The quantity that reaches the residual stream
is ``alpha_j * v_j``, and sink positions are precisely the ones whose value vectors are
drained -- that is *how* a sink works as a no-op.  A position can hold 0.69 of the
attention mass and contribute almost nothing.  So a null in raw attention weight is a
null in a quantity that does not measure contribution, and it is uninformative in both
directions: a signal there would have been equally hard to interpret.

This probe recomputes the profile as ``|alpha_j| * ||v_j||`` and asks two things: does
the newline sink survive the reweighting, and does the *contribution* profile behave
differently from the raw one across the exposure boundary.
"""

from __future__ import annotations

import numpy as np

from .. import config as C
from .. import stats as S
from ..measure import Bundle
from ..registry import ProbeResult, register

DEEP_FROM = 3
#: The sink counts as evaporating if norm-weighting removes this fraction of its mass.
EVAPORATION = 0.5

RULE = (
    "Recompute the final query's profile as |alpha_j| * ||v_j|| per head, normalised "
    "over positions and averaged over heads. Let m_raw and m_norm be the deep-layer "
    "mass on the raw profile's top-2 positions. The sink EVAPORATES if "
    f"m_norm <= {EVAPORATION} * m_raw. SUPPORTED (attention weight was not measuring "
    "contribution, so the attention null carries no information) if the sink "
    "evaporates. REFUTED if the sink survives reweighting (m_norm > "
    f"{EVAPORATION} * m_raw), which would make the raw attention null informative "
    "after all. NOT_RUN without the norm-weighted measurement."
)


@register(
    "norm_attribution",
    paper="Kobayashi et al. 2020, Attention is Not Only a Weight (arXiv:2004.10102) -- NOT on the candidate list",
    hypothesis="The attention null is uninformative because attention weight is not attribution.",
    question="Does the newline sink survive weighting by the norm of what it carries?",
    decision_rule=RULE,
    order=45,
)
def norm_attribution(bundle: Bundle) -> ProbeResult:
    aux = bundle.aux or {}
    norm = aux.get("norm_attn")
    if norm is None or not bundle.meta.get("attention_captured"):
        return ProbeResult(
            probe="norm_attribution",
            paper="arXiv:2004.10102",
            hypothesis="The attention null is uninformative because weight is not attribution.",
            question="Does the sink survive weighting by the norm of what it carries?",
            verdict="NOT_RUN",
            decision_rule=RULE,
            summary=(
                "Norm-weighted attention was not measured. Run "
                "m60482.aux.measure_norm_attribution."
            ),
            cannot_conclude=(
                "Nothing -- and that is the point. Without this measurement the "
                "attention_sink and focus_directions results rest on raw attention "
                "weight, which does not measure contribution. Their nulls should be "
                "read as 'not measured' rather than 'no effect'."
            ),
        )

    norm = np.asarray(norm)
    steps = list(aux.get("norm_attn_steps", C.EXPOSURE_BOUNDARY))
    nnames = list(aux.get("norm_attn_names", bundle.names))

    step = steps[-1]
    s_i = steps.index(step)
    a_i = nnames.index(C.ANCHOR)
    ctx = bundle.context_tokens.get(C.ANCHOR, [])

    raw = bundle.attn_last[bundle.ci(step), bundle.pi(C.ANCHOR), DEEP_FROM:].mean(axis=0)
    top2 = np.argsort(-raw)[:2]
    m_raw = float(raw[top2].sum())

    nrm = norm[s_i, a_i, DEEP_FROM:].mean(axis=0)
    m_norm = float(nrm[top2].sum())

    # where does contribution actually concentrate, once reweighted?
    n_top2 = np.argsort(-nrm)[:2]

    def label(pos: int) -> str:
        return repr(ctx[pos]) if pos < len(ctx) else f"pos{pos}"

    evaporates = m_norm <= EVAPORATION * m_raw

    rows = [
        {
            "position": int(p),
            "token": label(int(p)),
            "raw_attention": float(raw[p]),
            "norm_weighted": float(nrm[p]),
            "ratio": float(nrm[p] / raw[p]) if raw[p] > 0 else float("nan"),
        }
        for p in list(top2) + [p for p in n_top2 if p not in top2]
    ]

    # Does the anchor's contribution profile differ from the controls'?
    ctrl = [n for n in bundle.control_names if n in nnames]
    cmp_ = None
    if ctrl:
        ctrl_mass = []
        for n in ctrl:
            c_prof = norm[s_i, nnames.index(n), DEEP_FROM:].mean(axis=0)
            ctrl_mass.append(float(np.sort(c_prof)[-2:].sum()))
        cmp_ = S.compare_to_controls("norm_top2_mass", float(np.sort(nrm)[-2:].sum()),
                                     ctrl_mass, direction="greater")

    if evaporates:
        verdict = "SUPPORTED"
        summary = (
            f"At step{step} the two positions holding {m_raw:.3f} of raw deep-layer "
            f"attention ({', '.join(label(int(p)) for p in top2)}) hold only "
            f"{m_norm:.3f} once weighted by the norm of what they carry -- "
            f"{m_norm / m_raw * 100:.0f}% of their apparent share. Contribution "
            f"concentrates instead on {', '.join(label(int(p)) for p in n_top2)}. The "
            "raw-attention null therefore says nothing about whether the exposure "
            "changed what the model reads."
        )
    else:
        verdict = "REFUTED"
        summary = (
            f"At step{step} the sink survives reweighting: {m_raw:.3f} raw against "
            f"{m_norm:.3f} norm-weighted ({m_norm / m_raw * 100:.0f}% retained). The "
            "positions holding the attention are also carrying the content, so the raw "
            "attention profile was measuring contribution after all and its null is "
            "informative."
        )

    return ProbeResult(
        probe="norm_attribution",
        paper="Kobayashi et al. 2020 (arXiv:2004.10102)",
        hypothesis="The attention null is uninformative because attention weight is not attribution.",
        question="Does the newline sink survive weighting by the norm of what it carries?",
        verdict=verdict,
        decision_rule=RULE,
        summary=summary,
        evidence={
            "checkpoint": step,
            "raw_top2_positions": [int(p) for p in top2],
            "raw_top2_tokens": [label(int(p)) for p in top2],
            "raw_top2_mass": m_raw,
            "norm_weighted_mass_same_positions": m_norm,
            "retained_fraction": m_norm / m_raw if m_raw > 0 else float("nan"),
            "norm_top2_positions": [int(p) for p in n_top2],
            "norm_top2_tokens": [label(int(p)) for p in n_top2],
            "evaporates": bool(evaporates),
            "vs_controls": cmp_.as_dict() if cmp_ else None,
        },
        tables={f"raw vs norm-weighted attention, step{step}": rows},
        cannot_conclude=(
            "||alpha * v|| is a better proxy for contribution than alpha, not a "
            "measurement of it. It ignores the output projection (which rescales heads "
            "but not positions within a head, so the per-position ordering here is "
            "unaffected) and it ignores cancellation between positions: two large "
            "contributions that cancel look large here and matter not at all. Only "
            "ablation or patching measures contribution directly, and this suite runs "
            "neither."
        ),
        caveats=[
            "The value split relies on the fused GPT-NeoX QKV layout; "
            "m60482.aux._neox_value_states asserts the projection width rather than "
            "assuming it, because a silent mis-split would return numbers computed "
            "from the query projection.",
            "Measured at the two checkpoints bracketing the exposure only, to keep the "
            "second forward pass per passage affordable.",
            S.selection_note(False),
        ],
    )
