"""The explanation the candidate list does not contain.

The measured attention profile is not subtle: from layer 3 onward the final query puts
between 17% and 69% of its head-averaged mass on position 46 -- a newline, 160 tokens
before the end -- and a further ~19% on position 0.  Two positions, neither of them
content, absorb most of the attention at every depth.

That is the attention-sink / massive-activation phenomenon (Xiao et al., StreamingLLM,
arXiv:2309.17453; Sun et al., Massive Activations in LLMs, arXiv:2402.17762), in which
models route surplus attention onto semantically empty tokens -- the first token,
newlines, delimiters -- as a no-op.  None of the six candidate papers is about it.

If the profile is a sink profile and the anchor's is indistinguishable from the
controls', then "where the model looks" at this position is a property of the
architecture and the formatting, not of this passage or its exposure, and no
attention-based explanation of the effect can get started.
"""

from __future__ import annotations

import numpy as np

from .. import config as C
from .. import stats as S
from ..measure import Bundle
from ..registry import ProbeResult, register

#: Layers below this are excluded: early layers legitimately attend locally.
DEEP_FROM = 3
#: Mass threshold on the two dominant positions for the profile to count as a sink.
SINK_MASS = 0.50

RULE = (
    f"Take the head-averaged final-query attention at layers >= {DEEP_FROM}, averaged "
    "over layers. Let m be the mass on the two highest positions. The profile is a SINK "
    f"profile if m >= {SINK_MASS} and both positions are non-content (first token or a "
    "delimiter). SUPPORTED (sink explains the profile) if it is a sink profile AND the "
    "anchor's sink mass lies inside the 40 controls' range, i.e. control-referenced "
    "p > 0.05 in both directions. REFUTED if the profile is not a sink profile. "
    "INCONCLUSIVE if it is a sink profile but the anchor's mass is outside the control "
    "range, which would make the sink itself passage-specific and need its own "
    "explanation."
)


@register(
    "attention_sink",
    paper="Xiao et al. 2023 (arXiv:2309.17453); Sun et al. 2024 (arXiv:2402.17762) -- NOT on the candidate list",
    hypothesis="The attention profile is a generic sink, not a read of this passage.",
    question="Is the final position's attention a sink profile, and is the anchor's any different from a control's?",
    decision_rule=RULE,
    order=40,
)
def attention_sink(bundle: Bundle) -> ProbeResult:
    if not bundle.meta.get("attention_captured"):
        return ProbeResult(
            probe="attention_sink", paper="arXiv:2309.17453 / arXiv:2402.17762",
            hypothesis="The attention profile is a generic sink.",
            question="Is the final position's attention a sink profile?",
            verdict="NOT_RUN", decision_rule=RULE,
            summary="Attention was not captured in this bundle.",
            cannot_conclude="Nothing; the measurement was not made.",
        )

    step = C.EXPOSURE_BOUNDARY[1] if C.EXPOSURE_BOUNDARY[1] in bundle.checkpoints else bundle.checkpoints[-1]
    c_i = bundle.ci(step)
    a_i = bundle.pi(C.ANCHOR)
    ctx = bundle.context_tokens.get(C.ANCHOR, [])

    def sink_mass(p_i: int) -> tuple[float, list[int]]:
        prof = bundle.attn_last[c_i, p_i, DEEP_FROM:].mean(axis=0)   # [T]
        top2 = np.argsort(-prof)[:2]
        return float(prof[top2].sum()), [int(x) for x in top2]

    mass, positions = sink_mass(a_i)
    ctrl_mass = [sink_mass(bundle.pi(n))[0] for n in bundle.control_names]

    hi = S.compare_to_controls("sink_mass", mass, ctrl_mass, direction="greater")
    lo = S.compare_to_controls("sink_mass", mass, ctrl_mass, direction="less")
    inside_controls = hi.p_one_sided > 0.05 and lo.p_one_sided > 0.05

    def label(pos: int) -> str:
        tok = ctx[pos] if pos < len(ctx) else "?"
        return repr(tok)

    def is_content(pos: int) -> bool:
        if pos == 0:
            return False
        tok = ctx[pos] if pos < len(ctx) else ""
        return bool(tok.strip()) and not all(ch in " \n\t.,;:/-()" for ch in tok)

    non_content = not any(is_content(p) for p in positions)
    is_sink = mass >= SINK_MASS and non_content

    per_layer = [
        {
            "layer": l,
            "argmax_position": int(bundle.attn_last[c_i, a_i, l].argmax()),
            "token": label(int(bundle.attn_last[c_i, a_i, l].argmax())),
            "weight": float(bundle.attn_last[c_i, a_i, l].max()),
        }
        for l in range(bundle.n_layers)
    ]

    uniform = 2.0 / bundle.attn_last.shape[3]

    if not is_sink:
        verdict = "REFUTED"
        summary = (
            f"At step{step} the two dominant positions hold {mass:.3f} of deep-layer "
            f"attention ({', '.join(label(p) for p in positions)}), which does not meet "
            f"the sink criterion (>= {SINK_MASS} on non-content positions)."
        )
    elif inside_controls:
        verdict = "SUPPORTED"
        summary = (
            f"At step{step}, {mass:.3f} of deep-layer attention sits on positions "
            f"{positions} ({', '.join(label(p) for p in positions)}) against a uniform "
            f"baseline of {uniform:.4f} -- a {mass / uniform:.0f}x concentration on "
            "non-content tokens. The 40 controls do the same thing (median "
            f"{hi.control_median:.3f}, range {hi.control_min:.3f}-{hi.control_max:.3f}); "
            "the anchor is unremarkable among them. This is a generic sink, not a read "
            "of this passage."
        )
    else:
        verdict = "INCONCLUSIVE"
        summary = (
            f"The profile is a sink profile ({mass:.3f} on {positions}) but the anchor "
            f"sits outside the control range (p_greater={hi.p_one_sided:.4f}, "
            f"p_less={lo.p_one_sided:.4f}); the sink itself would then be "
            "passage-specific and needs its own explanation."
        )

    return ProbeResult(
        probe="attention_sink",
        paper="Xiao et al. 2023 (arXiv:2309.17453); Sun et al. 2024 (arXiv:2402.17762)",
        hypothesis="The attention profile is a generic sink, not a read of this passage.",
        question="Is the final position's attention a sink profile, and is the anchor's different from a control's?",
        verdict=verdict,
        decision_rule=RULE,
        summary=summary,
        evidence={
            "checkpoint": step,
            "deep_layers_from": DEEP_FROM,
            "sink_mass_top2": mass,
            "sink_positions": positions,
            "sink_tokens": [label(p) for p in positions],
            "positions_are_non_content": non_content,
            "uniform_baseline_top2": uniform,
            "inside_control_range": bool(inside_controls),
            "vs_controls_greater": hi.as_dict(),
            "vs_controls_less": lo.as_dict(),
        },
        tables={f"where each layer looks, anchor, step{step}": per_layer},
        cannot_conclude=(
            "A sink profile explains where attention goes; it does not explain the "
            "target probability. Attention mass is not attribution -- a head with 2% of "
            "the mass on a position can still carry the information that decides the "
            "prediction. Establishing that would need an ablation or a patching "
            "experiment, which this probe does not run."
        ),
        caveats=[
            "Head-averaging hides heads that disagree with the layer. The "
            "focus_directions probe looks per-head for exactly this reason.",
            "'Non-content' is decided by a token-string heuristic (whitespace and "
            "punctuation), not by a semantic judgement.",
            S.selection_note(False),
        ],
    )
