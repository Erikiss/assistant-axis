"""Does the anchor meet the entry criterion for "memorized" at all?

Every published account of verbatim memorization -- Carlini et al.'s k-extractability,
the memorization score used by the circuit-discovery work, the Pythia suite's own
"emergent memorization" metric -- is defined over the *greedy* continuation.  A
sequence is memorized when the model, given the prefix, emits the training continuation
under argmax decoding.

At state 60482 the argmax is " ph" (the model is writing "74 kmph") at every one of the
four checkpoints, with probability 0.64 to 0.80.  The target " per" is never the
model's choice.  Under any of those definitions this passage is not memorized, and the
mechanisms built to explain memorized passages have nothing to attach to.

This probe runs first because it is the cheapest thing in the suite and it decides
whether the rest of the suite is answering a real question.
"""

from __future__ import annotations

from .. import config as C
from ..measure import Bundle
from ..registry import ProbeResult, register

RULE = (
    "The anchor qualifies as verbatim-memorized iff the argmax continuation equals the "
    "held-out target at one or more checkpoints (the decoding rule every published "
    "memorization definition is stated over). SUPPORTED if argmax == target at >=1 "
    "checkpoint. SCOPE_FAILED if argmax != target at all four. No threshold on "
    "probability is used: the criterion is ordinal by construction."
)


@register(
    "memorization_entry",
    paper="Carlini et al. 2022 (extractability); Yu et al. 2025, arXiv:2506.21588 (circuit discovery)",
    hypothesis="The effect is verbatim memorization of the training passage.",
    question="Is the held-out target the model's greedy continuation at any checkpoint?",
    decision_rule=RULE,
    order=10,
)
def memorization_entry(bundle: Bundle) -> ProbeResult:
    p_i = bundle.pi(C.ANCHOR)
    vocab = bundle.meta.get("vocab", {})

    rows = []
    hits = 0
    for step in bundle.checkpoints:
        c_i = bundle.ci(step)
        top1 = int(bundle.top1_ids[c_i, p_i])
        tgt = int(bundle.target_ids[p_i])
        match = top1 == tgt
        hits += int(match)
        rows.append(
            {
                "checkpoint": step,
                "argmax_id": top1,
                "argmax": vocab.get(str(top1), str(top1)),
                "p_argmax": float(bundle.top1_probs[c_i, p_i]),
                "target": vocab.get(str(tgt), str(tgt)),
                "p_target": float(bundle.target_p[c_i, p_i]),
                "target_rank": int(bundle.target_rank[c_i, p_i]),
                "argmax_is_target": match,
            }
        )

    # The greedy continuation, when aux measurements were run, makes the point concrete.
    greedy_rows = []
    greedy = (bundle.aux or {}).get("greedy", {})
    for step in bundle.checkpoints:
        toks = greedy.get(str(step), {}).get(C.ANCHOR)
        if not toks:
            continue
        text = "".join(vocab.get(str(int(t)), f"<{int(t)}>") for t in toks)
        greedy_rows.append(
            {
                "checkpoint": step,
                "first_token": vocab.get(str(int(toks[0])), str(int(toks[0]))),
                "continuation": text,
                "starts_with_target": int(toks[0]) == int(bundle.target_ids[p_i]),
            }
        )

    if hits:
        verdict = "SUPPORTED"
        summary = (
            f"The target is the greedy continuation at {hits}/{len(bundle.checkpoints)} "
            "checkpoints; the memorization frame has something to explain."
        )
    else:
        best = max(rows, key=lambda r: r["p_target"])
        verdict = "SCOPE_FAILED"
        summary = (
            f"The argmax is {rows[0]['argmax']!r} (p={rows[0]['p_argmax']:.3f}) at every "
            f"checkpoint, never the target {rows[0]['target']!r} "
            f"(best p={best['p_target']:.5f}, rank {best['target_rank']}). The model is "
            "writing \"74 kmph\", not \"74 km per hour\". Under every published "
            "definition this passage is not memorized."
        )

    return ProbeResult(
        probe="memorization_entry",
        paper="Carlini et al. 2022; Yu et al. 2025 (arXiv:2506.21588)",
        hypothesis="The effect is verbatim memorization of the training passage.",
        question="Is the held-out target the model's greedy continuation at any checkpoint?",
        verdict=verdict,
        decision_rule=RULE,
        summary=summary,
        evidence={
            "argmax_ever_equals_target": bool(hits),
            "checkpoints_matched": hits,
            "n_checkpoints": len(bundle.checkpoints),
        },
        tables={"argmax vs target, per checkpoint": rows, **({"greedy continuation": greedy_rows} if greedy_rows else {})},
        cannot_conclude=(
            "This says the passage is not *verbatim* memorized under a greedy criterion. "
            "It does not rule out a weaker influence of the single exposure on the "
            "probability of the target -- that is what the remaining probes measure. It "
            "also cannot distinguish 'never memorized' from 'memorized and then "
            "forgotten by step130000', because no earlier checkpoint was measured."
        ),
        caveats=[
            "The greedy-continuation table is only present when the auxiliary "
            "measurements were run; without it the verdict rests on the argmax alone, "
            "which is the same criterion but only one token deep.",
            "A probability-based memorization definition (e.g. a threshold on p(target)) "
            "would give a different answer. No published definition in this suite's "
            "candidate list uses one.",
        ],
    )
