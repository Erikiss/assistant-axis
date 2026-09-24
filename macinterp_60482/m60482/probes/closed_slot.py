"""The three continuations are one slot, and the slot's total barely moves.

After `" 74 km"` English offers three ways to write the unit, and the tokenizer offers
exactly three entry points to them:

    " ph"   ->  kmph
    "/"     ->  km/h
    " per"  ->  km per hour

Those are the top three continuations at every checkpoint, and their probabilities sum
to 0.96685 / 0.97876 / 0.96549 / 0.95802 -- a mean of 0.967 with a standard deviation
of 0.009, while the members themselves swing by up to 0.16.  Between 2.1% and 4.2% of
the distribution lies outside them at any checkpoint.

That is a near-closed slot, and it changes what the headline number is.  Inside a fixed
budget the three probabilities cannot move independently: `" per"` rising by 0.014 while
`" ph"` falls by 0.054 is not two facts, it is one reallocation.  A mechanistic account
of the effect has to explain the reallocation among three unit conventions, not a
belief about a news passage -- and the quantity to report is the target's share *within*
the slot, which is what this probe computes.
"""

from __future__ import annotations

import numpy as np

from .. import config as C
from .. import stats as S
from ..measure import Bundle
from ..registry import ProbeResult, register

#: The slot counts as closed when its total varies this much less than its members do.
STABILITY_FACTOR = 4.0

RULE = (
    "Take the union of each checkpoint's top-3 continuations for the anchor. The slot "
    "is CLOSED if the standard deviation of the slot total across checkpoints is at "
    f"least {STABILITY_FACTOR}x smaller than the mean standard deviation of its "
    "individual members. SUPPORTED (the effect is a reallocation inside a fixed "
    "budget) if the slot is closed. REFUTED if the slot total moves as much as its "
    "members do, in which case the target's probability is free to move on its own and "
    "the raw delta is the right statistic after all."
)


@register(
    "closed_slot",
    paper="(no paper -- structural fact about the measurement; tokenizer competition)",
    hypothesis="The three candidate continuations form a near-closed budget, so the target cannot move independently.",
    question="Do the top three continuations sum to a constant while their members swing?",
    decision_rule=RULE,
    order=16,
)
def closed_slot(bundle: Bundle) -> ProbeResult:
    a_i = bundle.pi(C.ANCHOR)
    vocab = bundle.meta.get("vocab", {})
    tgt = int(bundle.target_ids[a_i])

    members = sorted(
        {int(t) for s in bundle.checkpoints for t in bundle.topk_ids[bundle.ci(s), a_i][:3]}
    )

    def prob(c_i: int, tid: int) -> float:
        where = np.flatnonzero(bundle.topk_ids[c_i, a_i] == tid)
        return float(bundle.topk_probs[c_i, a_i][where[0]]) if where.size else 0.0

    per_ckpt = {}
    for s in bundle.checkpoints:
        c_i = bundle.ci(s)
        per_ckpt[s] = {t: prob(c_i, t) for t in members}

    totals = np.asarray([sum(per_ckpt[s].values()) for s in bundle.checkpoints])
    member_sd = float(
        np.mean([np.std([per_ckpt[s][t] for s in bundle.checkpoints], ddof=1) for t in members])
    )
    total_sd = float(totals.std(ddof=1)) if totals.size > 1 else 0.0
    stability = member_sd / total_sd if total_sd > 0 else float("inf")
    closed = stability >= STABILITY_FACTOR

    rows = []
    for s in bundle.checkpoints:
        row = {"checkpoint": s}
        for t in members:
            row[vocab.get(str(t), str(t))] = per_ckpt[s][t]
        row["slot_total"] = float(sum(per_ckpt[s].values()))
        row["outside_slot"] = 1.0 - row["slot_total"]
        row["target_share_of_slot"] = (
            per_ckpt[s].get(tgt, 0.0) / row["slot_total"]
            if row["slot_total"] > 0 else float("nan")
        )
        rows.append(row)

    # the renormalised statistic, against the raw one
    def share(s: int) -> float:
        # The target need not be in the slot at all: for a control passage the top-3
        # are three other tokens. Its share is then 0, not a KeyError.
        tot = sum(per_ckpt[s].values())
        return per_ckpt[s].get(tgt, 0.0) / tot if tot > 0 else float("nan")

    boundaries = [
        b for b in C.BOUNDARIES
        if b[0] in bundle.checkpoints and b[1] in bundle.checkpoints
    ]
    d_share = {b: share(b[1]) - share(b[0]) for b in boundaries}
    d_raw = {
        b: float(bundle.target_p[bundle.ci(b[1]), a_i] - bundle.target_p[bundle.ci(b[0]), a_i])
        for b in boundaries
    }

    contrast_rows = [
        {
            "boundary": f"{b[0]} -> {b[1]}",
            "exposure": b == C.EXPOSURE_BOUNDARY,
            "delta_p_target": d_raw[b],
            "delta_share_of_slot": d_share[b],
        }
        for b in boundaries
    ]

    # do the controls also sit in a closed slot? if every passage does, this is a
    # property of next-token prediction, not of this passage.
    ctrl_stability = []
    for n in bundle.control_names:
        i = bundle.pi(n)
        ms = sorted({int(t) for s in bundle.checkpoints for t in bundle.topk_ids[bundle.ci(s), i][:3]})
        pc = {
            s: {
                t: (
                    float(bundle.topk_probs[bundle.ci(s), i][np.flatnonzero(bundle.topk_ids[bundle.ci(s), i] == t)[0]])
                    if np.any(bundle.topk_ids[bundle.ci(s), i] == t) else 0.0
                )
                for t in ms
            }
            for s in bundle.checkpoints
        }
        tt = np.asarray([sum(pc[s].values()) for s in bundle.checkpoints])
        msd = float(np.mean([np.std([pc[s][t] for s in bundle.checkpoints], ddof=1) for t in ms]))
        tsd = float(tt.std(ddof=1)) if tt.size > 1 else 0.0
        ctrl_stability.append(msd / tsd if tsd > 0 else float("inf"))

    finite_ctrl = [c for c in ctrl_stability if np.isfinite(c)]
    cmp_ = (
        S.compare_to_controls("slot_stability", stability, finite_ctrl, direction="greater")
        if finite_ctrl and np.isfinite(stability) else None
    )

    names = ", ".join(repr(vocab.get(str(t), str(t))) for t in members)
    target_in_slot = tgt in members
    if not target_in_slot:
        return ProbeResult(
            probe="closed_slot",
            paper="(structural fact about the measurement)",
            hypothesis="The three candidate continuations form a near-closed budget.",
            question="Do the top three continuations sum to a constant while their members swing?",
            verdict="INCONCLUSIVE",
            decision_rule=RULE,
            summary=(
                f"The target {vocab.get(str(tgt), str(tgt))!r} is not among this "
                "passage's top-3 at any checkpoint, so there is no slot containing it "
                "to be closed or open."
            ),
            evidence={"members": [int(t) for t in members], "target_in_slot": False},
            tables={"the slot, per checkpoint": rows},
            cannot_conclude=(
                "Nothing about the anchor. This branch exists so the probe returns a "
                "verdict rather than raising when handed a passage whose target sits "
                "outside the competition -- which is the normal case for a control."
            ),
        )

    if closed:
        verdict = "SUPPORTED"
        summary = (
            f"The {len(members)} continuations {names} sum to "
            f"{totals.mean():.5f} +- {total_sd:.5f} across the four checkpoints, while "
            f"their members move {member_sd:.5f} on average -- a {stability:.1f}x "
            f"difference. Only {(1 - totals.max()) * 100:.1f}-{(1 - totals.min()) * 100:.1f}% "
            "of the distribution lies outside them. The target cannot move "
            "independently: its rise and the dominant token's fall are one reallocation "
            "inside a fixed budget, not two facts."
        )
    else:
        verdict = "REFUTED"
        summary = (
            f"The slot total moves {total_sd:.5f} against member movement of "
            f"{member_sd:.5f} ({stability:.1f}x, below the {STABILITY_FACTOR}x "
            "threshold). The budget is not fixed, so the target's probability is free "
            "to move on its own."
        )

    return ProbeResult(
        probe="closed_slot",
        paper="(structural fact about the measurement)",
        hypothesis="The three candidate continuations form a near-closed budget, so the target cannot move independently.",
        question="Do the top three continuations sum to a constant while their members swing?",
        verdict=verdict,
        decision_rule=RULE,
        summary=summary,
        evidence={
            "members": [int(t) for t in members],
            "member_tokens": [vocab.get(str(t), str(t)) for t in members],
            "slot_total_mean": float(totals.mean()),
            "slot_total_sd": total_sd,
            "mean_member_sd": member_sd,
            "stability_factor": stability,
            "threshold": STABILITY_FACTOR,
            "closed": bool(closed),
            "delta_p_target": {f"{b[0]}->{b[1]}": v for b, v in d_raw.items()},
            "delta_share_of_slot": {f"{b[0]}->{b[1]}": v for b, v in d_share.items()},
            "vs_controls": cmp_.as_dict() if cmp_ else None,
        },
        tables={
            "the slot, per checkpoint": rows,
            "raw delta vs share-of-slot delta": contrast_rows,
        },
        cannot_conclude=(
            "A closed slot says the three probabilities are coupled; it does not say "
            "which of them the training touched. The share-of-slot delta is the "
            "renormalised statistic, and it still moves at the exposure boundary -- "
            "read it with the noise_floor and softmax_renormalization probes, which "
            "ask whether that movement clears the null. Nor does this identify the "
            "slot's cause: that the three tokens are the three ways English writes a "
            "speed unit is an observation about the strings, not a measurement."
        ),
        caveats=[
            "The slot membership is taken from the top-3 at each checkpoint, so a "
            "passage whose top-3 changes composition has a larger union and a "
            "mechanically less stable total.",
            "If the controls show the same stability, a closed top-3 slot is a property "
            "of next-token prediction in general rather than of this passage -- which "
            "is what the control comparison in the evidence reports.",
            S.selection_note(False),
        ],
    )
