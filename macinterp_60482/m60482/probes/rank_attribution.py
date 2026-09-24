"""The rank flip at the exposure boundary: whose movement produced it?

The headline observation was that the target dropped from rank 2 to rank 3 across the
exposure boundary.  Rank is a comparison, so a rank change has two possible sources:
the target moved, or something it was being compared against moved.

Across 131000 -> 132000 the target's probability went *up*, from 0.09087 to 0.10461.
It lost a rank anyway, because the competitor "/" went up faster.  A rank statistic
that moves in the opposite direction to the quantity it is supposed to track is
measuring the competitor, and the competitor swings just as hard at the placebo
boundaries (0.1468 -> 0.0897 -> 0.1166 -> 0.2119).
"""

from __future__ import annotations

import numpy as np

from .. import config as C
from ..measure import Bundle
from ..registry import ProbeResult, register

RULE = (
    "Decompose the rank change across the exposure boundary into the target's own "
    "movement and the movement of each token that crossed it. The rank statistic is "
    "attributable to the TARGET iff |dlogp(target)| >= |dlogp(crosser)| for every token "
    "that changed sides. SUPPORTED if so. REFUTED if any crosser moved further than the "
    "target, or if the target's probability moved in the direction opposite to its rank. "
    "INCONCLUSIVE if the rank did not change at all."
)


@register(
    "rank_attribution",
    paper="(no paper -- statistical decomposition of the headline observation)",
    hypothesis="The rank-2-to-3 flip reflects a change in the target token.",
    question="Did the target move, or did the tokens it is ranked against move?",
    decision_rule=RULE,
    order=20,
)
def rank_attribution(bundle: Bundle) -> ProbeResult:
    a_i = bundle.pi(C.ANCHOR)
    vocab = bundle.meta.get("vocab", {})
    lo, hi = C.EXPOSURE_BOUNDARY
    if lo not in bundle.checkpoints or hi not in bundle.checkpoints:
        return ProbeResult(
            probe="rank_attribution", paper="(decomposition)",
            hypothesis="The rank flip reflects a change in the target token.",
            question="Did the target move, or its competitors?", verdict="NOT_RUN",
            decision_rule=RULE, summary="exposure boundary not in bundle",
            cannot_conclude="Nothing; the measurement was not made.",
        )

    c_lo, c_hi = bundle.ci(lo), bundle.ci(hi)
    tgt = int(bundle.target_ids[a_i])
    r_lo, r_hi = int(bundle.target_rank[c_lo, a_i]), int(bundle.target_rank[c_hi, a_i])
    p_lo, p_hi = float(bundle.target_p[c_lo, a_i]), float(bundle.target_p[c_hi, a_i])
    dlp_tgt = float(np.log(max(p_hi, 1e-12)) - np.log(max(p_lo, 1e-12)))

    # everything in either top-k, with its probability at both checkpoints
    ids = sorted(set(int(x) for x in bundle.topk_ids[c_lo, a_i]) |
                 set(int(x) for x in bundle.topk_ids[c_hi, a_i]))

    def prob_of(c_i: int, tid: int) -> float:
        where = np.flatnonzero(bundle.topk_ids[c_i, a_i] == tid)
        return float(bundle.topk_probs[c_i, a_i][where[0]]) if where.size else 0.0

    rows = []
    crossers = []
    for tid in ids:
        a, b = prob_of(c_lo, tid), prob_of(c_hi, tid)
        if a <= 0 and b <= 0:
            continue
        dlp = float(np.log(max(b, 1e-12)) - np.log(max(a, 1e-12)))
        # a "crosser" changed sides relative to the target
        crossed = (a < p_lo) != (b < p_hi)
        row = {
            "token_id": tid,
            "token": vocab.get(str(tid), str(tid)),
            "is_target": tid == tgt,
            "p_before": a,
            "p_after": b,
            "delta_p": b - a,
            "delta_logp": dlp,
            "crossed_the_target": bool(crossed and tid != tgt),
        }
        rows.append(row)
        if row["crossed_the_target"]:
            crossers.append(row)

    rows.sort(key=lambda r: -max(r["p_before"], r["p_after"]))

    direction_conflict = (r_hi > r_lo and p_hi > p_lo) or (r_hi < r_lo and p_hi < p_lo)
    bigger_crossers = [c for c in crossers if abs(c["delta_logp"]) > abs(dlp_tgt)]

    # A rank without its margin is not a measurement. Report how much probability was
    # actually separating the target from its neighbours, and whether that gap is
    # larger than the run-to-run jitter -- when the jitter has been measured.
    jitter = (bundle.aux or {}).get("jitter_band")

    def margin_at(c_i: int) -> dict:
        probs = bundle.topk_probs[c_i, a_i]
        ids = bundle.topk_ids[c_i, a_i]
        where = np.flatnonzero(ids == tgt)
        if not where.size:
            return {"gap_above": float("nan"), "gap_below": float("nan")}
        j = int(where[0])
        p_here = float(probs[j])
        above = float(probs[j - 1]) - p_here if j > 0 else float("inf")
        below = p_here - float(probs[j + 1]) if j + 1 < probs.size else float("inf")
        return {
            "p_target": p_here,
            "gap_above": above,
            "gap_below": below,
            "nearest_gap": min(above, below),
            "relative_nearest_gap": min(above, below) / p_here if p_here > 0 else float("nan"),
        }

    margins = []
    for step in bundle.checkpoints:
        m = margin_at(bundle.ci(step))
        row = {
            "checkpoint": step,
            "rank": int(bundle.target_rank[bundle.ci(step), a_i]),
            **{k: v for k, v in m.items()},
        }
        if jitter is not None:
            row["jitter_band"] = float(jitter)
            row["rank_resolved"] = bool(m.get("nearest_gap", 0) > float(jitter))
        margins.append(row)

    unresolved = [
        m["checkpoint"] for m in margins if m.get("rank_resolved") is False
    ]

    if r_lo == r_hi:
        verdict = "INCONCLUSIVE"
        summary = f"The target's rank did not change ({r_lo} -> {r_hi}); nothing to attribute."
    elif direction_conflict or bigger_crossers:
        verdict = "REFUTED"
        parts = []
        if direction_conflict:
            parts.append(
                f"the target's probability moved {p_lo:.5f} -> {p_hi:.5f} "
                f"({'up' if p_hi > p_lo else 'down'}) while its rank moved "
                f"{r_lo} -> {r_hi} ({'worse' if r_hi > r_lo else 'better'})"
            )
        if bigger_crossers:
            c = max(bigger_crossers, key=lambda x: abs(x["delta_logp"]))
            parts.append(
                f"{c['token']!r} crossed it moving {c['delta_logp']:+.5f} in log-prob "
                f"against the target's {dlp_tgt:+.5f}"
            )
        summary = "The rank statistic is driven by the competition, not the target: " + "; ".join(parts) + "."
    else:
        verdict = "SUPPORTED"
        summary = (
            f"The target moved {dlp_tgt:+.5f} in log-prob, further than any token that "
            f"crossed it; the rank change tracks the target."
        )

    margin_note = (
        f" The rank was held by a margin of {margins[bundle.ci(lo)]['nearest_gap']:.5f} "
        f"at step{lo} ({margins[bundle.ci(lo)]['relative_nearest_gap'] * 100:.1f}% of "
        "the target's own probability)"
        + (
            f", against a measured run-to-run jitter of {float(jitter):.5f}"
            + (
                f" -- the rank is UNRESOLVED at step(s) {unresolved}."
                if unresolved else ", so the rank is resolved."
            )
            if jitter is not None
            else "; no jitter band has been measured, so whether that margin is "
                 "resolvable is unknown (see m60482.aux.measure_jitter)."
        )
    )

    return ProbeResult(
        probe="rank_attribution",
        paper="(decomposition)",
        hypothesis="The rank-2-to-3 flip reflects a change in the target token.",
        question="Did the target move, or did the tokens it is ranked against move?",
        verdict=verdict,
        decision_rule=RULE,
        summary=summary + margin_note,
        evidence={
            "rank_before": r_lo, "rank_after": r_hi,
            "margins": margins,
            "jitter_band": float(jitter) if jitter is not None else None,
            "unresolved_checkpoints": unresolved,
            "p_before": p_lo, "p_after": p_hi,
            "dlogp_target": dlp_tgt,
            "crossers": crossers,
            "crossers_larger_than_target": [c["token"] for c in bigger_crossers],
            "direction_conflict": bool(direction_conflict),
        },
        tables={
            f"top-of-distribution movement, {lo} -> {hi}": rows[:15],
            "rank margins per checkpoint": margins,
        },
        cannot_conclude=(
            "It says the rank statistic is the wrong instrument here; it does not say "
            "the target's own probability change is noise. That is the noise_floor "
            "probe's question, and the two must be read together."
        ),
        caveats=[
            "Tokens outside the stored top-k are treated as probability 0, which "
            "overstates the log-probability change for anything that entered or left "
            "the top-k. Crossers near the target are unaffected, since they are in the "
            "top-k at both checkpoints by construction.",
            "A rank reported without its margin is not a measurement (Miller 2024, "
            "arXiv:2411.00640). The margins table is the margin; a rank whose nearest "
            "gap is inside the run-to-run jitter band should be read as unresolved, "
            "not as a value.",
        ],
    )
