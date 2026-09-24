"""How often would a passage look like this if nothing had happened to any of them?

The study has 19 states.  Each has three checkpoint boundaries, one of which is its
exposure boundary.  Under the null -- nothing happened to anybody -- the largest of a
state's three boundary changes lands on its exposure boundary with probability 1/3, by
symmetry and nothing else.

Across 19 states that gives an expected count of about 6.3 such states, and a
probability of at least one that rounds to 1.  A state selected *because* it showed that
pattern is therefore not evidence of anything until the selection is accounted for, and
nothing in the original analysis accounted for it.

This probe computes the ledger for whatever states are in the bundle, and reports the
attainable p floor alongside, because the two constraints bite together: 40 controls
cannot produce a p below 1/41, and 19 states x 3 boundaries x the probes in this suite
is far more comparisons than that floor can survive.
"""

from __future__ import annotations

import numpy as np

from .. import config as C
from ..measure import Bundle
from ..registry import ProbeResult, register

RULE = (
    "Under the null, a state's largest absolute boundary change lands on its exposure "
    "boundary with probability 1/n_boundaries. Count how many of the bundle's states "
    "actually show that pattern and compare with the expected count. SUPPORTED (the "
    "anchor stands out beyond selection) if the observed count is at or below the "
    "expected count AND the anchor is the single largest effect among all states. "
    "REFUTED if the observed count is consistent with chance -- the anchor is then one "
    "of several states showing a pattern that chance produces at this rate. "
    "INCONCLUSIVE if fewer than 5 states are present."
)


@register(
    "multiplicity_ledger",
    paper="Card et al. 2020, With Little Power Come Great Responsibilities (arXiv:2010.06595) -- NOT on the candidate list",
    hypothesis="The anchor's pattern is rarer than the selection procedure would produce by chance.",
    question="How many of the 19 states show the anchor's pattern, and how many should by chance?",
    decision_rule=RULE,
    order=12,
)
def multiplicity_ledger(bundle: Bundle) -> ProbeResult:
    boundaries = [
        b for b in C.BOUNDARIES
        if b[0] in bundle.checkpoints and b[1] in bundle.checkpoints
    ]
    states = list(bundle.state_names)
    if len(states) < 5 or C.EXPOSURE_BOUNDARY not in boundaries:
        return ProbeResult(
            probe="multiplicity_ledger",
            paper="arXiv:2010.06595",
            hypothesis="The anchor's pattern is rarer than selection would produce.",
            question="How many states show the anchor's pattern by chance?",
            verdict="INCONCLUSIVE",
            decision_rule=RULE,
            summary=f"Only {len(states)} states present; the ledger needs at least 5.",
            cannot_conclude=(
                "Nothing about multiplicity. Note that this does not make the "
                "comparisons fewer -- it only makes them uncountable from this bundle."
            ),
        )

    def dlogp(p_i: int, b: tuple[int, int]) -> float:
        return float(
            bundle.target_logp[bundle.ci(b[1]), p_i] - bundle.target_logp[bundle.ci(b[0]), p_i]
        )

    rows = []
    hits = []
    for name in states:
        p_i = bundle.pi(name)
        ds = {b: dlogp(p_i, b) for b in boundaries}
        biggest = max(ds, key=lambda b: abs(ds[b]))
        hit = biggest == C.EXPOSURE_BOUNDARY
        if hit:
            hits.append(name)
        rows.append(
            {
                "state": name,
                "d_exposure": ds[C.EXPOSURE_BOUNDARY],
                "largest_boundary": f"{biggest[0]} -> {biggest[1]}",
                "largest_is_exposure": hit,
                "abs_d_exposure": abs(ds[C.EXPOSURE_BOUNDARY]),
            }
        )

    rows.sort(key=lambda r: -r["abs_d_exposure"])
    n_states = len(states)
    n_b = len(boundaries)
    expected = n_states / n_b
    observed = len(hits)
    p_at_least_one = 1.0 - (1.0 - 1.0 / n_b) ** n_states

    anchor_rank = next(
        (i + 1 for i, r in enumerate(rows) if r["state"] == C.ANCHOR), None
    )
    anchor_is_largest = anchor_rank == 1

    # every state's exposure boundary is the same checkpoints here, so the exposure
    # boundary of a state that was NOT trained in that window is itself a placebo.
    note_shared_boundary = (
        "All states in this bundle share one boundary set, so 'exposure boundary' is "
        "the anchor's. For the other states it is a placebo boundary, which is what "
        "makes them a usable reference for how often the pattern appears by chance."
    )

    if observed <= expected and anchor_is_largest:
        verdict = "SUPPORTED"
        summary = (
            f"{observed} of {n_states} states put their largest change on the exposure "
            f"boundary against {expected:.1f} expected by chance, and the anchor is the "
            "largest of them."
        )
    elif observed <= expected or not anchor_is_largest:
        verdict = "REFUTED"
        parts = []
        if observed > expected:
            parts.append(
                f"{observed} of {n_states} states show the pattern against "
                f"{expected:.1f} expected by chance"
            )
        if not anchor_is_largest:
            parts.append(
                f"the anchor is only the {anchor_rank}th largest exposure-boundary "
                f"change among {n_states} states"
            )
        summary = (
            "; ".join(parts).capitalize()
            + f". With {n_states} states and {n_b} boundaries, the probability that at "
            f"least one state shows this pattern by chance is {p_at_least_one:.3f}."
        )
    else:
        verdict = "REFUTED"
        summary = (
            f"{observed} of {n_states} states show the pattern against {expected:.1f} "
            f"expected; P(at least one by chance) = {p_at_least_one:.3f}."
        )

    return ProbeResult(
        probe="multiplicity_ledger",
        paper="Card et al. 2020 (arXiv:2010.06595)",
        hypothesis="The anchor's pattern is rarer than the selection procedure would produce by chance.",
        question="How many of the states show the anchor's pattern, and how many should by chance?",
        verdict=verdict,
        decision_rule=RULE,
        summary=summary,
        evidence={
            "n_states": n_states,
            "n_boundaries": n_b,
            "expected_by_chance": expected,
            "observed": observed,
            "states_with_largest_at_exposure": hits,
            "p_at_least_one_by_chance": p_at_least_one,
            "anchor_rank_among_states": anchor_rank,
            "anchor_is_largest": bool(anchor_is_largest),
            "attainable_p_floor": C.MIN_ATTAINABLE_P,
            "comparisons_in_this_suite": n_states * n_b,
            "bonferroni_alpha_needed": 0.05 / (n_states * n_b),
            "floor_survives_correction": C.MIN_ATTAINABLE_P <= 0.05 / (n_states * n_b),
        },
        tables={"largest boundary change per state": rows},
        cannot_conclude=(
            "The ledger counts comparisons; it does not say the anchor's effect is zero. "
            "A real effect can also be selected for. What it does say is that this "
            "design cannot distinguish the two, because the smallest attainable p "
            f"({C.MIN_ATTAINABLE_P:.4f}) is larger than the corrected alpha "
            f"({0.05 / (n_states * n_b):.5f}) that {n_states * n_b} comparisons require. "
            "No amount of care in the analysis fixes that; only more controls or a "
            "pre-registered single comparison would."
        ),
        caveats=[note_shared_boundary],
    )
