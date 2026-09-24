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
from .. import stats as S
from ..measure import Bundle
from ..registry import ProbeResult, register

RULE = (
    "Apply the anchor's own selection rule to every control: take each control's "
    "largest absolute boundary change over the three boundaries, exactly as the anchor "
    "was picked, and rank the anchor's selected statistic inside that distribution. "
    "SUPPORTED if the anchor's selected statistic is above the 95th percentile of the "
    "controls' selected statistics -- it survives being chosen the same way they were. "
    "REFUTED if it sits inside their range. INCONCLUSIVE if there are too few controls "
    "to rank against. The state counts below are a LEDGER, not a test: they say how "
    "many comparisons were made, and no count of them can support a hypothesis."
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

    # The claim was directional -- the target's probability was said to RISE at the
    # exposure boundary. For that, the pattern is "largest in magnitude AND positive",
    # which chance produces with probability 1/(2*n_b) rather than 1/n_b.
    directional_hits = [
        r["state"] for r in rows
        if r["largest_is_exposure"] and r["d_exposure"] > 0
    ]
    p_dir = 1.0 / (2 * n_b)
    expected_dir = n_states * p_dir
    p_at_least_one_dir = 1.0 - (1.0 - p_dir) ** n_states

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

    # The only defensible test here: put the anchor through the selection rule that
    # picked it, and put every control through the same rule. Comparing a selected
    # maximum against unselected values is how a 1-in-3 coin flip becomes a finding.
    a_i = bundle.pi(C.ANCHOR)
    anchor_selected = max(abs(dlogp(a_i, b)) for b in boundaries)
    ctrl_selected = [
        max(abs(dlogp(bundle.pi(n), b)) for b in boundaries) for n in bundle.control_names
    ]

    cmp_ = (
        S.compare_to_controls(
            "selected_max_abs_dlogp", anchor_selected, ctrl_selected, direction="greater"
        )
        if ctrl_selected else None
    )

    if cmp_ is None:
        verdict = "INCONCLUSIVE"
        summary = (
            "No controls to apply the selection rule to; the ledger below counts the "
            "comparisons but nothing tests them."
        )
    elif cmp_.p_one_sided <= 0.05:
        verdict = "SUPPORTED"
        summary = (
            f"Put through the same selection rule as the anchor -- largest |change| "
            f"over {n_b} boundaries -- the anchor's {anchor_selected:.5f} is at the "
            f"{cmp_.percentile:.0f}th percentile of the controls' selected maxima "
            f"(p = {cmp_.p_one_sided:.4f}). It survives being chosen the way they were."
        )
    else:
        verdict = "REFUTED"
        summary = (
            f"Selected the same way the anchor was -- largest |change| over {n_b} "
            f"boundaries -- the anchor's {anchor_selected:.5f} sits at the "
            f"{cmp_.percentile:.0f}th percentile of the controls' selected maxima "
            f"(p = {cmp_.p_one_sided:.4f}, median {cmp_.control_median:.5f}). "
            f"Separately, {observed} of {n_states} states put their largest change on "
            f"the exposure boundary against {expected:.1f} expected by chance."
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
            "directional_observed": len(directional_hits),
            "directional_expected_by_chance": expected_dir,
            "directional_p_at_least_one": p_at_least_one_dir,
            "directional_states": directional_hits,
            "anchor_rank_among_states": anchor_rank,
            "anchor_is_largest": bool(anchor_is_largest),
            "anchor_selected_statistic": anchor_selected,
            "selection_matched_comparison": cmp_.as_dict() if cmp_ else None,
            "attainable_p_floor": C.MIN_ATTAINABLE_P,
            "comparisons_in_this_suite": n_states * n_b,
            "bonferroni_alpha_needed": 0.05 / (n_states * n_b),
            "floor_survives_correction": C.MIN_ATTAINABLE_P <= 0.05 / (n_states * n_b),
        },
        tables={"largest boundary change per state": rows},
        cannot_conclude=(
            "The state counts are a ledger, not a test: no count of comparisons can "
            "support a hypothesis, and the verdict above rests only on the "
            "selection-matched control comparison. Note also that the anchor being the "
            "largest of the 19 states is guaranteed by how it was chosen and is "
            "therefore reported, never used. "
            "The ledger counts comparisons; it does not say the anchor's effect is zero. "
            "A real effect can also be selected for. What it does say is that this "
            "design cannot distinguish the two, because the smallest attainable p "
            f"({C.MIN_ATTAINABLE_P:.4f}) is larger than the corrected alpha "
            f"({0.05 / (n_states * n_b):.5f}) that {n_states * n_b} comparisons require. "
            "No amount of care in the analysis fixes that; only more controls or a "
            "pre-registered single comparison would."
        ),
        caveats=[
            note_shared_boundary,
            f"Two counts are reported. The magnitude version asks how often a state's "
            f"largest |change| lands on the exposure boundary (chance 1/{n_b}); the "
            "directional version asks how often it lands there AND is positive "
            f"(chance 1/{2 * n_b}), which is the shape the original claim had. The "
            f"directional version expects {expected_dir:.1f} of {n_states} states by "
            f"chance, with P(at least one) = {p_at_least_one_dir:.3f}.",
        ],
    )
