"""Is the prediction keyed on the passage the model was trained on?

If a single exposure left a retrievable trace, the trace has to be addressed by
something in the context.  The obvious address is the proper noun -- "McQuarrie" sits
80 tokens before the end and is the most distinctive string in the passage.

The passage set already contains the control for this: ten misspellings of that name
(Mcquarrie, McQuarr1e, McOuarrie, MCQUARRIE, ...) which the model has never seen in
this context.  If the exposure installed a name-keyed association, corrupting the key
should cost the target probability something.  If the ten corrupted versions predict
exactly what the real one predicts, the name is not the address, and nothing
passage-specific is being retrieved.
"""

from __future__ import annotations

import numpy as np

from .. import config as C
from ..measure import Bundle
from ..registry import ProbeResult, register

RULE = (
    "At the post-exposure checkpoint, compare the anchor's log p(target) with the ten "
    "never-trained name variants'. Let z = (anchor - mean(variants)) / sd(variants). "
    "SUPPORTED if z >= 3 (the trained name buys target probability that the corrupted "
    "names do not). REFUTED if |z| < 1 AND the anchor's target rank equals the modal "
    "variant rank -- the trained and untrained names are doing the same thing. "
    "Otherwise INCONCLUSIVE."
)


@register(
    "name_variant_equivalence",
    paper="Yu et al. 2025, arXiv:2506.21588 (verbatim memorization circuits) -- retrieval-key premise",
    hypothesis="A single exposure installed a passage-specific association addressed by the name.",
    question="Do never-trained misspellings of the name predict the same thing as the real one?",
    decision_rule=RULE,
    order=25,
)
def name_variant_equivalence(bundle: Bundle) -> ProbeResult:
    variants = bundle.variant_names
    if not variants:
        return ProbeResult(
            probe="name_variant_equivalence",
            paper="arXiv:2506.21588 (retrieval-key premise)",
            hypothesis="A single exposure installed a name-addressed association.",
            question="Do never-trained misspellings predict the same thing?",
            verdict="NOT_RUN", decision_rule=RULE,
            summary="No name variants in this passage set.",
            cannot_conclude="Nothing; the comparison was not available.",
        )

    step = C.EXPOSURE_BOUNDARY[1] if C.EXPOSURE_BOUNDARY[1] in bundle.checkpoints else bundle.checkpoints[-1]
    c_i = bundle.ci(step)
    a_i = bundle.pi(C.ANCHOR)
    v_idx = [bundle.pi(n) for n in variants]
    vocab = bundle.meta.get("vocab", {})

    a_lp = float(bundle.target_logp[c_i, a_i])
    v_lp = np.asarray([float(bundle.target_logp[c_i, i]) for i in v_idx])
    sd = float(v_lp.std(ddof=1)) if v_lp.size > 1 else 0.0
    z = float((a_lp - v_lp.mean()) / sd) if sd > 0 else float("nan")

    a_rank = int(bundle.target_rank[c_i, a_i])
    v_ranks = [int(bundle.target_rank[c_i, i]) for i in v_idx]
    modal_rank = int(np.bincount(v_ranks).argmax())

    a_top1 = int(bundle.top1_ids[c_i, a_i])
    same_top1 = sum(1 for i in v_idx if int(bundle.top1_ids[c_i, i]) == a_top1)

    rows = [
        {
            "passage": C.ANCHOR,
            "trained": True,
            "top1": vocab.get(str(a_top1), str(a_top1)),
            "p_top1": float(bundle.top1_probs[c_i, a_i]),
            "p_target": float(bundle.target_p[c_i, a_i]),
            "target_rank": a_rank,
        }
    ]
    for n, i in zip(variants, v_idx):
        t1 = int(bundle.top1_ids[c_i, i])
        rows.append(
            {
                "passage": n,
                "trained": False,
                "top1": vocab.get(str(t1), str(t1)),
                "p_top1": float(bundle.top1_probs[c_i, i]),
                "p_target": float(bundle.target_p[c_i, i]),
                "target_rank": int(bundle.target_rank[c_i, i]),
            }
        )

    if np.isfinite(z) and z >= 3:
        verdict = "SUPPORTED"
        summary = (
            f"At step{step} the trained name buys {z:.1f} sd more target log-probability "
            "than the never-trained spellings; the name is a retrieval key."
        )
    elif np.isfinite(z) and abs(z) < 1 and a_rank == modal_rank:
        verdict = "REFUTED"
        summary = (
            f"At step{step} the anchor sits {z:+.2f} sd from the ten never-trained "
            f"spellings (target rank {a_rank}, modal variant rank {modal_rank}; "
            f"{same_top1}/{len(variants)} share its argmax). Corrupting the name costs "
            "nothing, so nothing passage-specific is being addressed by it."
        )
    else:
        verdict = "INCONCLUSIVE"
        summary = (
            f"z = {z:+.2f} at step{step}: neither the retrieval pattern (z >= 3) nor the "
            f"equivalence pattern (|z| < 1 and matching rank; anchor {a_rank} vs modal "
            f"{modal_rank})."
        )

    return ProbeResult(
        probe="name_variant_equivalence",
        paper="arXiv:2506.21588 (retrieval-key premise)",
        hypothesis="A single exposure installed a passage-specific association addressed by the name.",
        question="Do never-trained misspellings of the name predict the same thing as the real one?",
        verdict=verdict,
        decision_rule=RULE,
        summary=summary,
        evidence={
            "checkpoint": step,
            "anchor_logp": a_lp,
            "variant_logp_mean": float(v_lp.mean()),
            "variant_logp_sd": sd,
            "z": z,
            "anchor_rank": a_rank,
            "variant_ranks": v_ranks,
            "modal_variant_rank": modal_rank,
            "variants_sharing_anchor_argmax": same_top1,
            "n_variants": len(variants),
        },
        tables={f"anchor vs never-trained name variants at step{step}": rows},
        cannot_conclude=(
            "The name is one candidate retrieval key, not the only one. A trace addressed "
            "by some other feature of the passage -- its topic, its formatting, the "
            "newline structure -- would survive this test untouched. It also cannot "
            "distinguish 'no trace' from 'a trace too small to move a 10-variant "
            "comparison'."
        ),
        caveats=[
            "The variants differ from the anchor only at positions "
            f"{C.NAME_POSITIONS[0]}-{C.NAME_POSITIONS[-1]}, so this is a clean "
            "substitution, but each misspelling also retokenises; a variant may occupy "
            "a different number of tokens in general. In this set all ten were "
            "constructed to preserve the four-token span.",
            "Ten variants give a sd estimate with 9 degrees of freedom. A z of 3 on "
            "that estimate is not a 3-sigma result in any strict sense.",
        ],
    )
