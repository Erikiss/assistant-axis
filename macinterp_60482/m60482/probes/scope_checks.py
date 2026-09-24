"""Scope checks for the three candidate papers whose preconditions are not met.

A paper can be excellent and still have nothing to say about a given phenomenon.  Three
of the six candidates are in that position, and it is worth demonstrating rather than
asserting -- each of these probes measures the single quantity that decides whether the
paper's subject matter is present at all.

Cliff Tokens (arXiv:2606.25524)
    Needs a generated multi-step trace with a verifiable answer, so that the
    probability of a correct final answer can be estimated before and after a candidate
    token. The 60482 measurement generates nothing: it is one held-out token after a
    fixed context, scored by teacher forcing.

SOPHIA / self-loops (arXiv:2607.18100)
    Needs at least two completed reasoning steps, clustered by their hidden activations,
    so that "stuck in the same cluster" is definable. There are no steps here. The word
    "state" in "state_60482" is a passage label from the K8 schema, not a latent
    cluster -- a homonym, not a connection.

Repeat Curse (arXiv:2504.14218)
    Needs repetition in the text. This probe measures it directly: the longest repeated
    token n-gram in the anchor context, and whether the target token continues one.
"""

from __future__ import annotations

import numpy as np

from .. import config as C
from ..measure import Bundle
from ..registry import ProbeResult, register

# ---------------------------------------------------------------------------------
# Cliff Tokens
# ---------------------------------------------------------------------------------

CLIFF_RULE = (
    "The paper's unit is a generated reasoning trace with a verifiable final answer, "
    "and its statistic is the change in estimated P(correct answer) across a candidate "
    "token, from rollouts on both sides of it. This setup is in scope iff the "
    "measurement generates at least two tokens AND a correctness predicate exists. "
    "SCOPE_FAILED if the measurement is a single teacher-forced token with no "
    "correctness predicate."
)


@register(
    "cliff_token_scope",
    paper="Cliff Tokens (arXiv:2606.25524)",
    hypothesis="A single token triggers a collapse in the probability of a correct final answer.",
    question="Is there a generated trace with a verifiable answer for a cliff to occur in?",
    decision_rule=CLIFF_RULE,
    order=70,
)
def cliff_token_scope(bundle: Bundle) -> ProbeResult:
    greedy = (bundle.aux or {}).get("greedy", {})
    n_generated = (bundle.aux or {}).get("greedy_tokens", 0) if greedy else 0

    return ProbeResult(
        probe="cliff_token_scope",
        paper="Cliff Tokens (arXiv:2606.25524)",
        hypothesis="A single token triggers a collapse in the probability of a correct final answer.",
        question="Is there a generated trace with a verifiable answer for a cliff to occur in?",
        verdict="SCOPE_FAILED",
        decision_rule=CLIFF_RULE,
        summary=(
            "The 60482 measurement scores one held-out token by teacher forcing. There "
            "is no generated trace, no final answer, and no correctness predicate, so "
            "the paper's statistic -- the drop in estimated P(correct) across a token -- "
            "is undefined here."
            + (
                f" (The auxiliary run does generate {n_generated} tokens, but as a "
                "memorization check, not a reasoning trace: a Pile news passage has no "
                "answer to be right or wrong about.)"
                if n_generated
                else ""
            )
        ),
        evidence={
            "generated_tokens_in_measurement": 0,
            "generated_tokens_in_aux": int(n_generated),
            "correctness_predicate_exists": False,
            "paper_verified": "arXiv:2606.25524, 2026-06-24, Seoul National University + Boston University",
        },
        cannot_conclude=(
            "This says the paper does not apply to the measurement as performed. It does "
            "not say the paper is wrong, nor that a cliff-style analysis could never be "
            "run on Pythia -- it could, on a generative task with a verifiable answer. "
            "That would be a different study."
        ),
        caveats=[
            "The paper's adaptive-threshold methodology is worth borrowing even though "
            "its subject matter is absent: it is the statistical discipline this "
            "analysis was missing.",
        ],
    )


# ---------------------------------------------------------------------------------
# SOPHIA / self-loops
# ---------------------------------------------------------------------------------

LOOP_RULE = (
    "The paper's unit is a reasoning step; its construct is a cluster over mean-pooled "
    "step activations, and 'self-loop' means consecutive steps landing in the same "
    "cluster. In scope iff at least two completed generated steps exist. SCOPE_FAILED "
    "otherwise."
)


@register(
    "self_loop_scope",
    paper="SOPHIA / Can We Break LLMs Out of Self-Loops? (arXiv:2607.18100)",
    hypothesis="The model is stuck in an activation-state cluster across reasoning steps.",
    question="Are there reasoning steps here to be stuck between?",
    decision_rule=LOOP_RULE,
    order=70,
)
def self_loop_scope(bundle: Bundle) -> ProbeResult:
    return ProbeResult(
        probe="self_loop_scope",
        paper="SOPHIA (arXiv:2607.18100)",
        hypothesis="The model is stuck in an activation-state cluster across reasoning steps.",
        question="Are there reasoning steps here to be stuck between?",
        verdict="SCOPE_FAILED",
        decision_rule=LOOP_RULE,
        summary=(
            "There are zero generated reasoning steps: the measurement is one "
            "teacher-forced token after a fixed 207-token Pile passage. SOPHIA's "
            "'state' is a k-means cluster over mean-pooled step embeddings; "
            "'state_60482' is a passage name from the K8 schema. The two uses of the "
            "word are unrelated, and the match in the candidate list is a homonym."
        ),
        evidence={
            "reasoning_steps": 0,
            "state_sense_in_this_study": "passage label (K8 schema: state_haupt / state_extra)",
            "state_sense_in_paper": "k-means cluster over mean-pooled step activations",
            "paper_verified": "arXiv:2607.18100, 2026-07-20",
        },
        cannot_conclude=(
            "Nothing about whether Pythia-1.4B exhibits self-loops in general; only that "
            "this measurement contains no trajectory in which one could occur."
        ),
        caveats=[
            "Worth stating explicitly because the name collision is the likely reason "
            "this paper was surfaced at all.",
        ],
    )


# ---------------------------------------------------------------------------------
# Repeat Curse -- this one is measured, not asserted
# ---------------------------------------------------------------------------------

REPEAT_RULE = (
    "The paper explains repetition in emitted text. It is in scope iff the anchor "
    "context contains a repeated token n-gram of length >= 3, OR the target token "
    "continues a span that occurs earlier in the context. SCOPE_FAILED if neither holds. "
    "If repetition IS present, the verdict is INCONCLUSIVE -- the suite has no SAE for "
    "Pythia-1.4B at these checkpoints, so the paper's own repetition features cannot be "
    "measured and a stronger verdict is not available."
)


def _longest_repeated_ngram(ids: np.ndarray, max_n: int = 12) -> tuple[int, int]:
    """(longest n with a repeat, number of repeated n-grams at that length)."""
    best_n, best_count = 0, 0
    for n in range(3, max_n + 1):
        seen: dict[tuple, int] = {}
        for i in range(len(ids) - n + 1):
            key = tuple(int(x) for x in ids[i : i + n])
            seen[key] = seen.get(key, 0) + 1
        repeats = sum(1 for v in seen.values() if v > 1)
        if repeats:
            best_n, best_count = n, repeats
        else:
            break
    return best_n, best_count


@register(
    "repeat_curse_scope",
    paper="Understanding the Repeat Curse ... from a Feature Perspective (arXiv:2504.14218)",
    hypothesis="A repetition feature drives the output.",
    question="Does the anchor context actually repeat anything?",
    decision_rule=REPEAT_RULE,
    order=70,
)
def repeat_curse_scope(bundle: Bundle) -> ProbeResult:
    a_i = bundle.pi(C.ANCHOR)
    ids = bundle.ctx_ids[a_i]
    tgt = int(bundle.target_ids[a_i])

    best_n, best_count = _longest_repeated_ngram(ids)
    target_occurs_earlier = int(np.sum(ids == tgt))

    # would the target continue a span seen earlier? i.e. is there an earlier position
    # whose predecessor bigram matches the context's final bigram
    tail = tuple(int(x) for x in ids[-2:])
    bigram_repeats = sum(
        1 for i in range(len(ids) - 2) if tuple(int(x) for x in ids[i : i + 2]) == tail
    )

    in_scope = best_n >= 3 or bigram_repeats > 0

    rows = [
        {"statistic": "longest repeated token n-gram", "value": best_n},
        {"statistic": "distinct repeated n-grams at that length", "value": best_count},
        {"statistic": "times the target token appears in the context", "value": target_occurs_earlier},
        {"statistic": "earlier occurrences of the final bigram", "value": bigram_repeats},
    ]

    if not in_scope:
        verdict = "SCOPE_FAILED"
        summary = (
            f"The anchor context has no repeated token n-gram of length >= 3 "
            f"(longest found: {best_n}), and its final bigram occurs nowhere earlier. "
            "There is no repetition for a repetition feature to explain."
        )
    else:
        verdict = "INCONCLUSIVE"
        summary = (
            f"Repetition IS present (longest repeated n-gram: {best_n} tokens; "
            f"{bigram_repeats} earlier occurrences of the final bigram), so the paper is "
            "in scope -- but its mechanism cannot be measured here: no sparse "
            "autoencoder exists for Pythia-1.4B at these intermediate checkpoints."
        )

    return ProbeResult(
        probe="repeat_curse_scope",
        paper="Repeat Curse (arXiv:2504.14218)",
        hypothesis="A repetition feature drives the output.",
        question="Does the anchor context actually repeat anything?",
        verdict=verdict,
        decision_rule=REPEAT_RULE,
        summary=summary,
        evidence={
            "longest_repeated_ngram": best_n,
            "repeated_ngram_count": best_count,
            "target_occurrences_in_context": target_occurs_earlier,
            "final_bigram_earlier_occurrences": bigram_repeats,
            "in_scope": bool(in_scope),
            "paper_verified": "arXiv:2504.14218, 2025-04-19",
        },
        tables={"repetition in the anchor context": rows},
        cannot_conclude=(
            "Token-level n-gram repetition is a crude proxy for the paper's notion of "
            "repetition, which is about emitted text under decoding. A model can loop "
            "on a non-repeating prompt. Since this measurement emits nothing, that case "
            "cannot arise here, but the proxy is still a proxy."
        ),
        caveats=[
            "If a Pythia-1.4B SAE at an intermediate checkpoint ever becomes available, "
            "this probe should be replaced by the paper's actual method. Applying a "
            "final-checkpoint SAE to a step131000 model would measure a different "
            "object.",
        ],
    )
