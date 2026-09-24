"""The candidate explanations, and where each one came from.

Six papers were handed to the user as candidates.  All six were located and confirmed
to exist; the table below records the verified citation next to the relayed one,
because the list arrived from an assistant that had been given an unreadable PDF and
was reconstructing from a verbal description.  Two things follow from that provenance
and are recorded here rather than in prose: the *reason* each paper was surfaced, and
whether that reason survives contact with the measurement.

Papers not on the user's list are marked ``on_user_list=False``.  They are here because
the measured facts point at them -- most of all the attention sink, which accounts for
the single largest number in the whole dataset (0.44 of the final query's attention on
one newline) and which no paper on the list is about.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Paper:
    key: str
    title: str
    citation: str
    date: str
    on_user_list: bool
    surfaced_because: str
    probes: tuple[str, ...]
    #: "explanation" -- a candidate account of the effect, rolled into the headline.
    #: "premise" -- a check on whether there is an effect to explain at all. These are
    #: reported separately, because "the effect clears the noise floor" is a
    #: precondition, not a mechanism, and lumping the two makes the headline nonsense.
    kind: str = "explanation"


PAPERS: tuple[Paper, ...] = (
    # --- the six the user was given ------------------------------------------------
    Paper(
        key="cliff_tokens",
        title="Cliff Tokens: Identifying Single-Token Failure Triggers in LLM Mathematical Reasoning",
        citation="arXiv:2606.25524",
        date="2026-06-24",
        on_user_list=True,
        surfaced_because=(
            "A single token position carries a large probability effect. The surface "
            "analogy is real; the paper's unit of analysis is a generated reasoning "
            "trace with a verifiable answer."
        ),
        probes=("cliff_token_scope",),
    ),
    Paper(
        key="self_loops",
        title="Can We Break LLMs Out of Self-Loops? Fine-Grained Reasoning Control with Activation Steering (SOPHIA)",
        citation="arXiv:2607.18100",
        date="2026-07-20",
        on_user_list=True,
        surfaced_because=(
            "A homonym. SOPHIA's 'state' is a k-means cluster over reasoning-step "
            "activations; 'state_60482' is a passage label in the K8 schema."
        ),
        probes=("self_loop_scope",),
    ),
    Paper(
        key="repeat_curse",
        title="Understanding the Repeat Curse in Large Language Models from a Feature Perspective",
        citation="arXiv:2504.14218",
        date="2025-04-19",
        on_user_list=True,
        surfaced_because="Repetition was one of the hypotheses under consideration.",
        probes=("repeat_curse_scope",),
    ),
    Paper(
        key="verbatim_circuits",
        title="Understanding Verbatim Memorization in LLMs Through Circuit Discovery",
        citation="arXiv:2506.21588",
        date="2025-06-17 (rev. 2026-02-03)",
        on_user_list=True,
        surfaced_because=(
            "The counter-check for the original memorization reading -- the one paper "
            "on the list squarely about the phenomenon that was claimed."
        ),
        probes=("memorization_entry", "name_variant_equivalence"),
    ),
    Paper(
        key="focus_directions",
        title="Focus Directions Make Your Language Models Pay More Attention to Relevant Contexts",
        citation="arXiv:2503.23306",
        date="2025-03-30",
        on_user_list=True,
        surfaced_because=(
            "It goes beyond behavioural observation of lost-in-the-middle to the heads "
            "and query/key directions that route attention."
        ),
        probes=("focus_directions",),
    ),
    Paper(
        key="lost_middle_birth",
        title="Lost in the Middle at Birth: An Exact Theory of Transformer Position Bias",
        citation="2026 (see notes; verify before citing)",
        date="2026-03-10",
        on_user_list=True,
        surfaced_because=(
            "It derives position bias from architecture alone and finds it in untrained "
            "models -- i.e. before any exposure could have mattered. Of the six, this is "
            "the only one whose scope conditions the 60482 setup actually meets."
        ),
        probes=("position_bias",),
    ),
    # --- not on the list, but where the measured numbers point ------------------------
    Paper(
        key="attention_sinks",
        title="Efficient Streaming Language Models with Attention Sinks; Massive Activations in LLMs",
        citation="arXiv:2309.17453; arXiv:2402.17762",
        date="2023-09 / 2024-02",
        on_user_list=False,
        surfaced_because=(
            "The largest single number in the dataset is 0.44 of the final query's "
            "attention on one newline, with a further 0.19-0.31 on the first token. "
            "That is the attention-sink signature, and nothing on the user's list "
            "addresses it."
        ),
        probes=("attention_sink",),
    ),
    Paper(
        key="bigram_prior",
        title="(no single paper) local continuation prior / tokenizer-driven prediction",
        citation="—",
        date="—",
        on_user_list=False,
        surfaced_because=(
            "The model's argmax is ' ph' -- it is writing '74 kmph'. The top three "
            "continuations are exactly the three ways English writes the unit: 'ph', "
            "'/', ' per'. This is a corpus-frequency competition among unit "
            "conventions, and it may be the whole phenomenon."
        ),
        probes=("context_dependence",),
    ),
    Paper(
        key="null_calibration",
        title="(no paper) checkpoint noise floor and rank-statistic decomposition",
        citation="—",
        date="—",
        on_user_list=False,
        surfaced_because=(
            "Before any mechanism is proposed, the effect has to clear what happens "
            "between two checkpoints when nothing happened."
        ),
        probes=("noise_floor", "rank_attribution", "ctx_nll_structure"),
        kind="premise",
    ),
)

BY_KEY = {p.key: p for p in PAPERS}

#: probe name -> paper key
PROBE_TO_PAPER = {probe: p.key for p in PAPERS for probe in p.probes}


def on_user_list() -> tuple[Paper, ...]:
    return tuple(p for p in PAPERS if p.on_user_list)


def off_list() -> tuple[Paper, ...]:
    return tuple(p for p in PAPERS if not p.on_user_list)


def explanations() -> tuple[Paper, ...]:
    return tuple(p for p in PAPERS if p.kind == "explanation")


def premises() -> tuple[Paper, ...]:
    return tuple(p for p in PAPERS if p.kind == "premise")
