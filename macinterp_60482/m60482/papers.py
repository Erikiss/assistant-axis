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
    #: ``{probe_name: polarity}``. A probe is a measurement, and papers disagree about
    #: how it should come out, so the polarity belongs to the pairing rather than to
    #: either side. ``+1``: the paper is supported when the probe reports SUPPORTED --
    #: the probe's hypothesis is the paper's. ``-1``: the paper is supported when the
    #: probe reports REFUTED. The name-variant measurement is the clear case: the
    #: memorization-circuit account predicts the variants differ, the reconstruction
    #: account predicts they do not, and both predictions are about the same numbers.
    probes: dict
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
        probes={"cliff_token_scope": 1},
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
        probes={"self_loop_scope": 1},
    ),
    Paper(
        key="repeat_curse",
        title="Understanding the Repeat Curse in Large Language Models from a Feature Perspective",
        citation="arXiv:2504.14218",
        date="2025-04-19",
        on_user_list=True,
        surfaced_because="Repetition was one of the hypotheses under consideration.",
        probes={"repeat_curse_scope": 1},
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
        probes={"memorization_entry": 1, "name_variant_equivalence": 1},
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
        probes={"focus_directions": 1},
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
        probes={"position_bias": 1},
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
        probes={"attention_sink": 1},
    ),
    Paper(
        key="softmax_artifact",
        title="Are Emergent Abilities of Large Language Models a Mirage?",
        citation="arXiv:2304.15004",
        date="2023-04",
        on_user_list=False,
        surfaced_because=(
            "Its general point is that a discontinuous metric -- a rank, or a "
            "probability read against a moving normaliser -- can manufacture a sharp "
            "change out of a smooth one. Both of the headline statistics here are of "
            "that kind: a rank, and a probability measured while the dominant token "
            "falls from 0.798 to 0.744."
        ),
        probes={"softmax_renormalization": -1},   # REFUTED there == renormalisation == this paper
    ),
    Paper(
        key="reconstruction_not_recollection",
        title="Recite, Reconstruct, Recollect: Memorization in LMs as a Multifaceted Phenomenon",
        citation="arXiv:2406.17746",
        date="2024-06",
        on_user_list=False,
        surfaced_because=(
            "It splits memorization into recitation (of duplicated text), "
            "reconstruction (of predictable templates) and recollection (of rare "
            "single-exposure content), and the anchor is textbook reconstruction: "
            "'74 km per hour' is a unit template the corpus supplies everywhere. It is "
            "the only account that PREDICTED the never-trained-variant result rather "
            "than merely tolerating it -- if the completion is reconstructive, "
            "corrupting the name must cost nothing."
        ),
        probes={
            # the variants predicting the same thing IS reconstruction
            "name_variant_equivalence": -1,
            # a flat truncation ladder IS a predictable template
            "context_dependence": 1,
        },
    ),
    Paper(
        key="optimizer_noise",
        title="Measuring Forgetting of Memorized Training Examples",
        citation="arXiv:2207.00099",
        date="2022-07",
        on_user_list=False,
        surfaced_because=(
            "Per-example drift between two checkpoints is driven by the other ~1.02 "
            "million sequences in the window, not by the one passage of interest. That "
            "predicts a diffuse, non-localised perturbation -- which is what the "
            "token-level analysis found, and what the attention null looks like."
        ),
        probes={"ctx_nll_structure": -1},   # no localised trace == diffuse optimizer noise
    ),
    Paper(
        key="sink_frozen",
        title="When Attention Sink Emerges in Language Models: An Empirical View",
        citation="arXiv:2410.10781",
        date="2024-10",
        on_user_list=False,
        surfaced_because=(
            "It is the only account that predicts the SIGN of the strangest number in "
            "the dataset -- that the anchor shifts LESS than 40 out of 40 controls. On "
            "its account the sink saturates in the first couple of thousand steps and "
            "is thereafter frozen, so by step130000 of 143000 two checkpoints 1000 "
            "steps apart cannot encode a single exposure. The attention null is then a "
            "derived consequence of which checkpoints were chosen, not a failed "
            "measurement."
        ),
        probes={"sink_stability": 1},
    ),
    Paper(
        key="attention_not_attribution",
        title="Attention is Not Only a Weight: Analyzing Transformers with Vector Norms",
        citation="arXiv:2004.10102",
        date="2020-11",
        on_user_list=False,
        surfaced_because=(
            "The attention result is a null, and this is the reason the null cannot yet "
            "be read. Sink positions are exactly the ones whose value vectors are "
            "drained, so raw attention weight does not measure contribution and a null "
            "in it is uninformative in both directions."
        ),
        probes={"norm_attribution": 1},
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
        probes={"context_dependence": 1},
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
        probes={"noise_floor": 1, "rank_attribution": 1, "ctx_nll_structure": 1},
        kind="premise",
    ),
    Paper(
        key="multiplicity",
        title="With Little Power Come Great Responsibilities",
        citation="arXiv:2010.06595",
        date="2020-11",
        on_user_list=False,
        surfaced_because=(
            "19 states x 3 boundaries is 57 comparisons, and the anchor was chosen "
            "after looking. Under the null a state's largest change lands on its "
            "exposure boundary one time in three, so about six of the nineteen should "
            "show the anchor's pattern with nothing having happened to any of them. "
            "Nothing in the original analysis accounted for that."
        ),
        probes={"multiplicity_ledger": 1},
        kind="premise",
    ),
)

BY_KEY = {p.key: p for p in PAPERS}

#: probe name -> the paper keys that read it (a probe may serve several)
PROBE_TO_PAPERS: dict[str, tuple[str, ...]] = {}
for _p in PAPERS:
    for _probe in _p.probes:
        PROBE_TO_PAPERS[_probe] = PROBE_TO_PAPERS.get(_probe, ()) + (_p.key,)

#: back-compat: one paper per probe, the first that claims it
PROBE_TO_PAPER = {k: v[0] for k, v in PROBE_TO_PAPERS.items()}


def on_user_list() -> tuple[Paper, ...]:
    return tuple(p for p in PAPERS if p.on_user_list)


def off_list() -> tuple[Paper, ...]:
    return tuple(p for p in PAPERS if not p.on_user_list)


def explanations() -> tuple[Paper, ...]:
    return tuple(p for p in PAPERS if p.kind == "explanation")


def premises() -> tuple[Paper, ...]:
    return tuple(p for p in PAPERS if p.kind == "premise")
