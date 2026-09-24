"""Frozen constants for the state-60482 study.

Every number here comes from the existing measurement runs (K8 NLL multi-checkpoint,
the token-level analysis, and the attention/ranking run of 2026-09-24).  They are
recorded so that probes can *assert* against them instead of silently re-deriving a
different value.  If a fresh run disagrees with one of these, the run is wrong (or the
model variant is wrong) -- that is the point of keeping them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------

MODEL_ID = "EleutherAI/pythia-1.4b"

#: Tokenizer revision the passage set was built with (step98000 snapshot).
TOKENIZER_REVISION = "step98000"
TOKENIZER_SHA256 = "c24618a1b3e6a38167beff1c72cffd126c3a66254347304b50547d12c5f25624"

#: The four checkpoints of the run.  Order matters: boundaries are consecutive pairs.
CHECKPOINTS: tuple[int, ...] = (130000, 131000, 132000, 133000)

#: (from, to) pairs.
BOUNDARIES: tuple[tuple[int, int], ...] = (
    (130000, 131000),
    (131000, 132000),
    (132000, 133000),
)

#: The boundary that straddles the anchor's training step.
EXPOSURE_BOUNDARY: tuple[int, int] = (131000, 132000)

#: The two boundaries with no exposure.  They define the noise floor.
PLACEBO_BOUNDARIES: tuple[tuple[int, int], ...] = (
    (130000, 131000),
    (132000, 133000),
)

# --------------------------------------------------------------------------------------
# The anchor
# --------------------------------------------------------------------------------------

ANCHOR = "state_60482"

#: Context length in tokens, excluding the held-out target.
ANCHOR_T = 207

#: The held-out target token.
TARGET_TOKEN_ID = 591
TARGET_TOKEN_STR = " per"

#: Positions of " Mc", "Qu", "ar", "rie" in the anchor context.  Position 123 is
#: " Christopher" and is deliberately NOT included.
NAME_POSITIONS: tuple[int, ...] = (124, 125, 126, 127)
NAME_TOKEN_IDS: tuple[int, ...] = (3044, 3864, 274, 6595)

#: global_sample_index // 1024.  Falls between step131000 and step132000, which is what
#: makes 131000 -> 132000 the exposure boundary.
ANCHOR_GLOBAL_SAMPLE_INDEX = 134428942
ANCHOR_TRAIN_STEP = ANCHOR_GLOBAL_SAMPLE_INDEX // 1024  # 131278

#: Positions that the earlier attention run found to dominate.  Probes that claim to
#: measure "where the model looks" must reproduce these or explain why they do not.
SINK_POSITION_NEWLINE = 46
SINK_POSITION_FIRST = 0

# --------------------------------------------------------------------------------------
# Passage set
# --------------------------------------------------------------------------------------

N_STATES = 19          # state_haupt + state_extra, anchor included
N_CONTROLS = 40        # kontrolle_00 .. kontrolle_39
N_VARIANTS = 10        # never-trained misspellings of the name
N_PASSAGES = N_STATES + N_CONTROLS + N_VARIANTS  # 69

PASSAGE_KINDS = ("state_haupt", "state_extra", "variante", "kontrolle")

#: The anchor's 207-token context is literally ``row[:207]`` of training sample
#: 134428942, and its target is ``row[207]``.  So the anchor family can be rebuilt
#: exactly from a 4098-byte range request -- no Drive, no 602 GB corpus download.
#: The 40 controls still need the seed-42 draw from pile-uncopyrighted.
ANCHOR_IS_PILE_ROW_PREFIX = True

#: sha256 over sorted (name, art, T, ids, ziel) of the canonical K8 passage set.
CANONICAL_PASSAGES_SHA256 = (
    "960bb0c0c54c9c4ce92e2b132570246724280c5953183b490f705cf83cd4d2fd"
)
#: sha256 of the passagen.json file itself as written by the K8 run.
PASSAGES_FILE_SHA256 = (
    "6cbfb16a9e75f881716251735f473a13a9ad43de363c61dc28ffe62976e96669"
)

#: Controls were drawn with this seed from the head of pile-uncopyrighted train/00.
CONTROL_SEED = 42
CONTROL_SOURCE = "monology/pile-uncopyrighted::train/00.jsonl.zst"

VARIANT_NAMES: tuple[str, ...] = (
    "variante_Mcquarrie",
    "variante_mcquarrie",
    "variante_McQUarrie",
    "variante_McQuarr1e",
    "variante_McOuarrie",
    "variante_Mc0uarrie",
    "variante_MCQUARRIE",
    "variante_mcQuarrie",
    "variante_McXuarrie",
    "variante_McQuarrxe",
)

# --------------------------------------------------------------------------------------
# Reference results -- what the previous runs measured
# --------------------------------------------------------------------------------------

#: p(" per" | anchor context) per checkpoint.
REFERENCE_TARGET_P: dict[int, float] = {
    130000: 0.09402,
    131000: 0.09087,
    132000: 0.10461,
    133000: 0.10833,
}

#: Rank of " per" in the full vocabulary per checkpoint (1-based).
REFERENCE_TARGET_RANK: dict[int, int] = {
    130000: 3,
    131000: 2,
    132000: 3,
    133000: 3,
}

#: p(" ph" | anchor context) -- the token the model actually wants ("kmph").
REFERENCE_TOP1_P: dict[int, float] = {
    130000: 0.72602,
    131000: 0.79824,
    132000: 0.74426,
    133000: 0.63780,
}
TOP1_TOKEN_ID = 545
TOP1_TOKEN_STR = "ph"

#: p("/" ) -- the competitor whose swing, not the target's, produced the rank 2->3 flip.
REFERENCE_COMPETITOR_P: dict[int, float] = {
    130000: 0.14681,
    131000: 0.08965,
    132000: 0.11662,
    133000: 0.21189,
}
COMPETITOR_TOKEN_ID = 16
COMPETITOR_TOKEN_STR = "/"

#: Attention-shift result: anchor total-variation distance vs the 40 controls.
#: boundary -> (tv_anchor, control_median, k_controls_above_anchor, n, p)
REFERENCE_ATTENTION_SHIFT: dict[tuple[int, int], tuple[float, float, int, int, float]] = {
    (130000, 131000): (0.01727, 0.03130, 40, 40, 1.000),
    (131000, 132000): (0.01723, 0.03282, 40, 40, 1.000),
    (132000, 133000): (0.02818, 0.03141, 31, 40, 0.78049),
}

#: Tolerance for "did this run reproduce the earlier measurement".
REPRODUCTION_TOLERANCE = 1e-4

# --------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------

#: Rank-based p against 40 controls: (k + 1) / (n + 1).  The floor is 1/41.
MIN_ATTAINABLE_P = 1.0 / (N_CONTROLS + 1)

#: Number of boundaries examined.  Any per-boundary claim must be read against this.
N_BOUNDARIES = len(BOUNDARIES)


@dataclass(frozen=True)
class RunConfig:
    """Everything a run needs that is not a frozen fact."""

    model_id: str = MODEL_ID
    checkpoints: tuple[int, ...] = CHECKPOINTS
    dtype: str = "float32"          # see model.py -- fp16 is not safe here
    device: str = "cuda"
    top_k: int = 50
    capture_attention: bool = True
    capture_heads_for: tuple[str, ...] = ()   # filled by the driver with the anchor family
    output_dir: str = "runs"
    run_id: str = ""
    seed: int = 0
    probes: tuple[str, ...] = ()    # empty = all registered probes
    degraded: bool = False          # True when the passage set is not the canonical one
    notes: str = ""
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------------------
# Architecture (verified against the HuggingFace config for EleutherAI/pythia-1.4b)
# --------------------------------------------------------------------------------------

N_LAYERS = 24
N_HEADS = 16
HIDDEN_SIZE = 2048
CONFIG_VOCAB_SIZE = 50304   # padded; the tokenizer's real vocabulary is smaller

#: RESOLVED, by fetching sample 134428942 from both preshuffled corpora and checking it
#: against the frozen anchor (see :mod:`m60482.pile`).  The standard corpus row carries
#: the name ids at 124-127, ``" per"`` at 207, ``"\n"`` at 46 and ``" st"`` at 0; the
#: deduped corpus at the same index is an unrelated document about gym instructors.
#:
#: The architecture cannot settle this -- the two configs are byte-identical -- and
#: neither can the tokenizer, whose ``tokenizer.json`` has the same sha256 on both
#: repos and at every revision.  Only the data order discriminates.
MODEL_VARIANT = "pythia-1.4b"        # NOT -deduped
MODEL_VARIANT_VERIFIED_BY = (
    "m60482.pile.check_anchor() passes on EleutherAI/pile-standard-pythia-preshuffled "
    "row 134428942 and fails on EleutherAI/pile-deduped-pythia-preshuffled at the same "
    "index. Re-run m60482.pile.identify_variant() to reproduce; it costs ~8 KB."
)

# --------------------------------------------------------------------------------------
# Reference top-15 continuations for the anchor, per checkpoint.
# (token_id, probability), rank order.  Used to check a fresh run reproduces the ranking,
# not merely the target's own probability.  Truncated at 15 as in the source tables.
# --------------------------------------------------------------------------------------

REFERENCE_TOP15: dict[int, tuple[tuple[int, float], ...]] = {
    130000: ((545, 0.72602), (16, 0.14681), (591, 0.09402), (271, 0.01029), (73, 0.00384),
             (313, 0.00326), (1227, 0.00191), (15, 0.00165), (793, 0.00118), (275, 0.00102),
             (247, 0.00087), (14, 0.00059), (468, 0.00045), (6285, 0.00031), (288, 0.00028)),
    131000: ((545, 0.79824), (591, 0.09087), (16, 0.08965), (271, 0.00422), (73, 0.00347),
             (313, 0.00236), (1227, 0.00115), (15, 0.00102), (468, 0.00058), (275, 0.00054),
             (793, 0.00054), (14, 0.00042), (247, 0.00039), (6285, 0.00030), (446, 0.00027)),
    132000: ((545, 0.74426), (16, 0.11662), (591, 0.10461), (271, 0.00949), (73, 0.00443),
             (313, 0.00421), (1227, 0.00181), (15, 0.00168), (275, 0.00095), (247, 0.00062),
             (14, 0.00062), (468, 0.00056), (793, 0.00055), (6285, 0.00053), (2617, 0.00037)),
    133000: ((545, 0.63780), (16, 0.21189), (591, 0.10833), (271, 0.01224), (313, 0.00439),
             (73, 0.00432), (1227, 0.00361), (15, 0.00315), (14, 0.00092), (275, 0.00081),
             (247, 0.00062), (468, 0.00053), (288, 0.00049), (793, 0.00039), (2617, 0.00037)),
}

#: Head-averaged attention of the anchor's final query: (argmax position, its weight) per
#: layer, at step131000.  Recorded for layers 0-19; the source table was cut off there.
#: Layers 3+ all point at position 46, a newline 160 tokens before the end.
REFERENCE_ATTN_ARGMAX_131000: dict[int, tuple[int, float]] = {
    0: (206, 0.05838), 1: (205, 0.11828), 2: (206, 0.12001), 3: (46, 0.16686),
    4: (46, 0.35485), 5: (46, 0.38859), 6: (46, 0.38638), 7: (46, 0.45974),
    8: (46, 0.32046), 9: (46, 0.38515), 10: (46, 0.48689), 11: (46, 0.48985),
    12: (46, 0.52372), 13: (46, 0.46232), 14: (46, 0.51652), 15: (46, 0.50006),
    16: (46, 0.59455), 17: (46, 0.69279), 18: (46, 0.59486), 19: (46, 0.56126),
}

#: Layer-averaged attention mass on the dominant positions, anchor, step131000/132000.
REFERENCE_SINK_MASS = {
    "position_46_newline": {131000: 0.440833, 132000: 0.439014},
    "position_0_first_last_query": {131000: 0.194255, 132000: 0.197970},
    "position_0_first_received": {131000: 0.309808, 132000: 0.313050},
    "position_206_km_last_query": {131000: 0.068289, 132000: 0.067357},
}
