"""Auxiliary measurements: the things one forward pass per passage cannot answer.

Three questions drove this module, and each needs its own passes:

**How much context does the prediction actually use?**
    The truncation ladder re-runs each passage with only its last *k* tokens.  If
    p(" per") is already at its full-context value at k=2 -- that is, from " 74 km"
    alone -- then nothing about this passage, its name, or its training exposure is
    doing the work, and the whole framing collapses into a bigram continuation prior.

**Would the model actually reproduce the training text?**
    Every published definition of verbatim memorization requires the greedy
    continuation to match the training continuation.  Generating it is the entry
    criterion for the memorization frame, and it is four short generations.

**Is the name load-bearing?**
    Substituting the name tokens in place measures the causal contribution of the
    span the memorization story depends on, without needing a separate passage set.

All of these are cheap (tens of extra passes), and all of them can kill an expensive
hypothesis before it is tested.  They run inside the same checkpoint context as the
main sweep, so the four checkpoint loads are not paid twice.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from . import config as C
from . import model as M
from .passages import Passage, PassageSet

#: Context lengths for the truncation ladder, counted back from the end.
LADDER: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 207)

#: How many tokens to generate greedily when testing the memorization entry criterion.
GREEDY_TOKENS = 24


def _target_prob(ckpt, ids: Sequence[int], target_id: int) -> float:
    import torch

    logits, _, _, _ = M.forward_pass_all(ckpt, ids, need_attention=False)
    return float(torch.softmax(logits, dim=-1)[target_id].item())


def truncation_ladder(
    ckpt,
    passages: Sequence[Passage],
    ladder: Sequence[int] = LADDER,
) -> np.ndarray:
    """``[passage, len(ladder)]`` -- p(target) with only the last k context tokens.

    The point of comparison is the last column (full context).  A flat ladder means
    the model never needed the passage.
    """
    out = np.zeros((len(passages), len(ladder)), dtype=np.float32)
    for p_i, passage in enumerate(passages):
        for k_i, k in enumerate(ladder):
            k = min(k, passage.T)
            out[p_i, k_i] = _target_prob(ckpt, passage.ids[-k:], passage.target)
    return out


def greedy_continuation(ckpt, ids: Sequence[int], n: int = GREEDY_TOKENS) -> list[int]:
    """Greedy (argmax) continuation, the decoding rule every verbatim-memorization
    definition is stated over."""
    import torch

    cur = list(ids)
    out: list[int] = []
    for _ in range(n):
        logits, _, _, _ = M.forward_pass_all(ckpt, cur, need_attention=False)
        nxt = int(torch.argmax(logits).item())
        out.append(nxt)
        cur.append(nxt)
    return out


def name_substitution(
    ckpt,
    anchor: Passage,
    *,
    positions: Sequence[int] = C.NAME_POSITIONS,
    replacements: Sequence[Sequence[int]] = (),
    rng_seed: int = 0,
) -> dict:
    """p(target) with the name span replaced in place.

    ``replacements`` defaults to three substitutions with different characters: a
    repeated filler token, a random token draw, and the span deleted by replacing it
    with the token that precedes it.  The name variants in the passage set already test
    *misspellings*; this tests removal.
    """
    rng = np.random.default_rng(rng_seed)
    base = _target_prob(ckpt, anchor.ids, anchor.target)

    if not replacements:
        filler = anchor.ids[0]
        replacements = [
            [filler] * len(positions),
            [int(x) for x in rng.integers(100, 40_000, size=len(positions))],
            [anchor.ids[positions[0] - 1]] * len(positions),
        ]

    rows = []
    for r_i, repl in enumerate(replacements):
        ids = list(anchor.ids)
        for pos, tok in zip(positions, repl):
            ids[pos] = int(tok)
        p = _target_prob(ckpt, ids, anchor.target)
        rows.append(
            {
                "substitution": r_i,
                "tokens": [int(t) for t in repl],
                "p_target": p,
                "delta": p - base,
                "relative": (p - base) / base if base > 0 else float("nan"),
            }
        )
    return {"baseline": base, "substitutions": rows}


def measure_aux(
    passage_set: PassageSet,
    cfg: C.RunConfig,
    *,
    ladder: Sequence[int] = LADDER,
    greedy_tokens: int = GREEDY_TOKENS,
    progress: bool = True,
) -> dict:
    """Run every auxiliary measurement across the checkpoints.

    Returns the ``aux`` dict a :class:`~m60482.measure.Bundle` carries.  Keys holding
    ``np.ndarray`` are stored as arrays; everything else is JSON.
    """
    passages = passage_set.passages
    names = [p.name for p in passages]
    n_c = len(cfg.checkpoints)

    ladder_arr = np.zeros((n_c, len(passages), len(ladder)), dtype=np.float32)
    greedy: dict[str, dict[str, list[int]]] = {}
    name_sub: dict[str, dict] = {}

    for c_i, step in enumerate(cfg.checkpoints):
        if progress:
            print(f"[aux {c_i + 1}/{n_c}] step{step}", flush=True)
        with M.checkpoint(
            step,
            model_id=cfg.model_id,
            dtype=cfg.dtype,
            device=cfg.device,
            need_attention=False,
        ) as ckpt:
            ladder_arr[c_i] = truncation_ladder(ckpt, passages, ladder)

            fam = passage_set.anchor_family
            greedy[str(step)] = {
                p.name: greedy_continuation(ckpt, p.ids, greedy_tokens) for p in fam
            }
            name_sub[str(step)] = name_substitution(ckpt, passage_set.anchor)

    return {
        "ladder": ladder_arr,
        "ladder_k": list(ladder),
        "ladder_names": names,
        "greedy": greedy,
        "greedy_tokens": greedy_tokens,
        "name_substitution": name_sub,
    }


def untrained_attention(
    passage_set: PassageSet,
    cfg: C.RunConfig,
    *,
    seed: int = 0,
    progress: bool = True,
) -> dict:
    """Head-averaged final-query attention under a randomly-initialised model.

    One sweep, no checkpoint download.  Gives the position-bias probe the comparison it
    actually needs: is the profile there *before* training?
    """
    passages = passage_set.passages
    T = passages[0].T
    out = None

    if progress:
        print(f"[aux] untrained baseline (seed={seed})", flush=True)

    ckpt = M.load_untrained(
        model_id=cfg.model_id, dtype=cfg.dtype, device=cfg.device,
        need_attention=True, seed=seed,
    )
    try:
        for p_i, passage in enumerate(passages):
            _, _, last_q, _ = M.forward_pass_all(ckpt, passage.ids, need_attention=True)
            if out is None:
                out = np.zeros((len(passages), last_q.shape[0], T), dtype=np.float32)
            out[p_i] = last_q.mean(dim=1).numpy()
    finally:
        ckpt.free()

    return {
        "untrained_attn_last": out,
        "untrained_names": [p.name for p in passages],
        "untrained_seed": seed,
    }
