"""A bundle with the real anchor numbers planted in it, for running without a GPU.

This exists so the probe suite can be developed, tested and demonstrated end to end on
a laptop.  The anchor's measured values -- target probability and rank per checkpoint,
the top-15 ranking, the attention argmax per layer, the sink masses -- are the ones the
earlier runs actually reported, so a probe that fires here would fire on the real
bundle.  Everything else (controls, variants, context NLL, per-head weights) is drawn
from a seeded RNG that reproduces the *shape* of the real distributions and none of
their content.

Consequences, which the bundle carries in its own metadata so no report can hide them:

- Control-referenced p-values are p-values against invented controls.  They say whether
  a probe's plumbing works, not whether a hypothesis holds.
- The variants are constructed to match the anchor, because that is what was measured.
  A probe that tests variant-equality will therefore pass here by construction.

Use it for ``--smoke``.  Never for a claim.
"""

from __future__ import annotations

import numpy as np

from . import config as C
from .measure import Bundle

_MEANINGLESS = (
    "SYNTHETIC BUNDLE. The anchor's headline numbers are the real measured ones; the "
    "40 controls, the context NLL and all per-head weights are random. No number "
    "produced from this bundle is evidence about Pythia-1.4B."
)


def fake_bundle(seed: int = 0, top_k: int = 50) -> Bundle:
    rng = np.random.default_rng(seed)

    names = [C.ANCHOR]
    kinds = ["state_haupt"]
    # the three further main states named in the source tables, then the rest
    for nm in ("state_41909", "state_51158", "state_37508"):
        names.append(nm)
        kinds.append("state_haupt")
    for i in range(C.N_STATES - 4):
        names.append(f"state_extra_{i:02d}")
        kinds.append("state_extra")
    for nm in C.VARIANT_NAMES:
        names.append(nm)
        kinds.append("variante")
    for i in range(C.N_CONTROLS):
        names.append(f"kontrolle_{i:02d}")
        kinds.append("kontrolle")

    n_c, n_p, T, L, H = len(C.CHECKPOINTS), len(names), C.ANCHOR_T, C.N_LAYERS, C.N_HEADS
    idx = {n: i for i, n in enumerate(names)}

    topk_ids = np.zeros((n_c, n_p, top_k), dtype=np.int32)
    topk_probs = np.zeros((n_c, n_p, top_k), dtype=np.float32)
    target_p = np.zeros((n_c, n_p), dtype=np.float32)
    target_rank = np.zeros((n_c, n_p), dtype=np.int32)
    target_rank_in_topk = np.zeros((n_c, n_p), dtype=np.int32)
    top1_ids = np.zeros((n_c, n_p), dtype=np.int32)
    top1_probs = np.zeros((n_c, n_p), dtype=np.float32)
    entropy = np.zeros((n_c, n_p), dtype=np.float32)
    target_ids = np.full(n_p, C.TARGET_TOKEN_ID, dtype=np.int32)

    # controls and other states get their own targets
    for nm in names:
        if not (nm == C.ANCHOR or kinds[idx[nm]] == "variante"):
            target_ids[idx[nm]] = int(rng.integers(100, 40_000))

    def fill_from_reference(p_i: int, c_i: int, step: int, jitter: float = 0.0) -> None:
        ref = C.REFERENCE_TOP15[step]
        ids = [t for t, _ in ref]
        ps = np.array([p for _, p in ref], dtype=np.float64)
        if jitter:
            ps = np.clip(ps * (1.0 + rng.normal(0, jitter, ps.shape)), 1e-9, None)
        # pad the top-k tail with a decaying skirt
        tail_n = top_k - len(ids)
        tail_ids = rng.choice(np.arange(1000, 45_000), size=tail_n, replace=False)
        tail_ps = ps[-1] * np.exp(-0.25 * np.arange(1, tail_n + 1))
        all_ids = np.concatenate([np.array(ids), tail_ids]).astype(np.int32)
        all_ps = np.concatenate([ps, tail_ps]).astype(np.float32)
        order = np.argsort(-all_ps)
        topk_ids[c_i, p_i] = all_ids[order]
        topk_probs[c_i, p_i] = all_ps[order]
        tgt = int(target_ids[p_i])
        where = np.flatnonzero(topk_ids[c_i, p_i] == tgt)
        target_rank_in_topk[c_i, p_i] = int(where[0]) + 1 if where.size else 0
        target_p[c_i, p_i] = float(all_ps[all_ids == tgt][0]) if (all_ids == tgt).any() else 1e-6
        target_rank[c_i, p_i] = int((all_ps > target_p[c_i, p_i]).sum()) + 1
        top1_ids[c_i, p_i] = topk_ids[c_i, p_i][0]
        top1_probs[c_i, p_i] = topk_probs[c_i, p_i][0]
        q = np.clip(topk_probs[c_i, p_i].astype(np.float64), 1e-12, None)
        entropy[c_i, p_i] = float(-(q * np.log(q)).sum())

    for c_i, step in enumerate(C.CHECKPOINTS):
        fill_from_reference(idx[C.ANCHOR], c_i, step)
        # the never-trained variants measured the same distribution as the anchor
        for nm in C.VARIANT_NAMES:
            fill_from_reference(idx[nm], c_i, step, jitter=0.004)
        for nm in names:
            if nm == C.ANCHOR or kinds[idx[nm]] == "variante":
                continue
            fill_from_reference(idx[nm], c_i, step, jitter=0.35)

    # --- attention: layers 0-2 local, 3+ locked on the newline sink at position 46 ----
    attn_last = np.zeros((n_c, n_p, L, T), dtype=np.float32)
    attn_received = np.zeros((n_c, n_p, L, T), dtype=np.float32)
    attn_heads = np.zeros((n_c, 1 + C.N_VARIANTS, L, H, T), dtype=np.float32)
    fam = [C.ANCHOR] + list(C.VARIANT_NAMES)

    # One base profile per passage, then a small per-checkpoint perturbation. The
    # perturbation scale is chosen so that the across-boundary total-variation distances
    # land near the measured ones -- anchor ~0.017, control median ~0.033 -- because the
    # sink_stability probe reads exactly those numbers and a bundle that got them wildly
    # wrong would exercise the probe without testing it.
    ANCHOR_JITTER, CONTROL_JITTER = 0.0385, 0.077

    bases = np.zeros((n_p, L, T), dtype=np.float32)
    for p_i in range(n_p):
        base = rng.random((L, T)).astype(np.float32) * 0.002
        for layer in range(L):
            ref = C.REFERENCE_ATTN_ARGMAX_131000.get(layer)
            if ref is None:  # layers 20-23: the source table was truncated
                pos, w = C.SINK_POSITION_NEWLINE, 0.55
            else:
                pos, w = ref
            base[layer, pos] += w
            # The first token is a sink too, but it must not outrank the layer's own
            # primary target -- in the measured profile layers 0-2 still point locally
            # at " km"/" 74". Cap it below the primary weight.
            base[layer, C.SINK_POSITION_FIRST] += min(0.19, 0.8 * w)
            base[layer] /= base[layer].sum()
        bases[p_i] = base

    for c_i in range(n_c):
        for p_i in range(n_p):
            quiet = names[p_i] == C.ANCHOR or kinds[p_i] == "variante"
            scale = ANCHOR_JITTER if quiet else CONTROL_JITTER
            prof = bases[p_i] * (1.0 + rng.normal(0, scale, (L, T)).astype(np.float32))
            prof = np.clip(prof, 1e-9, None)
            prof /= prof.sum(axis=1, keepdims=True)
            attn_last[c_i, p_i] = prof

            recv = prof.copy()
            recv[:, C.SINK_POSITION_FIRST] += 0.12
            recv /= recv.sum(axis=1, keepdims=True)
            attn_received[c_i, p_i] = recv

            if names[p_i] in fam:
                f = fam.index(names[p_i])
                heads = np.repeat(prof[:, None, :], H, axis=1)
                heads = heads * (1.0 + rng.normal(0, 0.15, heads.shape).astype(np.float32))
                heads = np.clip(heads, 1e-8, None)
                heads /= heads.sum(axis=2, keepdims=True)
                attn_heads[c_i, f] = heads

    ctx_nll = (rng.gamma(2.0, 1.3, size=(n_c, n_p, T - 1))).astype(np.float32)
    ctx_ids = rng.integers(1, 40_000, size=(n_p, T)).astype(np.int32)
    ctx_ids[idx[C.ANCHOR], list(C.NAME_POSITIONS)] = C.NAME_TOKEN_IDS

    vocab = {
        "591": " per", "545": "ph", "16": "/", "271": " an", "73": "h", "313": " (",
        "1227": " /", "15": ".", "793": "ps", "275": " in", "247": " a", "14": "-",
        "468": "per", "6285": "hr", "288": " h", "446": "pl", "2617": "pm",
        "3044": " Mc", "3864": "Qu", "274": "ar", "6595": "rie",
    }
    for tid in np.unique(topk_ids):
        vocab.setdefault(str(int(tid)), f"<{int(tid)}>")
    for tid in np.unique(target_ids):
        vocab.setdefault(str(int(tid)), f"<{int(tid)}>")

    ctx = ["<tok>"] * T
    ctx[C.SINK_POSITION_NEWLINE] = "\n"
    ctx[C.SINK_POSITION_FIRST] = " st"
    for pos, tid in zip(C.NAME_POSITIONS, C.NAME_TOKEN_IDS):
        ctx[pos] = vocab[str(tid)]
    for pos, tok in ((206, " km"), (205, " 74"), (204, " approximately"),
                     (203, " reach"), (202, " could"), (201, " speeds"), (200, " wind")):
        ctx[pos] = tok

    return Bundle(
        checkpoints=tuple(C.CHECKPOINTS),
        names=tuple(names),
        kinds=tuple(kinds),
        topk_ids=topk_ids,
        topk_probs=topk_probs,
        topk_logprobs=np.log(np.clip(topk_probs, 1e-12, None)).astype(np.float32),
        target_ids=target_ids,
        target_p=target_p,
        target_logp=np.log(np.clip(target_p, 1e-12, None)).astype(np.float32),
        target_rank=target_rank,
        target_rank_in_topk=target_rank_in_topk,
        entropy=entropy,
        top1_ids=top1_ids,
        top1_probs=top1_probs,
        final_logits=np.zeros((0, 0, 0), dtype=np.float32),
        ctx_nll=ctx_nll,
        ctx_ids=ctx_ids,
        attn_last=attn_last,
        attn_received=attn_received,
        attn_heads=attn_heads,
        attn_head_names=tuple(fam),
        meta={
            "model_id": "SYNTHETIC",
            "dtype": "float32",
            "top_k": top_k,
            "T": T,
            "provenance": "synthetic",
            "passage_source": f"synthetic_bundle(seed={seed})",
            "canonical_sha256": "",
            "canonical_match": False,
            "warning": _MEANINGLESS,
            "run_id": f"smoke_{seed}",
            "seed": seed,
            "elapsed_s": 0.0,
            "vocab": vocab,
            "attention_captured": True,
            "full_logits_captured": False,
        },
        context_tokens={nm: list(ctx) for nm in fam},
        aux={**_fake_aux(rng, names, kinds, idx, target_p, T),
             **_fake_norm_attn(rng, names, T)},
    )


def _fake_aux(rng, names, kinds, idx, target_p, T) -> dict:
    """Auxiliary measurements shaped like the real ones.

    The anchor's ladder is built to be *flat* -- p(target) reaching its full-context
    value from the last two tokens alone -- because that is the hypothesis the real run
    is most likely to confirm and the one the suite must be able to detect. It is a
    guess about the real data, not a measurement of it, which is exactly why this
    bundle may never be used for a claim.
    """
    from . import config as C

    ladder_k = list(__import__("m60482.aux", fromlist=["LADDER"]).LADDER)
    n_c, n_p = len(C.CHECKPOINTS), len(names)
    arr = np.zeros((n_c, n_p, len(ladder_k)), dtype=np.float32)
    for c_i in range(n_c):
        for p_i in range(n_p):
            full = float(target_p[c_i, p_i])
            # a short shoulder at k=1, flat from k=2 onward
            curve = np.full(len(ladder_k), full, dtype=np.float32)
            curve[0] = full * 0.55
            curve *= (1.0 + rng.normal(0, 0.02, curve.shape)).astype(np.float32)
            arr[c_i, p_i] = np.clip(curve, 1e-8, None)

    fam = [C.ANCHOR] + list(C.VARIANT_NAMES)
    greedy = {
        str(step): {nm: [545, 73, 15, 187] for nm in fam}   # " ph", "h", ".", " hours"
        for step in C.CHECKPOINTS
    }
    name_sub = {
        str(step): {
            "baseline": float(target_p[c_i, idx[C.ANCHOR]]),
            "substitutions": [
                {"substitution": i, "tokens": [0, 0, 0, 0],
                 "p_target": float(target_p[c_i, idx[C.ANCHOR]]) * (1 + 0.01 * i),
                 "delta": 0.0, "relative": 0.0}
                for i in range(3)
            ],
        }
        for c_i, step in enumerate(C.CHECKPOINTS)
    }
    return {
        # fp32 with TF32 off: batch-order effects land around 1e-5 on a probability,
        # which is below the 0.00122 margin the rank-2 claim rests on. A guess about
        # the real measurement, and labelled as one.
        "jitter_band": 2.0e-5,
        "jitter_step": C.EXPOSURE_BOUNDARY[0],
        "jitter_n": 0,
        "jitter_note": "SYNTHETIC: not a measurement.",
        "ladder": arr,
        "ladder_k": ladder_k,
        "ladder_names": list(names),
        "greedy": greedy,
        "greedy_tokens": 4,
        "name_substitution": name_sub,
    }


def _fake_norm_attn(rng, names, T) -> dict:
    """A norm-weighted profile in which the sink evaporates.

    Drained value vectors at the sink positions are the mechanism that makes a sink a
    no-op, so this is the expected shape -- but it is a guess about the real data, not a
    measurement of it.
    """
    from . import config as C

    steps = list(C.EXPOSURE_BOUNDARY)
    arr = np.zeros((len(steps), len(names), C.N_LAYERS, T), dtype=np.float32)
    for s_i in range(len(steps)):
        for p_i in range(len(names)):
            prof = rng.random((C.N_LAYERS, T)).astype(np.float32) * 0.01 + 0.004
            # content positions near the end carry the contribution
            for pos, w in ((206, 0.22), (205, 0.15), (201, 0.08), (203, 0.06)):
                prof[:, pos] += w
            # the sinks keep a small residue -- drained, not absent
            prof[:, C.SINK_POSITION_NEWLINE] += 0.03
            prof[:, C.SINK_POSITION_FIRST] += 0.02
            prof /= prof.sum(axis=1, keepdims=True)
            arr[s_i, p_i] = prof
    return {
        "norm_attn": arr,
        "norm_attn_steps": steps,
        "norm_attn_names": list(names),
    }
