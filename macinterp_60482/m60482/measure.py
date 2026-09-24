"""The measurement core: one pass over (checkpoint x passage), one Bundle out.

Every probe reads a :class:`Bundle` and nothing else.  Probes never load a model, never
run a forward pass and never touch the GPU.  That is what makes them independently
implementable, cheap to re-run, and testable on a synthetic bundle.

The bundle deliberately stores more than any single probe needs -- the previous run's
central complaint was that it had saved rank and probability but not the ranking, and
the residual stream but not the attention.  Storage here is a few tens of megabytes;
re-running four checkpoints is an hour.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from . import config as C
from . import model as M
from .passages import PassageSet


_AUX_PREFIX = "aux__"


def _aux_arrays(aux: dict) -> dict:
    return {
        _AUX_PREFIX + k: v for k, v in aux.items() if isinstance(v, np.ndarray)
    }


def _aux_json(aux: dict) -> dict:
    return {k: v for k, v in aux.items() if not isinstance(v, np.ndarray)}


def _aux_restore(z) -> dict:
    out: dict = {}
    if "aux_json" in z.files:
        out.update(json.loads(str(z["aux_json"])))
    for key in z.files:
        if key.startswith(_AUX_PREFIX):
            out[key[len(_AUX_PREFIX):]] = z[key]
    return out


@dataclass
class Bundle:
    """Everything measured, indexed consistently.

    Axis order is always ``[checkpoint, passage, ...]``.  ``index[name]`` gives the
    passage axis position for a name; ``checkpoints.index(step)`` gives the checkpoint
    axis position.  Use :meth:`ci` and :meth:`pi` rather than indexing by hand.
    """

    checkpoints: tuple[int, ...]
    names: tuple[str, ...]
    kinds: tuple[str, ...]

    # --- next-token distribution at the final position -------------------------------
    topk_ids: np.ndarray          # [ckpt, passage, K] int32
    topk_probs: np.ndarray        # [ckpt, passage, K] float32
    topk_logprobs: np.ndarray     # [ckpt, passage, K] float32

    target_ids: np.ndarray        # [passage] int32
    target_p: np.ndarray          # [ckpt, passage] float32
    target_logp: np.ndarray       # [ckpt, passage] float32
    target_rank: np.ndarray       # [ckpt, passage] int32, 1-based, full vocabulary
    target_rank_in_topk: np.ndarray  # [ckpt, passage] int32, 0 when outside top-K

    entropy: np.ndarray           # [ckpt, passage] float32, nats
    top1_ids: np.ndarray          # [ckpt, passage] int32
    top1_probs: np.ndarray        # [ckpt, passage] float32

    # --- the context itself ----------------------------------------------------------
    ctx_nll: np.ndarray           # [ckpt, passage, T-1] float32, teacher-forced
    ctx_ids: np.ndarray           # [passage, T] int32

    # --- attention -------------------------------------------------------------------
    attn_last: np.ndarray         # [ckpt, passage, layer, T] float32, head-averaged
    attn_received: np.ndarray     # [ckpt, passage, layer, T] float32, query-averaged
    attn_heads: np.ndarray        # [ckpt, family, layer, head, T] float32 (or empty)
    attn_head_names: tuple[str, ...]

    # --- auxiliary measurements ------------------------------------------------------
    #: Extra measurements that need their own forward passes -- the truncation ladder,
    #: greedy continuations, name ablations.  See :mod:`m60482.aux`.  Arrays are stored
    #: under ``aux__<key>`` in the npz; JSON-able values under ``aux_json``.
    aux: dict = field(default_factory=dict)

    # --- provenance ------------------------------------------------------------------
    meta: dict = field(default_factory=dict)
    context_tokens: dict = field(default_factory=dict)   # name -> list[str]

    # ---------------------------------------------------------------------------------

    def ci(self, step: int) -> int:
        return self.checkpoints.index(step)

    def pi(self, name: str) -> int:
        return self.names.index(name)

    def kind_of(self, name: str) -> str:
        return self.kinds[self.pi(name)]

    def names_of_kind(self, *kinds: str) -> tuple[str, ...]:
        return tuple(n for n, k in zip(self.names, self.kinds) if k in kinds)

    @property
    def control_names(self) -> tuple[str, ...]:
        return self.names_of_kind("kontrolle")

    @property
    def variant_names(self) -> tuple[str, ...]:
        return self.names_of_kind("variante")

    @property
    def state_names(self) -> tuple[str, ...]:
        return self.names_of_kind("state_haupt", "state_extra")

    @property
    def n_layers(self) -> int:
        return int(self.attn_last.shape[2])

    @property
    def trustworthy(self) -> bool:
        return bool(self.meta.get("provenance") in ("canonical", "regenerated"))

    def ranking(self, name: str, step: int, k: int = 10) -> list[dict]:
        """The top-k continuations, as a list of dicts, for display."""
        c, p = self.ci(step), self.pi(name)
        out = []
        for r in range(min(k, self.topk_ids.shape[2])):
            tid = int(self.topk_ids[c, p, r])
            out.append(
                {
                    "rank": r + 1,
                    "token_id": tid,
                    "token": self.meta.get("vocab", {}).get(str(tid), str(tid)),
                    "p": float(self.topk_probs[c, p, r]),
                    "logp": float(self.topk_logprobs[c, p, r]),
                    "is_target": tid == int(self.target_ids[p]),
                }
            )
        return out

    def attn_profile(self, name: str, step: int) -> np.ndarray:
        """``[layer, T]`` head-averaged attention of the final query."""
        return self.attn_last[self.ci(step), self.pi(name)]

    # ---------------------------------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        # np.savez_compressed appends .npz when it is missing; return the real path.
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(path.suffix + ".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            checkpoints=np.asarray(self.checkpoints),
            names=np.asarray(self.names),
            kinds=np.asarray(self.kinds),
            topk_ids=self.topk_ids,
            topk_probs=self.topk_probs,
            topk_logprobs=self.topk_logprobs,
            target_ids=self.target_ids,
            target_p=self.target_p,
            target_logp=self.target_logp,
            target_rank=self.target_rank,
            target_rank_in_topk=self.target_rank_in_topk,
            entropy=self.entropy,
            top1_ids=self.top1_ids,
            top1_probs=self.top1_probs,
            ctx_nll=self.ctx_nll,
            ctx_ids=self.ctx_ids,
            attn_last=self.attn_last,
            attn_received=self.attn_received,
            attn_heads=self.attn_heads,
            attn_head_names=np.asarray(self.attn_head_names),
            meta=np.asarray(json.dumps(self.meta)),
            context_tokens=np.asarray(json.dumps(self.context_tokens)),
            **_aux_arrays(self.aux),
            aux_json=np.asarray(json.dumps(_aux_json(self.aux))),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Bundle":
        z = np.load(path, allow_pickle=False)
        return cls(
            checkpoints=tuple(int(x) for x in z["checkpoints"]),
            names=tuple(str(x) for x in z["names"]),
            kinds=tuple(str(x) for x in z["kinds"]),
            topk_ids=z["topk_ids"],
            topk_probs=z["topk_probs"],
            topk_logprobs=z["topk_logprobs"],
            target_ids=z["target_ids"],
            target_p=z["target_p"],
            target_logp=z["target_logp"],
            target_rank=z["target_rank"],
            target_rank_in_topk=z["target_rank_in_topk"],
            entropy=z["entropy"],
            top1_ids=z["top1_ids"],
            top1_probs=z["top1_probs"],
            ctx_nll=z["ctx_nll"],
            ctx_ids=z["ctx_ids"],
            attn_last=z["attn_last"],
            attn_received=z["attn_received"],
            attn_heads=z["attn_heads"],
            attn_head_names=tuple(str(x) for x in z["attn_head_names"]),
            aux=_aux_restore(z),
            meta=json.loads(str(z["meta"])),
            context_tokens=json.loads(str(z["context_tokens"])),
        )


# --------------------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------------------


def _target_stats(logits: "np.ndarray", target_id: int, top_k: int):
    """Return (topk_ids, topk_probs, topk_logprobs, target_p, target_logp, rank, entropy)."""
    import torch

    logprobs = torch.log_softmax(logits, dim=-1)
    probs = logprobs.exp()

    vals, idx = torch.topk(probs, k=top_k)
    # 1-based full-vocabulary rank: how many tokens are strictly more probable, +1.
    tgt_p = probs[target_id]
    rank = int((probs > tgt_p).sum().item()) + 1
    entropy = float(-(probs * logprobs).sum().item())

    return (
        idx.numpy().astype(np.int32),
        vals.numpy().astype(np.float32),
        logprobs[idx].numpy().astype(np.float32),
        float(tgt_p.item()),
        float(logprobs[target_id].item()),
        rank,
        entropy,
    )


def measure(
    passage_set: PassageSet,
    cfg: C.RunConfig,
    *,
    tokenizer=None,
    progress: bool = True,
) -> Bundle:
    """Run every checkpoint over every passage and build the bundle.

    Checkpoints are loaded one at a time and freed; passages are run un-batched.  On an
    A100-40GB this is dominated by the four checkpoint downloads, not by compute.
    """
    import torch

    M.set_determinism(cfg.seed)

    passages = passage_set.passages
    names = tuple(p.name for p in passages)
    kinds = tuple(p.kind for p in passages)
    n_ckpt, n_pass = len(cfg.checkpoints), len(passages)
    K = cfg.top_k

    T_values = {p.T for p in passages}
    if len(T_values) != 1:
        raise ValueError(f"passages have differing T: {sorted(T_values)}")
    T = T_values.pop()

    family = tuple(cfg.capture_heads_for) or tuple(p.name for p in passage_set.anchor_family)
    family = tuple(n for n in family if n in names)
    fam_idx = {n: i for i, n in enumerate(family)}

    if tokenizer is None:
        tokenizer = M.load_tokenizer(cfg.model_id)
    M.verify_tokenizer(tokenizer)

    topk_ids = np.zeros((n_ckpt, n_pass, K), dtype=np.int32)
    topk_probs = np.zeros((n_ckpt, n_pass, K), dtype=np.float32)
    topk_logprobs = np.zeros((n_ckpt, n_pass, K), dtype=np.float32)
    target_p = np.zeros((n_ckpt, n_pass), dtype=np.float32)
    target_logp = np.zeros((n_ckpt, n_pass), dtype=np.float32)
    target_rank = np.zeros((n_ckpt, n_pass), dtype=np.int32)
    target_rank_in_topk = np.zeros((n_ckpt, n_pass), dtype=np.int32)
    entropy = np.zeros((n_ckpt, n_pass), dtype=np.float32)
    top1_ids = np.zeros((n_ckpt, n_pass), dtype=np.int32)
    top1_probs = np.zeros((n_ckpt, n_pass), dtype=np.float32)
    ctx_nll = np.zeros((n_ckpt, n_pass, T - 1), dtype=np.float32)
    ctx_ids = np.asarray([p.ids for p in passages], dtype=np.int32)
    target_ids = np.asarray([p.target for p in passages], dtype=np.int32)

    attn_last = np.zeros((n_ckpt, n_pass, 0, T), dtype=np.float32)
    attn_received = np.zeros((n_ckpt, n_pass, 0, T), dtype=np.float32)
    attn_heads = np.zeros((n_ckpt, len(family), 0, 0, T), dtype=np.float32)
    allocated = False

    started = time.time()
    for c, step in enumerate(cfg.checkpoints):
        t0 = time.time()
        if progress:
            print(f"[{c + 1}/{n_ckpt}] loading step{step} ...", flush=True)
        with M.checkpoint(
            step,
            model_id=cfg.model_id,
            dtype=cfg.dtype,
            device=cfg.device,
            need_attention=cfg.capture_attention,
            free_disk=bool(cfg.extra.get("free_disk")),
        ) as ckpt:
            if cfg.capture_attention and not allocated:
                attn_last = np.zeros((n_ckpt, n_pass, ckpt.n_layers, T), dtype=np.float32)
                attn_received = np.zeros((n_ckpt, n_pass, ckpt.n_layers, T), dtype=np.float32)
                attn_heads = np.zeros(
                    (n_ckpt, len(family), ckpt.n_layers, ckpt.n_heads, T), dtype=np.float32
                )
                allocated = True

            for p_i, passage in enumerate(passages):
                logits, nll, last_q, received = M.forward_pass_all(
                    ckpt, passage.ids, need_attention=cfg.capture_attention
                )
                if cfg.capture_attention:
                    attn_last[c, p_i] = last_q.mean(dim=1).numpy()
                    attn_received[c, p_i] = received.mean(dim=1).numpy()
                    if passage.name in fam_idx:
                        attn_heads[c, fam_idx[passage.name]] = last_q.numpy()

                (
                    ids_k,
                    probs_k,
                    logprobs_k,
                    tp,
                    tlp,
                    rank,
                    ent,
                ) = _target_stats(logits, passage.target, K)

                topk_ids[c, p_i] = ids_k
                topk_probs[c, p_i] = probs_k
                topk_logprobs[c, p_i] = logprobs_k
                target_p[c, p_i] = tp
                target_logp[c, p_i] = tlp
                target_rank[c, p_i] = rank
                where = np.flatnonzero(ids_k == passage.target)
                target_rank_in_topk[c, p_i] = int(where[0]) + 1 if where.size else 0
                entropy[c, p_i] = ent
                top1_ids[c, p_i] = int(ids_k[0])
                top1_probs[c, p_i] = float(probs_k[0])

                ctx_nll[c, p_i] = nll.numpy().astype(np.float32)

                if progress and (p_i + 1) % 20 == 0:
                    print(f"    {p_i + 1}/{n_pass} passages", flush=True)

        if progress:
            print(f"    step{step} done in {time.time() - t0:.0f}s", flush=True)

    vocab = {}
    for tid in sorted({int(x) for x in np.unique(topk_ids)} | {int(x) for x in target_ids}):
        vocab[str(tid)] = tokenizer.decode([tid])

    context_tokens = {
        p.name: [tokenizer.decode([i]) for i in p.ids]
        for p in passage_set.anchor_family
    }

    meta = {
        "model_id": cfg.model_id,
        "dtype": cfg.dtype,
        "top_k": K,
        "T": T,
        "provenance": passage_set.provenance,
        "passage_source": passage_set.source,
        "canonical_sha256": passage_set.canonical_sha256,
        "canonical_match": passage_set.canonical_sha256 == C.CANONICAL_PASSAGES_SHA256,
        "warning": passage_set.warning(),
        "run_id": cfg.run_id,
        "seed": cfg.seed,
        "elapsed_s": round(time.time() - started, 1),
        "vocab": vocab,
        "attention_captured": bool(cfg.capture_attention),
        "model_variant": C.MODEL_VARIANT,
        "free_disk": bool(cfg.extra.get("free_disk")),
    }

    return Bundle(
        checkpoints=tuple(cfg.checkpoints),
        names=names,
        kinds=kinds,
        topk_ids=topk_ids,
        topk_probs=topk_probs,
        topk_logprobs=topk_logprobs,
        target_ids=target_ids,
        target_p=target_p,
        target_logp=target_logp,
        target_rank=target_rank,
        target_rank_in_topk=target_rank_in_topk,
        entropy=entropy,
        top1_ids=top1_ids,
        top1_probs=top1_probs,
        ctx_nll=ctx_nll,
        ctx_ids=ctx_ids,
        attn_last=attn_last,
        attn_received=attn_received,
        attn_heads=attn_heads,
        attn_head_names=family,
        meta=meta,
        context_tokens=context_tokens,
    )


def verify_against_reference(bundle: Bundle, tol: float = C.REPRODUCTION_TOLERANCE) -> dict:
    """Check a fresh bundle against the numbers the earlier runs reported.

    A mismatch here means the run is not measuring the same thing -- wrong model
    variant, wrong dtype, wrong passage set -- and every probe result downstream is
    about a different object.  Returned rather than raised so the driver can decide.
    """
    if C.ANCHOR not in bundle.names:
        return {"checked": False, "reason": "anchor not present"}

    rows = []
    p_i = bundle.pi(C.ANCHOR)
    for step, expected in C.REFERENCE_TARGET_P.items():
        if step not in bundle.checkpoints:
            continue
        got = float(bundle.target_p[bundle.ci(step), p_i])
        rows.append(
            {
                "step": step,
                "quantity": "p_target",
                "expected": expected,
                "got": got,
                "delta": got - expected,
                "ok": abs(got - expected) <= max(tol, 5e-4),
            }
        )
    for step, expected in C.REFERENCE_TARGET_RANK.items():
        if step not in bundle.checkpoints:
            continue
        got = int(bundle.target_rank[bundle.ci(step), p_i])
        rows.append(
            {
                "step": step,
                "quantity": "rank_target",
                "expected": expected,
                "got": got,
                "delta": got - expected,
                "ok": got == expected,
            }
        )

    return {
        "checked": True,
        "all_ok": all(r["ok"] for r in rows),
        "rows": rows,
        "note": (
            "p_target is compared at 5e-4 because the reference table is rounded to 5 "
            "decimals; rank must match exactly."
        ),
    }
