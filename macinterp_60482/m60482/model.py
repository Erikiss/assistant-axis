"""Checkpoint loading and the determinism rules the study depends on.

Two things in here are not stylistic preferences and should not be "optimised":

``float32``
    The whole study turns on the ordering of tokens whose probabilities differ by
    ~0.001 (at step131000 the target sits at 0.09087 and the competitor ``/`` at
    0.08965 -- a gap of 0.00122).  In float16 that ordering is not reliably
    reproducible, and *rank* is the headline statistic.  Run in fp32.  Pythia-1.4B is
    1.4e9 parameters, so fp32 weights are ~5.7 GB and an A100-40GB has room to spare.

``attn_implementation="eager"``
    ``output_attentions=True`` silently returns ``None`` under the SDPA and
    FlashAttention paths.  A probe that reads attention would then get no error, just
    nothing.  :func:`load_checkpoint` forces eager and :func:`forward_pass` asserts the
    tensors actually arrived.
"""

from __future__ import annotations

import gc
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

from . import config as C

_DTYPES = {"float32": "float32", "bfloat16": "bfloat16", "float16": "float16"}


def set_determinism(seed: int = 0) -> None:
    """Make repeated runs on the same machine produce the same logits."""
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # TF32 changes the low bits of every matmul.  Off, or fp32 buys us nothing.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def revision_for(step: int) -> str:
    """HuggingFace revision tag for a Pythia checkpoint."""
    return f"step{step}"


@dataclass
class LoadedCheckpoint:
    step: int
    model: object
    tokenizer: object
    revision: str
    dtype: str
    device: str
    n_layers: int
    n_heads: int

    def free(self) -> None:
        import torch

        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_tokenizer(model_id: str = C.MODEL_ID, revision: str = C.TOKENIZER_REVISION):
    """Load the tokenizer at the revision the passage set was built with.

    The passage ids are frozen in ``passagen.json``; the tokenizer is only needed to
    *display* them.  Pinning the revision keeps the displayed strings honest.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def verify_tokenizer(tokenizer) -> None:
    """Assert the tokenizer agrees with the frozen ids.

    This catches a wrong *tokenizer* -- a different model family, a corrupted download.
    It cannot catch a wrong model *variant*: ``tokenizer.json`` has the same sha256 on
    ``pythia-1.4b`` and ``pythia-1.4b-deduped``, and at every revision of both. Only the
    training data order discriminates those, which is what
    :func:`m60482.pile.identify_variant` is for.
    """
    got = tokenizer.decode([C.TARGET_TOKEN_ID])
    if got != C.TARGET_TOKEN_STR:
        raise RuntimeError(
            f"tokenizer maps {C.TARGET_TOKEN_ID} -> {got!r}, expected "
            f"{C.TARGET_TOKEN_STR!r}. Wrong model or wrong revision."
        )
    name = "".join(tokenizer.decode([i]) for i in C.NAME_TOKEN_IDS)
    if name.strip() != "McQuarrie":
        raise RuntimeError(
            f"name ids {C.NAME_TOKEN_IDS} decode to {name!r}, expected ' McQuarrie'."
        )


def load_checkpoint(
    step: int,
    *,
    model_id: str = C.MODEL_ID,
    dtype: str = "float32",
    device: str = "cuda",
    need_attention: bool = True,
    cache_dir: str | None = None,
) -> LoadedCheckpoint:
    import torch
    from transformers import AutoModelForCausalLM

    if dtype not in _DTYPES:
        raise ValueError(f"dtype must be one of {sorted(_DTYPES)}, got {dtype!r}")
    if dtype == "float16" and need_attention:
        # Not forbidden, but the caller has to say it out loud via dtype="float16".
        print(
            "WARNING: float16 with rank-sensitive measurements. The target and its "
            "competitor are ~0.001 apart in probability; ranks may not reproduce.",
            flush=True,
        )

    torch_dtype = getattr(torch, dtype)
    revision = revision_for(step)
    kwargs: dict = {"revision": revision, "torch_dtype": torch_dtype}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    if need_attention:
        # Without this, output_attentions=True returns None instead of raising.
        kwargs["attn_implementation"] = "eager"

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    model.to(device)

    cfg = model.config
    return LoadedCheckpoint(
        step=step,
        model=model,
        tokenizer=None,
        revision=revision,
        dtype=dtype,
        device=device,
        n_layers=cfg.num_hidden_layers,
        n_heads=cfg.num_attention_heads,
    )


def _snapshot_dir(step: int, model_id: str, cache_dir: str | None) -> "Path | None":
    """Where huggingface_hub put this revision's files, if it is still there."""
    from pathlib import Path

    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except ImportError:
        return None
    root = Path(cache_dir or HF_HUB_CACHE) / f"models--{model_id.replace('/', '--')}"
    snapshots = root / "snapshots"
    if not snapshots.exists():
        return None
    return root


def free_checkpoint_disk(step: int, model_id: str = C.MODEL_ID, cache_dir: str | None = None) -> int:
    """Delete a downloaded revision's blobs.  Returns bytes freed.

    Four fp32 checkpoints are ~22.6 GB of safetensors, and HuggingFace does NOT share
    blobs between revisions of the same repo -- each ``stepNNNNNN`` is a full copy. On
    Colab the disk fills before the GPU does, usually on the fourth checkpoint, after
    the run has already cost forty minutes.

    This is deliberately not automatic: re-downloading is expensive, so a run that fits
    should keep its cache. :func:`checkpoint` takes ``free_disk=True`` when it does not.
    """
    import shutil

    root = _snapshot_dir(step, model_id, cache_dir)
    if root is None or not root.exists():
        return 0
    freed = sum(f.stat().st_size for f in root.rglob("*") if f.is_file())
    shutil.rmtree(root, ignore_errors=True)
    return freed


@contextmanager
def checkpoint(step: int, *, free_disk: bool = False, **kwargs) -> Iterator[LoadedCheckpoint]:
    """Load one checkpoint, hand it over, free the GPU memory, and optionally the disk.

    Four fp32 Pythia-1.4B checkpoints do not need to be resident at once. Always use
    this rather than holding four models. Pass ``free_disk=True`` on a machine whose
    disk cannot hold ~22.6 GB of cached revisions.
    """
    ckpt = load_checkpoint(step, **kwargs)
    try:
        yield ckpt
    finally:
        ckpt.free()
        if free_disk:
            freed = free_checkpoint_disk(
                step, kwargs.get("model_id", C.MODEL_ID), kwargs.get("cache_dir")
            )
            if freed:
                print(f"    freed {freed / 1e9:.1f} GB of step{step} from disk", flush=True)


def forward_pass(
    ckpt: LoadedCheckpoint,
    ids: Sequence[int],
    *,
    need_attention: bool = True,
):
    """One un-batched forward pass over a single passage.

    Un-batched on purpose: padding and batched reduction order both perturb logits at
    the 1e-4 level, which is the level this study reads.  69 passages x 207 tokens is
    cheap enough that batching buys nothing worth the risk.

    Returns ``(logits_last, attentions)`` where ``logits_last`` is ``[vocab]`` float32
    on CPU and ``attentions`` is ``[layer, head, T]`` float32 on CPU -- the last
    query's attention over the context -- or ``None`` when not requested.
    """
    import torch

    device = ckpt.device
    input_ids = torch.tensor([list(ids)], dtype=torch.long, device=device)

    with torch.inference_mode():
        out = ckpt.model(input_ids, output_attentions=need_attention, use_cache=False)

    logits_last = out.logits[0, -1].to(torch.float32).cpu()

    attentions = None
    if need_attention:
        if out.attentions is None or any(a is None for a in out.attentions):
            raise RuntimeError(
                "output_attentions=True returned None. The model was not loaded with "
                'attn_implementation="eager"; attention numbers would be missing '
                "rather than wrong, which is worse."
            )
        # each entry is [batch, head, q, k]; keep the last query row only
        attentions = torch.stack(
            [a[0, :, -1, :].to(torch.float32).cpu() for a in out.attentions]
        )

    return logits_last, attentions


def forward_pass_all(ckpt: LoadedCheckpoint, ids: Sequence[int], *, need_attention: bool = True):
    """Everything the study wants from one passage, in a single forward pass.

    Returns ``(logits_last, ctx_nll, last_query, received)``:

    ``logits_last``
        ``[vocab]`` float32 on CPU -- the next-token distribution at the final position.
    ``ctx_nll``
        ``[T-1]`` float32 -- teacher-forced NLL of each context token given its prefix.
        Entry *i* is the NLL of ``ids[i+1]``.  The earlier token-level analysis lived on
        exactly this array.
    ``last_query``
        ``[layer, head, T]`` -- the final query's attention, or ``None``.
    ``received``
        ``[layer, head, T]`` -- attention received by each position, averaged over the
        queries that could attend to it, or ``None``.

    Computing the context NLL here rather than in a second pass halves the sweep: 69
    passages x 4 checkpoints is 276 forward passes saved.

    Materialising the full ``[layer, head, T, T]`` attention costs ~63 MB in fp32 at
    T=207 with 24 layers and 16 heads.  That is fine one passage at a time and is why
    the sweep is un-batched.
    """
    import torch

    device = ckpt.device
    input_ids = torch.tensor([list(ids)], dtype=torch.long, device=device)

    with torch.inference_mode():
        out = ckpt.model(input_ids, output_attentions=need_attention, use_cache=False)

    logits = out.logits[0]                                   # [T, vocab]
    logits_last = logits[-1].to(torch.float32).cpu()

    # NLL of each context token given its prefix, in the same pass.
    ctx_logprobs = torch.log_softmax(logits[:-1].to(torch.float32), dim=-1)
    targets = input_ids[0, 1:].unsqueeze(1)
    ctx_nll = -ctx_logprobs.gather(1, targets).squeeze(1).cpu()

    if not need_attention:
        return logits_last, ctx_nll, None, None

    if out.attentions is None or any(a is None for a in out.attentions):
        raise RuntimeError(
            "output_attentions=True returned None. The model was not loaded with "
            'attn_implementation="eager"; attention numbers would be missing rather '
            "than wrong, which is worse."
        )

    last_query = torch.stack([a[0, :, -1, :].to(torch.float32).cpu() for a in out.attentions])

    # Causal masking means position j is attended to by queries j..T-1 only. Dividing by
    # T would understate late positions; divide by the number of queries that could have
    # attended at all.
    T = input_ids.shape[1]
    denom = torch.arange(T, 0, -1, dtype=torch.float32)
    received = torch.stack(
        [a[0].to(torch.float32).cpu().sum(dim=1) / denom for a in out.attentions]
    )

    return logits_last, ctx_nll, last_query, received


def forward_pass_full_attention(ckpt: LoadedCheckpoint, ids: Sequence[int]):
    """Backwards-compatible shim: ``(logits_last, last_query, received)``."""
    logits_last, _, last_query, received = forward_pass_all(ckpt, ids, need_attention=True)
    return logits_last, last_query, received


def load_untrained(
    *,
    model_id: str = C.MODEL_ID,
    dtype: str = "float32",
    device: str = "cuda",
    need_attention: bool = True,
    seed: int = 0,
) -> LoadedCheckpoint:
    """A randomly-initialised Pythia-1.4B: the architecture with no training at all.

    This is the baseline the position-bias theory is actually stated over.  It claims
    the influence profile follows from causal attention plus residual connections, and
    demonstrates it on untrained GPT-2 and Qwen2.  Without this baseline, "all 69
    passages share a profile" is equally consistent with "every passage learned the
    same thing"; with it, the question becomes whether the profile was there before any
    token was ever seen.

    Costs one extra sweep and no download -- only the config is fetched.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    torch.manual_seed(seed)
    cfg = AutoConfig.from_pretrained(model_id)
    kwargs = {"torch_dtype": getattr(torch, dtype)}
    if need_attention:
        kwargs["attn_implementation"] = "eager"
    model = AutoModelForCausalLM.from_config(cfg, **kwargs)
    model.eval()
    model.to(device)

    return LoadedCheckpoint(
        step=-1,
        model=model,
        tokenizer=None,
        revision=f"untrained(seed={seed})",
        dtype=dtype,
        device=device,
        n_layers=cfg.num_hidden_layers,
        n_heads=cfg.num_attention_heads,
    )
