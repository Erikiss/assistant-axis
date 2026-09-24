"""Fetch the exact Pythia training sequence for a global sample index.

Verified end to end on 2026-09-24 against ``global_sample_index`` 134428942:
the row returned by :func:`fetch_training_row` has ``" Mc","Qu","ar","rie"``
(3044/3864/274/6595) at positions 124-127, ``" per"`` (591) at position 207 and
``"\\n"`` (187) at position 46 -- i.e. exactly the frozen anchor in
:mod:`m60482.config`, and exactly the 208-token window [0:208] of the row.

Why this module exists
----------------------
The documented route (``utils/batch_viewer.py`` in EleutherAI/pythia) wants the
unsharded memmap on local disk.  That is 600,078,336,000 bytes of ``.bin`` plus a
1.76 GB ``.idx`` -- 602 GB.  Colab cannot hold it and does not need to.

The preshuffled corpus is a *flat* array of ``uint16``, laid out as
``143000 * 1024`` rows of exactly 2049 tokens, in training order, with no header
and no padding.  The arithmetic is exact, not approximate::

    143000 * 1024 * 2049 * 2 == 600_078_336_000 == sum of the 21 .bin shards

So row ``i`` lives at byte ``i * 4098`` of the concatenation of the shards, and a
single HTTP range request of 4098 bytes retrieves it.  The ``.idx`` file is never
needed: it only maps document ids to offsets, and for the preshuffled corpus every
"document" is one fixed-stride 2049-token training sample.

Shards are exactly 30_000_000_000 bytes each (the last one 78_336_000), which is
*not* a multiple of 4098, so a row can straddle a shard boundary.
:func:`fetch_training_row` handles that by issuing two requests.
"""

from __future__ import annotations

import struct
import urllib.request

#: Tokens per stored training sample.  2048 inputs + 1 shifted target.
SEQ_LEN = 2049

#: Bytes per stored training sample (uint16).
ROW_BYTES = SEQ_LEN * 2  # 4098

#: Sequences per optimizer step.  ``step = global_sample_index // BATCH_SEQS``.
BATCH_SEQS = 1024

#: Byte size of each shard, in order.  20 full shards plus a short tail.
SHARD_BYTES: tuple[int, ...] = (30_000_000_000,) * 20 + (78_336_000,)

TOTAL_BYTES = sum(SHARD_BYTES)  # 600_078_336_000
N_ROWS = TOTAL_BYTES // ROW_BYTES  # 146_432_000 == 143000 * 1024

#: The non-deduplicated Pile in Pythia training order -- the corpus that
#: ``EleutherAI/pythia-1.4b`` (no ``-deduped``) was trained on.
STANDARD_REPO = "EleutherAI/pile-standard-pythia-preshuffled"

#: The deduplicated corpus, for ``EleutherAI/pythia-1.4b-deduped``.  Use it as a
#: negative control: the same index there is an unrelated document.
DEDUPED_REPO = "EleutherAI/pile-deduped-pythia-preshuffled"

_URL = "https://huggingface.co/datasets/{repo}/resolve/main/document-{n:05d}-of-00020.bin"


def step_of(global_sample_index: int) -> int:
    """Optimizer step during which this sample was consumed."""
    return global_sample_index // BATCH_SEQS


def _locate(byte_offset: int) -> tuple[int, int]:
    """(shard index, offset within that shard) for a global byte offset."""
    run = 0
    for n, size in enumerate(SHARD_BYTES):
        if byte_offset < run + size:
            return n, byte_offset - run
        run += size
    raise IndexError(f"byte offset {byte_offset} past end of corpus ({TOTAL_BYTES})")


def _range(repo: str, shard: int, start: int, length: int, timeout: float) -> bytes:
    req = urllib.request.Request(
        _URL.format(repo=repo, n=shard),
        headers={"Range": f"bytes={start}-{start + length - 1}"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if resp.status != 206:
            raise RuntimeError(
                f"expected HTTP 206 Partial Content, got {resp.status}. The HF CDN "
                "must honour Range requests for this to be cheap."
            )
        data = resp.read()
    if len(data) != length:
        raise RuntimeError(f"short read: asked {length} bytes, got {len(data)}")
    return data


def fetch_training_row(
    global_sample_index: int,
    *,
    repo: str = STANDARD_REPO,
    timeout: float = 60.0,
) -> list[int]:
    """Return the 2049 token ids of one training sample.  ~4 KB over the wire."""
    if not 0 <= global_sample_index < N_ROWS:
        raise IndexError(f"{global_sample_index} outside [0, {N_ROWS})")

    offset = global_sample_index * ROW_BYTES
    shard, local = _locate(offset)
    room = SHARD_BYTES[shard] - local
    if room >= ROW_BYTES:
        raw = _range(repo, shard, local, ROW_BYTES, timeout)
    else:  # row straddles a shard boundary
        raw = _range(repo, shard, local, room, timeout)
        raw += _range(repo, shard + 1, 0, ROW_BYTES - room, timeout)

    return list(struct.unpack(f"<{SEQ_LEN}H", raw))


def check_anchor(row: list[int]) -> None:
    """Assert that ``row`` is the frozen anchor.  Raises on any mismatch.

    This is the one check that decides ``pythia-1.4b`` vs ``pythia-1.4b-deduped``
    from artifacts instead of from assumption: run it on a row fetched from
    :data:`STANDARD_REPO` and again on one from :data:`DEDUPED_REPO`.  Exactly one
    of them passes, and that names the variant.
    """
    from . import config as C

    if len(row) != SEQ_LEN:
        raise AssertionError(f"row has {len(row)} tokens, expected {SEQ_LEN}")

    got_name = tuple(row[C.NAME_POSITIONS[0] : C.NAME_POSITIONS[-1] + 1])
    if got_name != tuple(C.NAME_TOKEN_IDS):
        raise AssertionError(
            f"positions {C.NAME_POSITIONS} are {got_name}, expected "
            f"{C.NAME_TOKEN_IDS} (' McQuarrie')"
        )
    if row[C.ANCHOR_T] != C.TARGET_TOKEN_ID:
        raise AssertionError(
            f"position {C.ANCHOR_T} is {row[C.ANCHOR_T]}, expected "
            f"{C.TARGET_TOKEN_ID} ({C.TARGET_TOKEN_STR!r})"
        )
    if row[C.SINK_POSITION_NEWLINE] != 187:
        raise AssertionError(
            f"position {C.SINK_POSITION_NEWLINE} is {row[C.SINK_POSITION_NEWLINE]}, "
            "expected 187 ('\\n') -- the sink the attention probe reports"
        )


def anchor_context_and_target(row: list[int]) -> tuple[list[int], int]:
    """Split a verified anchor row into its 207-token context and its target."""
    from . import config as C

    return row[: C.ANCHOR_T], row[C.ANCHOR_T]
