"""Loading, verifying and (when necessary) degrading the 69-passage set.

The canonical ``passagen.json`` is the K8 artefact: 19 states, 40 controls, 10 name
variants, each ``T`` context tokens plus one held-out target id.  This module will use
it when it can find it, verify it hard when it does, and fall back -- loudly and with a
permanent marker on every downstream result -- when it cannot.

Provenance levels, in decreasing order of trust:

``canonical``
    A file whose canonical hash matches :data:`~m60482.config.CANONICAL_PASSAGES_SHA256`.
    Results are comparable with the earlier runs.

``regenerated``
    Rebuilt from the Exp-2 tables (``study_states.csv`` + ``token_map_all_states.csv``)
    and the Pile stream.  Structurally identical; hash checked against the canonical
    value and reported either way.

``reconstructed``
    Only the anchor family could be rebuilt (from the anchor text + the variant
    spellings), controls re-drawn from the Pile with the documented seed.  Cross-run
    comparisons of control statistics are NOT valid.

``synthetic``
    Nothing was available.  The shapes are right and the pipeline exercises end to end,
    but every number is meaningless.  Present so that the code can be tested without
    Drive access; never for a claim.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from . import config as C

PROVENANCE_LEVELS = ("canonical", "regenerated", "reconstructed", "synthetic")


@dataclass(frozen=True)
class Passage:
    name: str
    kind: str          # one of config.PASSAGE_KINDS
    ids: tuple[int, ...]
    target: int
    T: int

    def __post_init__(self) -> None:
        if len(self.ids) != self.T:
            raise ValueError(f"{self.name}: T={self.T} but {len(self.ids)} ids")
        if self.kind not in C.PASSAGE_KINDS:
            raise ValueError(f"{self.name}: unknown kind {self.kind!r}")


@dataclass(frozen=True)
class PassageSet:
    passages: tuple[Passage, ...]
    provenance: str
    source: str
    canonical_sha256: str

    @property
    def trustworthy(self) -> bool:
        """True when control-referenced statistics may be compared across runs."""
        return self.provenance in ("canonical", "regenerated")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.passages)

    def by_name(self, name: str) -> Passage:
        for p in self.passages:
            if p.name == name:
                return p
        raise KeyError(name)

    def of_kind(self, *kinds: str) -> tuple[Passage, ...]:
        return tuple(p for p in self.passages if p.kind in kinds)

    @property
    def anchor(self) -> Passage:
        return self.by_name(C.ANCHOR)

    @property
    def controls(self) -> tuple[Passage, ...]:
        return self.of_kind("kontrolle")

    @property
    def variants(self) -> tuple[Passage, ...]:
        return self.of_kind("variante")

    @property
    def states(self) -> tuple[Passage, ...]:
        return self.of_kind("state_haupt", "state_extra")

    @property
    def anchor_family(self) -> tuple[Passage, ...]:
        """The anchor plus its 10 never-trained name variants -- 14 in the K8 run's
        accounting, which also stored per-head weights for the other main states."""
        return (self.anchor,) + self.variants

    def warning(self) -> str | None:
        if self.provenance == "canonical":
            return None
        if self.provenance == "regenerated":
            return (
                "Passage set was regenerated from the Exp-2 tables rather than loaded "
                "from the original K8 snapshot. The canonical hash matched, so results "
                "are comparable, but the file itself is new."
            )
        if self.provenance == "reconstructed":
            return (
                "DEGRADED: only the anchor family is authentic; the 40 controls were "
                "re-drawn. Control-referenced p-values are NOT comparable with the "
                "earlier runs and must not be quoted against them."
            )
        return (
            "SYNTHETIC passage set. Every number produced from this run is meaningless "
            "and exists only to exercise the pipeline."
        )


# ------------------------------------------------------------------------------------
# hashing / verification
# ------------------------------------------------------------------------------------


def canonical_hash(passages: Iterable[Passage]) -> str:
    """The K8 canonical hash: sha256 over sorted ``(name, art, T, ids, ziel)``."""
    canonical = sorted(
        (p.name, p.kind, p.T, list(p.ids), p.target) for p in passages
    )
    return hashlib.sha256(
        json.dumps(canonical, separators=(",", ":")).encode()
    ).hexdigest()


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _structure_ok(passages: Sequence[Passage]) -> str | None:
    """Return a human-readable reason the set is not a K8 set, or None if it is."""
    counts = Counter(p.kind for p in passages)
    n_states = counts["state_haupt"] + counts["state_extra"]
    if (n_states, counts["kontrolle"], counts["variante"]) != (
        C.N_STATES,
        C.N_CONTROLS,
        C.N_VARIANTS,
    ):
        return (
            f"expected {C.N_STATES} states / {C.N_CONTROLS} controls / "
            f"{C.N_VARIANTS} variants, got {n_states} / {counts['kontrolle']} / "
            f"{counts['variante']}"
        )
    try:
        anchor = next(p for p in passages if p.name == C.ANCHOR)
    except StopIteration:
        return f"no passage named {C.ANCHOR}"
    if anchor.T != C.ANCHOR_T:
        return f"anchor T={anchor.T}, expected {C.ANCHOR_T}"
    if tuple(anchor.ids[124:128]) != C.NAME_TOKEN_IDS:
        return (
            f"anchor name tokens at 124:128 are {tuple(anchor.ids[124:128])}, "
            f"expected {C.NAME_TOKEN_IDS}"
        )
    if anchor.target != C.TARGET_TOKEN_ID:
        return f"anchor target is {anchor.target}, expected {C.TARGET_TOKEN_ID}"
    return None


def parse_passages(obj: dict) -> tuple[Passage, ...]:
    """Parse the K8 ``{"passagen": [...]}`` structure. German keys are the on-disk
    format; they are translated here and nowhere else."""
    raw = obj["passagen"] if isinstance(obj, dict) else obj
    out = []
    for entry in raw:
        ids = tuple(int(i) for i in entry["ids"])
        out.append(
            Passage(
                name=entry["name"],
                kind=entry["art"],
                ids=ids,
                target=int(entry["ziel"]),
                T=int(entry.get("T", len(ids))),
            )
        )
    return tuple(out)


def load_file(path: str | Path, *, strict: bool = True) -> PassageSet:
    """Load a ``passagen.json`` and classify its provenance."""
    path = Path(path)
    passages = parse_passages(json.loads(path.read_text()))
    reason = _structure_ok(passages)
    if reason is not None:
        if strict:
            raise ValueError(f"{path} is not a K8 passage set: {reason}")
        return PassageSet(passages, "synthetic", str(path), canonical_hash(passages))
    digest = canonical_hash(passages)
    provenance = "canonical" if digest == C.CANONICAL_PASSAGES_SHA256 else "reconstructed"
    return PassageSet(passages, provenance, str(path), digest)


def _candidate_paths(roots: Sequence[Path]) -> list[Path]:
    seen: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        if root.is_file():
            seen.append(root)
            continue
        seen.append(root / "passagen.json")
        seen.append(root / "data" / "passagen.json")
        # the K8 result tree, up to three levels down
        pattern = "K8_Pythia_NLL_MultiCheckpoint_GPUAI"
        for depth in ("*", "*/*", "*/*/*"):
            seen.extend(root.glob(f"{depth}/{pattern}/**/passagen.json"))
            seen.extend(root.glob(f"{depth}/{pattern}/**/*.zip"))
    return [p for p in seen if p.exists()]


def discover(roots: Sequence[str | Path], *, verbose: bool = True) -> PassageSet | None:
    """Search ``roots`` for a usable passage set, including inside ZIP archives.

    Every candidate is printed, as in the original run -- when this fails you want to
    see what it looked at, not just that it failed.
    """
    for cand in _candidate_paths([Path(r) for r in roots]):
        if verbose:
            print(f"candidate: {cand}", flush=True)
        try:
            if cand.suffix == ".zip":
                with zipfile.ZipFile(cand) as zf:
                    for member in zf.namelist():
                        if not member.endswith("passagen.json"):
                            continue
                        obj = json.loads(zf.read(member))
                        passages = parse_passages(obj)
                        if _structure_ok(passages) is None:
                            digest = canonical_hash(passages)
                            prov = (
                                "canonical"
                                if digest == C.CANONICAL_PASSAGES_SHA256
                                else "reconstructed"
                            )
                            return PassageSet(passages, prov, f"{cand}::{member}", digest)
                continue
            ps = load_file(cand, strict=False)
            if ps.provenance != "synthetic":
                return ps
        except (ValueError, KeyError, TypeError, OSError, zipfile.BadZipFile) as exc:
            if verbose:
                print(f"  rejected: {exc}", flush=True)
    return None


# ------------------------------------------------------------------------------------
# synthetic fallback
# ------------------------------------------------------------------------------------


def synthetic(seed: int = 0, vocab_size: int = 50_254) -> PassageSet:
    """A structurally valid but meaningless passage set, for tests and dry runs.

    The anchor's name tokens and target are real so that position-indexed code paths
    are exercised; everything else is noise from a seeded RNG.
    """
    import random

    rng = random.Random(seed)

    def ids(n: int) -> tuple[int, ...]:
        return tuple(rng.randrange(1, vocab_size) for _ in range(n))

    passages: list[Passage] = []

    anchor_ids = list(ids(C.ANCHOR_T))
    for offset, tok in zip(C.NAME_POSITIONS, C.NAME_TOKEN_IDS):
        anchor_ids[offset] = tok
    passages.append(
        Passage(C.ANCHOR, "state_haupt", tuple(anchor_ids), C.TARGET_TOKEN_ID, C.ANCHOR_T)
    )

    for i in range(C.N_STATES - 1):
        kind = "state_haupt" if i < 3 else "state_extra"
        passages.append(
            Passage(f"state_extra_{i:02d}", kind, ids(C.ANCHOR_T), rng.randrange(1, vocab_size), C.ANCHOR_T)
        )

    for name in C.VARIANT_NAMES:
        v = list(anchor_ids)
        for offset in C.NAME_POSITIONS:
            v[offset] = rng.randrange(1, vocab_size)
        passages.append(
            Passage(name, "variante", tuple(v), C.TARGET_TOKEN_ID, C.ANCHOR_T)
        )

    for i in range(C.N_CONTROLS):
        passages.append(
            Passage(f"kontrolle_{i:02d}", "kontrolle", ids(C.ANCHOR_T), rng.randrange(1, vocab_size), C.ANCHOR_T)
        )

    return PassageSet(tuple(passages), "synthetic", f"synthetic(seed={seed})", canonical_hash(passages))


def resolve(
    explicit: str | Path | None = None,
    roots: Sequence[str | Path] = (),
    *,
    allow_synthetic: bool = False,
    verbose: bool = True,
) -> PassageSet:
    """The one entry point a driver should call.

    ``explicit`` wins; then a search of ``roots``; then, only if ``allow_synthetic``,
    a synthetic set.  Anything other than a canonical set prints its warning.
    """
    ps: PassageSet | None = None
    if explicit:
        ps = load_file(explicit)
    else:
        ps = discover(roots, verbose=verbose)
    if ps is None:
        if not allow_synthetic:
            raise FileNotFoundError(
                "No K8 passagen.json found. Pass an explicit path, or set "
                "allow_synthetic=True to run the pipeline on meaningless data."
            )
        ps = synthetic()
    if verbose:
        print(f"passages: {len(ps.passages)} from {ps.source} [{ps.provenance}]", flush=True)
        warn = ps.warning()
        if warn:
            print(f"  !! {warn}", flush=True)
    return ps


def verify_anchor_against_pile(ps: PassageSet, *, timeout: float = 60.0) -> dict:
    """Check a passage set's anchor against the actual Pile training sample.

    This is the strongest available check on a non-canonical passage set.  The anchor's
    context is exactly ``row[:207]`` of training sample 134428942, so a set whose anchor
    matches that is carrying the real passage even when its file hash does not match the
    K8 snapshot -- which usually means only the controls were re-drawn.

    Costs one 4 KB range request and needs network access.  Returns a dict rather than
    raising, so a driver can record the outcome either way.
    """
    from . import pile

    try:
        ctx, target = pile.rebuild_anchor(timeout=timeout)
    except (OSError, RuntimeError, AssertionError) as exc:
        return {"checked": False, "reason": f"{type(exc).__name__}: {exc}"}

    anchor = ps.anchor
    ctx_ok = tuple(anchor.ids) == tuple(ctx)
    tgt_ok = anchor.target == target
    first_diff = next(
        (i for i, (a, b) in enumerate(zip(anchor.ids, ctx)) if a != b), None
    )
    return {
        "checked": True,
        "context_matches": ctx_ok,
        "target_matches": tgt_ok,
        "first_differing_position": first_diff,
        "note": (
            "The anchor is the real Pile passage regardless of the file hash."
            if ctx_ok and tgt_ok
            else "The anchor does NOT match the Pile sample; this is not state 60482."
        ),
    }
