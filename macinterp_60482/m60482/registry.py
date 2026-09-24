"""The probe contract.

A probe is a pure function of a :class:`~m60482.measure.Bundle`.  It may not load a
model, run a forward pass, or reach the network.  It answers one question, against a
decision rule written before the data existed, and it says what it cannot conclude.

The ``cannot_conclude`` field is not decoration.  The finding this project exists to
adjudicate was over-read once already; a probe that cannot state its own limits has not
finished thinking.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field, asdict
from typing import Callable, Protocol

from .measure import Bundle

#: The verdicts a probe may return.
VERDICTS = (
    "SUPPORTED",      # the pre-registered pattern appeared
    "REFUTED",        # the pre-registered killing pattern appeared
    "INCONCLUSIVE",   # neither; usually means underpowered, which is the honest answer
    "SCOPE_FAILED",   # the hypothesis does not apply to this phenomenon at all
    "NOT_RUN",        # a dependency was missing
    "ERROR",
)


@dataclass
class ProbeResult:
    probe: str
    paper: str
    hypothesis: str
    question: str
    verdict: str
    decision_rule: str
    summary: str
    evidence: dict = field(default_factory=dict)
    tables: dict = field(default_factory=dict)
    cannot_conclude: str = ""
    caveats: list = field(default_factory=list)
    runtime_s: float = 0.0

    def __post_init__(self) -> None:
        if self.verdict not in VERDICTS:
            raise ValueError(f"{self.probe}: verdict {self.verdict!r} not in {VERDICTS}")
        if not self.cannot_conclude:
            raise ValueError(
                f"{self.probe}: cannot_conclude is empty. Every probe must state its "
                "limits explicitly."
            )

    def as_dict(self) -> dict:
        return asdict(self)


class Probe(Protocol):
    name: str
    paper: str
    hypothesis: str
    question: str
    decision_rule: str
    order: int

    def __call__(self, bundle: Bundle) -> ProbeResult: ...


_REGISTRY: dict[str, Probe] = {}


def register(
    name: str,
    *,
    paper: str,
    hypothesis: str,
    question: str,
    decision_rule: str,
    order: int = 100,
) -> Callable:
    """Decorator registering a probe.

    ``order`` sets the run sequence.  Cheap probes that can kill the premise outright
    run first; expensive mechanistic probes run last, so a run that is going to end in
    "the premise does not hold" says so in the first minute.
    """

    def deco(fn: Callable[[Bundle], ProbeResult]) -> Callable:
        if name in _REGISTRY:
            raise ValueError(f"probe {name!r} already registered")
        fn.name = name                 # type: ignore[attr-defined]
        fn.paper = paper               # type: ignore[attr-defined]
        fn.hypothesis = hypothesis     # type: ignore[attr-defined]
        fn.question = question         # type: ignore[attr-defined]
        fn.decision_rule = decision_rule  # type: ignore[attr-defined]
        fn.order = order               # type: ignore[attr-defined]
        _REGISTRY[name] = fn           # type: ignore[assignment]
        return fn

    return deco


def all_probes() -> tuple[Probe, ...]:
    return tuple(sorted(_REGISTRY.values(), key=lambda p: (p.order, p.name)))


def get(name: str) -> Probe:
    return _REGISTRY[name]


def names() -> tuple[str, ...]:
    return tuple(p.name for p in all_probes())


def run(bundle: Bundle, only: tuple[str, ...] = (), *, progress: bool = True) -> list[ProbeResult]:
    """Run every registered probe (or the named subset) over one bundle.

    A probe that raises is recorded as ``ERROR`` and the suite continues -- one broken
    probe must not cost the whole run.
    """
    from . import probes  # noqa: F401  -- import for side effect: registration

    selected = [p for p in all_probes() if not only or p.name in only]
    results: list[ProbeResult] = []
    for probe in selected:
        t0 = time.time()
        if progress:
            print(f"-- {probe.name} ({probe.paper})", flush=True)
        try:
            res = probe(bundle)
            res.runtime_s = round(time.time() - t0, 2)
        except Exception as exc:  # noqa: BLE001 -- deliberate: keep the suite alive
            res = ProbeResult(
                probe=probe.name,
                paper=probe.paper,
                hypothesis=probe.hypothesis,
                question=probe.question,
                verdict="ERROR",
                decision_rule=probe.decision_rule,
                summary=f"{type(exc).__name__}: {exc}",
                evidence={"traceback": traceback.format_exc()},
                cannot_conclude="The probe did not run; nothing follows from it.",
                runtime_s=round(time.time() - t0, 2),
            )
        if progress:
            print(f"   -> {res.verdict}: {res.summary}", flush=True)
        results.append(res)
    return results
