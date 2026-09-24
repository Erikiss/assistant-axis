"""Turning probe results into an adjudication a reader can argue with.

The report is deliberately blunt about three things the previous round of this work got
wrong by omission: what the run could not have detected, which comparisons were made,
and whether the passage set was authentic.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from . import config as C
from . import papers as PAP
from .measure import Bundle
from .registry import ProbeResult

_VERDICT_ORDER = {
    "SUPPORTED": 0,
    "REFUTED": 1,
    "INCONCLUSIVE": 2,
    "SCOPE_FAILED": 3,
    "NOT_RUN": 4,
    "ERROR": 5,
}

_VERDICT_MARK = {
    "SUPPORTED": "SUPPORTED",
    "REFUTED": "refuted",
    "INCONCLUSIVE": "inconclusive",
    "SCOPE_FAILED": "out of scope",
    "NOT_RUN": "not run",
    "ERROR": "ERROR",
}


#: How a paper's probe verdicts roll up into a verdict on the paper itself.
_PAPER_VERDICT_RANK = ("SUPPORTED", "INCONCLUSIVE", "NOT_RUN", "REFUTED", "SCOPE_FAILED", "ERROR")


#: With polarity -1 the probe's verdict is read the other way round: the paper predicts
#: the measurement comes out against the probe's own hypothesis. SCOPE_FAILED and
#: NOT_RUN do not flip -- "the subject matter is absent" and "not measured" are not
#: evidence for anybody.
_FLIP = {"SUPPORTED": "REFUTED", "REFUTED": "SUPPORTED"}


def _apply_polarity(verdict: str, polarity: int) -> str:
    return _FLIP.get(verdict, verdict) if polarity < 0 else verdict


def _paper_verdict(verdicts: Sequence[str]) -> str:
    """A paper is only as strong as its strongest surviving probe.

    One SUPPORTED probe keeps a paper alive; a paper whose every probe is REFUTED or
    SCOPE_FAILED is out. NOT_RUN is treated as unknown, never as evidence.
    """
    for v in _PAPER_VERDICT_RANK:
        if v in verdicts:
            return v
    return "NOT_RUN"


def adjudicate(results: Sequence[ProbeResult]) -> dict:
    """Roll the probe verdicts up into an answer to the question actually asked:
    which of these papers explains the effect?"""
    by_probe = {r.probe: r for r in results}

    by_verdict: dict[str, list[str]] = {}
    for r in results:
        by_verdict.setdefault(r.verdict, []).append(r.probe)

    papers = []
    for paper in PAP.PAPERS:
        raw = {pr: by_probe[pr].verdict for pr in paper.probes if pr in by_probe}
        if not raw:
            continue
        read = {pr: _apply_polarity(v, paper.probes[pr]) for pr, v in raw.items()}
        papers.append(
            {
                "key": paper.key,
                "title": paper.title,
                "citation": paper.citation,
                "on_user_list": paper.on_user_list,
                "kind": paper.kind,
                "verdict": _paper_verdict(list(read.values())),
                "probes": {
                    pr: (
                        f"{v} (reads as {read[pr]} here)"
                        if paper.probes[pr] < 0 and v in _FLIP else v
                    )
                    for pr, v in raw.items()
                },
                "inverted_probes": [pr for pr, pol in paper.probes.items() if pol < 0],
                "surfaced_because": paper.surfaced_because,
            }
        )

    explanations = [p for p in papers if p["kind"] == "explanation"]
    listed = [p for p in explanations if p["on_user_list"]]
    unlisted = [p for p in explanations if not p["on_user_list"]]
    listed_alive = [p for p in listed if p["verdict"] in ("SUPPORTED", "INCONCLUSIVE")]
    unlisted_alive = [p for p in unlisted if p["verdict"] == "SUPPORTED"]

    if listed_alive:
        headline = (
            f"{len(listed_alive)} of {len(listed)} papers from the list survived: "
            + ", ".join(p["key"] for p in listed_alive)
        )
        if unlisted_alive:
            headline += (
                "; also supported, and not on the list: "
                + ", ".join(p["key"] for p in unlisted_alive)
            )
    elif unlisted_alive:
        headline = (
            f"None of the {len(listed)} papers on the list explains the effect. "
            "What is supported is not on it: "
            + ", ".join(p["key"] for p in unlisted_alive)
        )
    else:
        headline = (
            f"Nothing was supported: {len(listed)} listed papers and "
            f"{len(unlisted)} off-list candidates all failed or could not be tested."
        )

    return {
        "headline": headline,
        "papers": papers,
        "premise_checks": [p for p in papers if p["kind"] == "premise"],
        "listed_surviving": [p["key"] for p in listed_alive],
        "unlisted_supported": [p["key"] for p in unlisted_alive],
        "supported": by_verdict.get("SUPPORTED", []),
        "refuted": by_verdict.get("REFUTED", []),
        "inconclusive": by_verdict.get("INCONCLUSIVE", []),
        "scope_failed": by_verdict.get("SCOPE_FAILED", []),
        "not_run": by_verdict.get("NOT_RUN", []),
        "errors": by_verdict.get("ERROR", []),
        "n_probes": len(results),
    }


def _table(rows: Sequence[dict], cols: Sequence[str] | None = None) -> str:
    if not rows:
        return "_(no rows)_\n"
    cols = list(cols or rows[0].keys())

    def cell(v) -> str:
        if isinstance(v, float):
            return f"{v:.5f}"
        return str(v)

    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(cell(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out) + "\n"


def markdown(bundle: Bundle, results: Sequence[ProbeResult]) -> str:
    """The full report."""
    verdict = adjudicate(results)
    meta = bundle.meta
    lines: list[str] = []

    lines.append("# State 60482: which published mechanism explains the effect?\n")
    lines.append(f"**{verdict['headline']}**\n")

    # --- provenance, first, because it conditions everything below -------------------
    lines.append("## Run\n")
    lines.append(
        _table(
            [
                {"field": "model", "value": meta.get("model_id")},
                {"field": "dtype", "value": meta.get("dtype")},
                {"field": "checkpoints", "value": ", ".join(str(c) for c in bundle.checkpoints)},
                {"field": "exposure boundary", "value": f"{C.EXPOSURE_BOUNDARY[0]} -> {C.EXPOSURE_BOUNDARY[1]}"},
                {"field": "anchor train step", "value": C.ANCHOR_TRAIN_STEP},
                {"field": "passages", "value": f"{len(bundle.names)} ({len(bundle.state_names)} states / {len(bundle.control_names)} controls / {len(bundle.variant_names)} variants)"},
                {"field": "passage provenance", "value": meta.get("provenance")},
                {"field": "canonical hash match", "value": meta.get("canonical_match")},
                {"field": "run id", "value": meta.get("run_id")},
                {"field": "elapsed", "value": f"{meta.get('elapsed_s')} s"},
            ],
            ["field", "value"],
        )
    )
    if meta.get("warning"):
        lines.append(f"> **{meta['warning']}**\n")

    # --- the standing limits ----------------------------------------------------------
    n_states = max(len(bundle.state_names), 1)
    n_comparisons = n_states * C.N_BOUNDARIES
    corrected_alpha = 0.05 / n_comparisons

    lines.append("## What this run cannot do, regardless of what it found\n")
    lines.append(
        f"- With {C.N_CONTROLS} controls, the smallest attainable p-value is "
        f"1/{C.N_CONTROLS + 1} = {C.MIN_ATTAINABLE_P:.4f}. This study makes "
        f"{n_states} states x {C.N_BOUNDARIES} boundaries = {n_comparisons} "
        f"comparisons, which Bonferroni would require to clear "
        f"alpha = {corrected_alpha:.5f}. The floor is "
        f"{C.MIN_ATTAINABLE_P / corrected_alpha:.0f}x above that, so **no result in "
        "this design can survive correction**. That is a property of the design, not "
        "of the data, and no care in the analysis repairs it -- only more controls, or "
        "a single pre-registered comparison, would.\n"
        f"- There are {C.N_BOUNDARIES} boundaries and one of them is the exposure "
        "boundary. Selecting it after the fact is a 1-in-3 choice; the two placebo "
        "boundaries are the entire null, so 'outside the placebo range' means "
        "'outside a range estimated from two numbers'.\n"
        "- The anchor was chosen because it looked interesting. Any statistic that is "
        "a function of the selection criterion is descriptive only, and each probe "
        "below says which side of that line it sits on.\n"
        "- Attention shows which layer reads which position. It does not show an order "
        "of operations; a transformer has depth, not time.\n"
    )

    # --- the answer, per paper ---------------------------------------------------------
    lines.append("## Verdict per candidate paper\n")
    lines.append("Papers the user was given:\n")
    lines.append(
        _table(
            [
                {
                    "paper": p["key"],
                    "citation": p["citation"],
                    "verdict": _VERDICT_MARK.get(p["verdict"], p["verdict"]),
                    "probes": ", ".join(f"{k}={v}" for k, v in p["probes"].items()),
                }
                for p in verdict["papers"]
                if p["on_user_list"] and p["kind"] == "explanation"
            ],
            ["paper", "citation", "verdict", "probes"],
        )
    )
    lines.append("Candidates the list did not contain:\n")
    lines.append(
        _table(
            [
                {
                    "paper": p["key"],
                    "citation": p["citation"],
                    "verdict": _VERDICT_MARK.get(p["verdict"], p["verdict"]),
                    "why it is here": p["surfaced_because"],
                }
                for p in verdict["papers"]
                if not p["on_user_list"] and p["kind"] == "explanation"
            ],
            ["paper", "citation", "verdict", "why it is here"],
        )
    )
    lines.append(
        "Premise checks. These are not explanations -- they decide whether there is an "
        "effect to explain, and a `refuted` here means the *statistic* did not hold up, "
        "which is a finding about the measurement rather than about the model:\n"
    )
    by_probe_name = {r.probe: r for r in results}
    lines.append(
        _table(
            [
                {
                    "check": k,
                    "verdict": _VERDICT_MARK.get(v, v),
                    "finding": by_probe_name[k].summary if k in by_probe_name else "",
                }
                for p in verdict["papers"] if p["kind"] == "premise"
                for k, v in p["probes"].items()
            ],
            ["check", "verdict", "finding"],
        )
    )

    # --- probe results ----------------------------------------------------------------
    lines.append("## Probes\n")
    ordered = sorted(results, key=lambda r: (_VERDICT_ORDER.get(r.verdict, 9), r.probe))
    lines.append(
        _table(
            [
                {
                    "probe": r.probe,
                    "paper": r.paper,
                    "verdict": _VERDICT_MARK.get(r.verdict, r.verdict),
                    "summary": r.summary,
                }
                for r in ordered
            ],
            ["probe", "paper", "verdict", "summary"],
        )
    )

    for r in ordered:
        lines.append(f"### {r.probe} — {_VERDICT_MARK.get(r.verdict, r.verdict)}\n")
        lines.append(f"**Paper.** {r.paper}\n")
        lines.append(f"**Mechanism under test.** {r.hypothesis}\n")
        lines.append(f"**Question.** {r.question}\n")
        lines.append(f"**Decision rule (pre-registered).** {r.decision_rule}\n")
        lines.append(f"**Result.** {r.summary}\n")
        for title, rows in (r.tables or {}).items():
            lines.append(f"*{title}*\n")
            lines.append(_table(rows))
        if r.evidence:
            shown = {k: v for k, v in r.evidence.items() if k != "traceback"}
            lines.append("```json\n" + json.dumps(shown, indent=2, default=float) + "\n```\n")
        lines.append(f"**Cannot conclude.** {r.cannot_conclude}\n")
        for c in r.caveats or []:
            lines.append(f"- {c}\n")

    return "\n".join(lines)


def save(
    bundle: Bundle,
    results: Sequence[ProbeResult],
    out_dir: str | Path,
) -> dict[str, Path]:
    """Write ``report.md`` and ``results.json`` and return the paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    md = out / "report.md"
    md.write_text(markdown(bundle, results), encoding="utf-8")

    js = out / "results.json"
    js.write_text(
        json.dumps(
            {
                "adjudication": adjudicate(results),
                "meta": {k: v for k, v in bundle.meta.items() if k != "vocab"},
                "results": [asdict(r) for r in results],
            },
            indent=2,
            default=float,
        ),
        encoding="utf-8",
    )
    return {"report": md, "results": js}
