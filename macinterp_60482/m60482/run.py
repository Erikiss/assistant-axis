"""The driver: resolve passages, measure, probe, report.

Usable three ways -- as a Colab cell (``run.main([...])``), as a module
(``python -m m60482.run --smoke``), and piecewise from a notebook.

Resumability matters more than it looks: four fp32 Pythia checkpoints are a ~23 GB
download, and a Colab session that dies at checkpoint four should not start again at
checkpoint one.  The bundle is written once and every probe run afterwards reads it
from disk.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Sequence

from . import aux as A
from . import config as C
from . import passages as P
from . import registry, report
from .measure import Bundle, measure, verify_against_reference

DEFAULT_ROOTS = (
    "/content/drive/MyDrive/Colab_Pythia_Results",
    "/content/pythia_attn_runtime",
    ".",
)


def make_run_id(stamp: str | None = None) -> str:
    return stamp or time.strftime("%Y%m%d_%H%M%S")


def _gate_passages(ps, *, strict: bool = True) -> None:
    """Check the anchor against the real Pile sample before spending an hour on it.

    A wrong model variant is the failure this catches, and nothing else in the pipeline
    can: ``pythia-1.4b`` and ``-deduped`` have byte-identical configs and the same
    ``tokenizer.json`` sha256 at every revision, so ``verify_tokenizer`` passes on both.
    Only the training data order tells them apart, and that is one 4 KB range request.

    Needs network. A check that could not run is reported and does not block -- an
    offline environment is not evidence of a wrong passage set -- but a check that ran
    and *failed* stops the run, because every number after it would be about a
    different object.
    """
    check = P.verify_anchor_against_pile(ps)
    if not check.get("checked"):
        print(f"anchor/Pile check skipped: {check.get('reason')}", flush=True)
        return
    if check["context_matches"] and check["target_matches"]:
        print("anchor matches the Pile training sample (variant confirmed).", flush=True)
        return
    message = (
        "The anchor does not match Pile sample "
        f"{C.ANCHOR_GLOBAL_SAMPLE_INDEX}: first differing position "
        f"{check.get('first_differing_position')}. This is not state 60482 -- most "
        "likely the passage set was built against pythia-1.4b-deduped, whose data "
        "order is different. Every measurement below would be about another passage."
    )
    if strict:
        raise RuntimeError(message)
    print(f"WARNING: {message}", flush=True)


def build_bundle(
    cfg: C.RunConfig,
    *,
    passages_path: str | None = None,
    roots: Sequence[str] = DEFAULT_ROOTS,
    allow_synthetic: bool = False,
    cache: Path | None = None,
    aux: bool = True,
    untrained: bool = True,
    norm_attn: bool = True,
) -> Bundle:
    """Measure, or load a previously measured bundle from ``cache``.

    The auxiliary measurements default to on. Four probes -- context_dependence,
    norm_attribution, the greedy-continuation half of memorization_entry, and the
    untrained comparison in position_bias -- report NOT_RUN without them, and
    context_dependence is the one most likely to decide the whole question.

    Each stage is appended to the cached bundle as it completes, so a session that dies
    part-way keeps what it measured.
    """
    if cache and cache.exists():
        print(f"reusing bundle: {cache}", flush=True)
        bundle = Bundle.load(cache)
        ps = None
    else:
        ps = P.resolve(passages_path, roots, allow_synthetic=allow_synthetic)
        _gate_passages(ps, strict=not allow_synthetic)
        bundle = measure(ps, cfg)
        if cache:
            bundle.save(cache)
            print(f"bundle written: {cache}", flush=True)

    stages = []
    if aux and "ladder" not in bundle.aux:
        stages.append(("auxiliary measurements", lambda p: A.measure_aux(p, cfg)))
    if untrained and "untrained_attn_last" not in bundle.aux:
        stages.append(("untrained baseline", lambda p: A.untrained_attention(p, cfg)))
    if norm_attn and "norm_attn" not in bundle.aux:
        stages.append(("norm-weighted attention", lambda p: A.measure_norm_attribution(p, cfg)))
    if aux and "jitter_band" not in bundle.aux:
        stages.append(("run-to-run jitter band", lambda p: A.measure_jitter(p, cfg)))

    if stages and ps is None:
        ps = P.resolve(passages_path, roots, allow_synthetic=allow_synthetic, verbose=False)

    for label, fn in stages:
        print(f"\n{label} ...", flush=True)
        bundle.aux.update(fn(ps))
        if cache:
            bundle.save(cache)

    return bundle


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="m60482",
        description="Test published mechanisms against the Pythia-1.4B state-60482 effect.",
    )
    ap.add_argument("--passages", default=None, help="explicit path to passagen.json")
    ap.add_argument("--roots", nargs="*", default=list(DEFAULT_ROOTS))
    ap.add_argument("--output", default="runs", help="output directory")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--checkpoints", default=",".join(str(c) for c in C.CHECKPOINTS))
    ap.add_argument("--probes", nargs="*", default=[], help="subset of probes; default all")
    ap.add_argument("--no-attention", action="store_true", help="skip attention capture")
    ap.add_argument(
        "--no-aux",
        action="store_true",
        help="skip the truncation ladder, greedy continuations and name substitutions "
        "(context_dependence then cannot run, and it is the probe most likely to "
        "settle the question)",
    )
    ap.add_argument("--no-untrained", action="store_true", help="skip the untrained baseline")
    ap.add_argument("--no-norm-attn", action="store_true", help="skip norm-weighted attention")
    ap.add_argument(
        "--free-disk",
        action="store_true",
        help="delete each checkpoint's files after use; four fp32 revisions are ~22.6 GB "
        "and HuggingFace does not share blobs between them",
    )
    ap.add_argument(
        "--check-variant",
        action="store_true",
        help="fetch the anchor's Pile sample from both preshuffled corpora to confirm "
        "pythia-1.4b vs -deduped (~8 KB, needs network)",
    )
    ap.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="run on a meaningless passage set when none is found (pipeline test only)",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="synthetic passages, no model, no GPU: exercise probes on a fake bundle",
    )
    ap.add_argument("--reuse-bundle", default=None, help="path to an existing bundle .npz")
    args = ap.parse_args(argv)

    run_id = make_run_id(args.run_id)
    out_dir = Path(args.output) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.check_variant and not args.smoke:
        from . import pile

        print("identifying the model variant from the training data ...")
        try:
            variant = pile.identify_variant(verbose=True)
            print(f"variant: {variant}")
            if variant != C.MODEL_VARIANT:
                print(
                    f"  !! config says {C.MODEL_VARIANT}. The measurement would be "
                    "against the wrong suite."
                )
        except (OSError, RuntimeError) as exc:
            print(f"  could not check: {exc}")

    if args.smoke:
        from .synthetic_bundle import fake_bundle

        print("SMOKE RUN: synthetic bundle, no model loaded. Numbers are meaningless.")
        bundle = fake_bundle()
    else:
        cfg = C.RunConfig(
            checkpoints=tuple(int(x) for x in args.checkpoints.split(",")),
            dtype=args.dtype,
            device=args.device,
            top_k=args.top_k,
            capture_attention=not args.no_attention,
            output_dir=str(out_dir),
            run_id=run_id,
            probes=tuple(args.probes),
            extra={"free_disk": bool(args.free_disk)},
        )
        cache = Path(args.reuse_bundle) if args.reuse_bundle else out_dir / "bundle.npz"
        bundle = build_bundle(
            cfg,
            passages_path=args.passages,
            roots=args.roots,
            allow_synthetic=args.allow_synthetic,
            cache=cache,
            aux=not args.no_aux,
            untrained=not args.no_untrained,
            norm_attn=not args.no_norm_attn,
        )

        check = verify_against_reference(bundle)
        if check.get("checked"):
            status = "matches" if check["all_ok"] else "DIFFERS FROM"
            print(f"\nreproduction check: this run {status} the earlier measurement")
            for row in check["rows"]:
                flag = "ok" if row["ok"] else "!!"
                print(
                    f"  {flag} step{row['step']} {row['quantity']}: "
                    f"got {row['got']}, expected {row['expected']}"
                )
            if not check["all_ok"]:
                print(
                    "\n  A mismatch means this run is measuring a different object "
                    "(model variant, dtype or passage set). Probe results below are "
                    "about that object, not about the original finding.\n"
                )
                bundle.meta["reproduction_mismatch"] = [
                    r for r in check["rows"] if not r["ok"]
                ]

    print("\nrunning probes\n" + "-" * 60)
    results = registry.run(bundle, tuple(args.probes))

    paths = report.save(bundle, results, out_dir)
    adj = report.adjudicate(results)
    print("\n" + "=" * 60)
    print(adj["headline"])
    print("=" * 60)
    print(f"report:  {paths['report']}")
    print(f"results: {paths['results']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
