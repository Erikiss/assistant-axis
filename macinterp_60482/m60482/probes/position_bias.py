"""Lost in the Middle at Birth: is the profile architectural rather than learned?

The one candidate paper that survived triage as even weakly applicable derives position
bias from causal attention plus residual connections, and finds the corresponding
influence profiles in *untrained* GPT-2 and Qwen2 -- before any training, and therefore
before any memorization.

The prediction that makes it testable here is sharp: if the attention profile at the
final position is set by the architecture and by where the delimiters fall, then every
passage should show the same profile shape, and the anchor should be unremarkable.  A
passage-specific, exposure-sensitive profile would be evidence against.

The comparison is made on position-independent features, because the 69 passages do not
share a token layout: for each passage, the profile is summarised by its concentration
(how much mass sits on its own top positions) and by the depth at which it stops
attending locally and locks onto a sink.  Those are comparable across passages;
raw position indices are not.
"""

from __future__ import annotations

import numpy as np

from .. import config as C
from .. import stats as S
from ..measure import Bundle
from ..registry import ProbeResult, register

DEEP_FROM = 3
RHO_SHARED = 0.7

RULE = (
    "For every passage compute (a) the layer at which the final query's argmax stops "
    "being within 8 positions of the end and locks onto a fixed earlier position, and "
    "(b) the deep-layer mass on its own top-2 positions. The profile is ARCHITECTURAL "
    "if the median pairwise Spearman correlation between passages' per-layer "
    f"concentration curves is >= {RHO_SHARED} AND the anchor's lock-on layer is inside "
    "the controls' range. SUPPORTED if both hold. REFUTED if the anchor's lock-on layer "
    "or concentration is outside the control range (control-referenced p <= 0.05 in "
    "either direction). INCONCLUSIVE otherwise."
)


def _lock_on_layer(prof: np.ndarray, near: int = 8) -> int:
    """First layer from which the argmax stays on one non-local position."""
    T = prof.shape[1]
    argmax = prof.argmax(axis=1)
    for l in range(prof.shape[0]):
        tail = argmax[l:]
        if np.all(tail == tail[0]) and (T - 1 - tail[0]) > near:
            return l
    return prof.shape[0]


@register(
    "position_bias",
    paper="Lost in the Middle at Birth: An Exact Theory of Transformer Position Bias (2026)",
    hypothesis="The attention profile follows from causal attention plus residuals, not from training on this passage.",
    question="Do all 69 passages share one profile shape, and is the anchor's unremarkable?",
    decision_rule=RULE,
    order=60,
)
def position_bias(bundle: Bundle) -> ProbeResult:
    if not bundle.meta.get("attention_captured"):
        return ProbeResult(
            probe="position_bias", paper="Lost in the Middle at Birth (2026)",
            hypothesis="The profile is architectural.",
            question="Do all passages share one profile shape?",
            verdict="NOT_RUN", decision_rule=RULE,
            summary="Attention was not captured in this bundle.",
            cannot_conclude="Nothing; the measurement was not made.",
        )

    step = C.EXPOSURE_BOUNDARY[1] if C.EXPOSURE_BOUNDARY[1] in bundle.checkpoints else bundle.checkpoints[-1]
    c_i = bundle.ci(step)

    def concentration_curve(p_i: int) -> np.ndarray:
        """Per-layer mass on that layer's own top-2 positions -- shape-comparable
        across passages with different token layouts."""
        prof = bundle.attn_last[c_i, p_i]                      # [layer, T]
        part = np.partition(prof, -2, axis=1)[:, -2:]
        return part.sum(axis=1)

    names = list(bundle.names)
    curves = np.stack([concentration_curve(i) for i in range(len(names))])

    # median pairwise Spearman between passages
    def rank(a: np.ndarray) -> np.ndarray:
        order = a.argsort()
        r = np.empty_like(order, dtype=float)
        r[order] = np.arange(len(a), dtype=float)
        return r

    rc = np.stack([rank(c) for c in curves])
    rc = rc - rc.mean(axis=1, keepdims=True)
    denom = np.sqrt((rc ** 2).sum(axis=1))
    corr = (rc @ rc.T) / np.outer(denom, denom)
    iu = np.triu_indices(len(names), k=1)
    median_rho = float(np.median(corr[iu]))

    a_i = bundle.pi(C.ANCHOR)
    lock_anchor = _lock_on_layer(bundle.attn_last[c_i, a_i])
    lock_controls = [_lock_on_layer(bundle.attn_last[c_i, bundle.pi(n)]) for n in bundle.control_names]

    conc_anchor = float(curves[a_i, DEEP_FROM:].mean())
    conc_controls = [float(curves[bundle.pi(n), DEEP_FROM:].mean()) for n in bundle.control_names]

    lock_hi = S.compare_to_controls("lock_on_layer", lock_anchor, lock_controls, direction="greater")
    lock_lo = S.compare_to_controls("lock_on_layer", lock_anchor, lock_controls, direction="less")
    conc_hi = S.compare_to_controls("deep_concentration", conc_anchor, conc_controls, direction="greater")
    conc_lo = S.compare_to_controls("deep_concentration", conc_anchor, conc_controls, direction="less")

    anchor_inside = all(
        c.p_one_sided > 0.05 for c in (lock_hi, lock_lo, conc_hi, conc_lo)
    )
    shared = median_rho >= RHO_SHARED

    rows = [
        {"layer": l,
         "anchor_concentration": float(curves[a_i, l]),
         "control_median": float(np.median(curves[[bundle.pi(n) for n in bundle.control_names], l])),
         "control_min": float(curves[[bundle.pi(n) for n in bundle.control_names], l].min()),
         "control_max": float(curves[[bundle.pi(n) for n in bundle.control_names], l].max())}
        for l in range(curves.shape[1])
    ]

    # The comparison the paper is actually about: is the profile there before training?
    untrained = (bundle.aux or {}).get("untrained_attn_last")
    untrained_rho = None
    untrained_rows = []
    if untrained is not None:
        u = np.asarray(untrained)
        u_names = list((bundle.aux or {}).get("untrained_names", names))
        # Pair each passage with its OWN untrained curve. Keeping the two index lists
        # explicit matters: a partially-measured untrained sweep would otherwise
        # silently correlate passage i's trained curve with passage j's untrained one.
        paired = [(names.index(nm), u_names.index(nm)) for nm in names if nm in u_names]
        if paired:
            t_idx = [t for t, _ in paired]
            u_idx = [v for _, v in paired]
            u_curves = np.stack([
                np.partition(u[v], -2, axis=1)[:, -2:].sum(axis=1) for v in u_idx
            ])
            ru = np.stack([rank(c) for c in u_curves])
            ru = ru - ru.mean(axis=1, keepdims=True)
            rt = rc[t_idx]
            num = (rt * ru).sum(axis=1)
            den = np.sqrt((rt ** 2).sum(axis=1) * (ru ** 2).sum(axis=1))
            per_passage = num / np.where(den == 0, np.nan, den)
            untrained_rho = float(np.nanmedian(per_passage))
            anchor_row = t_idx.index(a_i) if a_i in t_idx else None
            untrained_rows = [
                {"layer": l,
                 "trained_concentration": float(curves[a_i, l]),
                 "untrained_concentration": (
                     float(u_curves[anchor_row, l]) if anchor_row is not None else float("nan")
                 )}
                for l in range(curves.shape[1])
            ]

    if shared and anchor_inside:
        verdict = "SUPPORTED"
        summary = (
            f"All {len(names)} passages share one profile shape (median pairwise "
            f"Spearman {median_rho:.3f} over the per-layer concentration curve), and the "
            f"anchor is unremarkable within the controls (lock-on layer {lock_anchor} vs "
            f"control median {lock_hi.control_median:.0f}; deep concentration "
            f"{conc_anchor:.3f} vs {conc_hi.control_median:.3f}). The profile is a "
            "property of the architecture and the layout, not of this passage."
        )
    elif not anchor_inside:
        verdict = "REFUTED"
        worst = min((lock_hi, lock_lo, conc_hi, conc_lo), key=lambda c: c.p_one_sided)
        summary = (
            f"The anchor is atypical: {worst.statistic} = {worst.anchor:.3f} against a "
            f"control median of {worst.control_median:.3f} "
            f"(p_{worst.direction} = {worst.p_one_sided:.4f}). A purely architectural "
            "profile would not single this passage out."
        )
    else:
        verdict = "INCONCLUSIVE"
        summary = (
            f"The anchor sits inside the control range, but the passages do not share a "
            f"single profile shape (median pairwise Spearman {median_rho:.3f} < "
            f"{RHO_SHARED}); the architectural account is not confirmed."
        )

    return ProbeResult(
        probe="position_bias",
        paper="Lost in the Middle at Birth: An Exact Theory of Transformer Position Bias (2026)",
        hypothesis="The attention profile follows from causal attention plus residuals, not from training on this passage.",
        question="Do all 69 passages share one profile shape, and is the anchor's unremarkable?",
        verdict=verdict,
        decision_rule=RULE,
        summary=summary,
        evidence={
            "checkpoint": step,
            "median_pairwise_spearman": median_rho,
            "anchor_lock_on_layer": lock_anchor,
            "anchor_deep_concentration": conc_anchor,
            "lock_vs_controls_greater": lock_hi.as_dict(),
            "lock_vs_controls_less": lock_lo.as_dict(),
            "concentration_vs_controls_greater": conc_hi.as_dict(),
            "concentration_vs_controls_less": conc_lo.as_dict(),
            "anchor_inside_control_range": bool(anchor_inside),
            "untrained_baseline_measured": untrained is not None,
            "median_trained_vs_untrained_spearman": untrained_rho,
        },
        tables={
            f"per-layer attention concentration, step{step}": rows,
            **({"trained vs untrained concentration (anchor)": untrained_rows}
               if untrained_rows else {}),
        },
        cannot_conclude=(
            (
                "The untrained baseline was measured, so 'the profile predates "
                f"training' is testable here (median trained-vs-untrained Spearman "
                f"{untrained_rho:.3f}). What it still cannot do is attribute the target "
                "probability: a profile that is architectural explains where attention "
                "goes, not what the model predicts."
            )
            if untrained_rho is not None
            else (
                "The paper's claim is about untrained networks and this probe saw only "
                "trained checkpoints. A shared profile across passages is consistent "
                "with the architectural account but equally with every passage having "
                "learned the same thing. Run m60482.aux.untrained_attention to get the "
                "baseline the paper is actually stated over -- one sweep, no download."
            )
        ),
        caveats=[
            "Concentration is used instead of raw position profiles because the 69 "
            "passages have different token layouts; two passages can share a mechanism "
            "and put their sinks at different indices.",
            S.selection_note(False),
        ],
    )
