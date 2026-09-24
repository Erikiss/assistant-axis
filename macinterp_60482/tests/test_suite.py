"""Tests that run without a GPU, a model download, or network access.

They cover the three things that would silently corrupt a real run: the passage set
being something other than the K8 set, a probe's decision rule not matching the numbers
it is handed, and the bundle losing data across a save/load cycle.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from m60482 import config as C
from m60482 import papers as PAP
from m60482 import passages as P
from m60482 import registry, report, stats as S
from m60482.measure import Bundle
from m60482.synthetic_bundle import fake_bundle

import m60482.probes  # noqa: F401 -- registers the probes


# ---------------------------------------------------------------------------------
# passage set
# ---------------------------------------------------------------------------------


def test_synthetic_passages_have_k8_structure():
    ps = P.synthetic()
    assert len(ps.passages) == C.N_PASSAGES
    assert len(ps.states) == C.N_STATES
    assert len(ps.controls) == C.N_CONTROLS
    assert len(ps.variants) == C.N_VARIANTS
    assert P._structure_ok(ps.passages) is None


def test_anchor_facts_are_enforced():
    ps = P.synthetic()
    a = ps.anchor
    assert a.T == C.ANCHOR_T
    assert a.target == C.TARGET_TOKEN_ID
    assert tuple(a.ids[124:128]) == C.NAME_TOKEN_IDS


def test_wrong_structure_is_rejected():
    ps = P.synthetic()
    broken = ps.passages[:-1]           # 39 controls instead of 40
    assert P._structure_ok(broken) is not None


def test_canonical_hash_is_order_independent():
    ps = P.synthetic()
    a = P.canonical_hash(ps.passages)
    b = P.canonical_hash(tuple(reversed(ps.passages)))
    assert a == b


def test_non_canonical_file_is_marked_reconstructed():
    ps = P.synthetic()
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "passagen.json"
        path.write_text(json.dumps({"passagen": [
            {"name": p.name, "art": p.kind, "ids": list(p.ids), "ziel": p.target, "T": p.T}
            for p in ps.passages
        ]}))
        loaded = P.load_file(path)
    assert loaded.provenance == "reconstructed"
    assert not loaded.trustworthy
    assert "DEGRADED" in loaded.warning()


def test_resolve_refuses_synthetic_by_default():
    with pytest.raises(FileNotFoundError):
        P.resolve(roots=["/nonexistent"], allow_synthetic=False, verbose=False)


# ---------------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------------


def test_control_p_has_the_documented_floor():
    c = S.compare_to_controls("x", 1e9, list(range(C.N_CONTROLS)))
    assert c.p_one_sided == pytest.approx(1 / (C.N_CONTROLS + 1))
    assert c.p_floor == pytest.approx(C.MIN_ATTAINABLE_P)


def test_control_p_is_never_zero():
    c = S.compare_to_controls("x", float("inf"), [0.0] * 40)
    assert c.p_one_sided > 0


def test_direction_matters():
    hi = S.compare_to_controls("x", 0.0, [1.0] * 40, direction="greater")
    lo = S.compare_to_controls("x", 0.0, [1.0] * 40, direction="less")
    assert hi.p_one_sided == pytest.approx(1.0)
    assert lo.p_one_sided == pytest.approx(1 / 41)


def test_boundary_contrast_flags_outside_range():
    c = S.contrast_boundaries("d", {
        (130000, 131000): -0.003,
        (131000, 132000): 0.014,
        (132000, 133000): 0.004,
    })
    assert c.outside_placebo_range
    assert c.n_placebo == 2


def test_total_variation_bounds():
    p = np.array([0.5, 0.5])
    q = np.array([1.0, 0.0])
    assert S.total_variation(p, q) == pytest.approx(0.5)
    assert S.total_variation(p, p) == pytest.approx(0.0)


# ---------------------------------------------------------------------------------
# bundle
# ---------------------------------------------------------------------------------


def test_bundle_round_trips_including_aux():
    b = fake_bundle()
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "bundle.npz"
        b.save(path)
        back = Bundle.load(path)
    assert back.names == b.names
    assert np.allclose(back.target_p, b.target_p)
    assert np.allclose(back.attn_last, b.attn_last)
    assert np.allclose(back.aux["ladder"], b.aux["ladder"])
    assert back.aux["greedy_tokens"] == b.aux["greedy_tokens"]
    assert back.meta["provenance"] == "synthetic"


def test_synthetic_bundle_reproduces_the_measured_anchor():
    """The whole point of the synthetic bundle: the anchor's headline numbers are real."""
    b = fake_bundle()
    a = b.pi(C.ANCHOR)
    for step, expected in C.REFERENCE_TARGET_P.items():
        assert b.target_p[b.ci(step), a] == pytest.approx(expected, abs=1e-5)
    for step, expected in C.REFERENCE_TARGET_RANK.items():
        assert int(b.target_rank[b.ci(step), a]) == expected
    for step, expected in C.REFERENCE_TOP1_P.items():
        assert b.target_p[b.ci(step), a] < expected   # target never wins
        assert int(b.top1_ids[b.ci(step), a]) == C.TOP1_TOKEN_ID


def test_attention_rows_are_distributions():
    b = fake_bundle()
    sums = b.attn_last.sum(axis=-1)
    assert np.allclose(sums, 1.0, atol=1e-4)


def test_synthetic_attention_matches_the_measured_argmax_profile():
    b = fake_bundle()
    prof = b.attn_profile(C.ANCHOR, 131000)
    for layer, (pos, _) in C.REFERENCE_ATTN_ARGMAX_131000.items():
        assert int(prof[layer].argmax()) == pos


# ---------------------------------------------------------------------------------
# probes
# ---------------------------------------------------------------------------------


def test_every_probe_returns_a_valid_result():
    b = fake_bundle()
    results = registry.run(b, progress=False)
    assert len(results) == len(registry.names())
    for r in results:
        assert r.verdict in registry.VERDICTS
        assert r.verdict != "ERROR", f"{r.probe}: {r.summary}"
        assert r.cannot_conclude, f"{r.probe} has no stated limits"
        assert r.decision_rule
        assert r.summary


def test_probes_are_ordered_premise_first():
    order = [p.name for p in registry.all_probes()]
    assert order[0] == "memorization_entry"
    assert order.index("noise_floor") < order.index("attention_sink")
    assert order.index("name_variant_equivalence") < order.index("position_bias")


def test_probe_result_requires_stated_limits():
    with pytest.raises(ValueError, match="cannot_conclude"):
        registry.ProbeResult(
            probe="x", paper="y", hypothesis="h", question="q",
            verdict="SUPPORTED", decision_rule="r", summary="s",
        )


def test_probe_result_rejects_unknown_verdict():
    with pytest.raises(ValueError, match="verdict"):
        registry.ProbeResult(
            probe="x", paper="y", hypothesis="h", question="q",
            verdict="MAYBE", decision_rule="r", summary="s", cannot_conclude="c",
        )


def test_memorization_entry_fails_on_the_real_numbers():
    """The argmax is ' ph' at every checkpoint, so the memorization frame cannot start."""
    b = fake_bundle()
    r = registry.get("memorization_entry")(b)
    assert r.verdict == "SCOPE_FAILED"
    assert r.evidence["argmax_ever_equals_target"] is False


def test_rank_attribution_catches_the_direction_conflict():
    """The target's probability rose while its rank fell -- the statistic is measuring
    the competitor."""
    b = fake_bundle()
    r = registry.get("rank_attribution")(b)
    assert r.verdict == "REFUTED"
    assert r.evidence["direction_conflict"] is True
    assert r.evidence["rank_before"] == 2 and r.evidence["rank_after"] == 3


def test_name_variants_detect_equivalence():
    b = fake_bundle()
    r = registry.get("name_variant_equivalence")(b)
    assert r.verdict == "REFUTED"
    assert abs(r.evidence["z"]) < 1


def test_probes_handle_a_bundle_without_attention():
    b = fake_bundle()
    b.meta["attention_captured"] = False
    for name in ("attention_sink", "position_bias"):
        assert registry.get(name)(b).verdict == "NOT_RUN"


def test_probes_handle_a_bundle_without_aux():
    b = fake_bundle()
    b.aux = {}
    assert registry.get("context_dependence")(b).verdict == "NOT_RUN"


# ---------------------------------------------------------------------------------
# adjudication + report
# ---------------------------------------------------------------------------------


def test_every_probe_maps_to_a_paper():
    missing = set(registry.names()) - set(PAP.PROBE_TO_PAPER)
    assert not missing, f"probes with no paper: {sorted(missing)}"


def test_premise_checks_are_excluded_from_the_headline():
    b = fake_bundle()
    adj = report.adjudicate(registry.run(b, progress=False))
    assert "null_calibration" not in adj["unlisted_supported"]
    assert adj["premise_checks"]


def test_all_six_listed_papers_are_adjudicated():
    b = fake_bundle()
    adj = report.adjudicate(registry.run(b, progress=False))
    listed = {p["key"] for p in adj["papers"] if p["on_user_list"]}
    assert listed == {p.key for p in PAP.on_user_list()}


def test_report_renders_and_names_the_provenance():
    b = fake_bundle()
    md = report.markdown(b, registry.run(b, progress=False))
    assert "SYNTHETIC BUNDLE" in md
    assert "Verdict per candidate paper" in md
    assert "Cannot conclude" in md
    assert str(C.MIN_ATTAINABLE_P)[:6] in md


def test_report_writes_both_files():
    b = fake_bundle()
    results = registry.run(b, progress=False)
    with tempfile.TemporaryDirectory() as d:
        paths = report.save(b, results, d)
        assert paths["report"].exists()
        payload = json.loads(paths["results"].read_text())
    assert payload["adjudication"]["n_probes"] == len(results)
    assert "vocab" not in payload["meta"]


# ---------------------------------------------------------------------------------
# untrained baseline (the comparison the position-bias paper is actually stated over)
# ---------------------------------------------------------------------------------


def _with_untrained(b, shuffle: bool = False):
    """Attach an untrained sweep to a bundle, optionally in a different passage order."""
    rng = np.random.default_rng(1)
    prof = rng.random((len(b.names), C.N_LAYERS, C.ANCHOR_T)).astype(np.float32)
    prof /= prof.sum(axis=-1, keepdims=True)
    names = list(b.names)
    if shuffle:
        order = list(rng.permutation(len(names)))
        prof = prof[order]
        names = [names[i] for i in order]
    b.aux["untrained_attn_last"] = prof
    b.aux["untrained_names"] = names
    return b


def test_position_bias_uses_the_untrained_baseline_when_present():
    b = _with_untrained(fake_bundle())
    r = registry.get("position_bias")(b)
    assert r.evidence["untrained_baseline_measured"] is True
    assert r.evidence["median_trained_vs_untrained_spearman"] is not None
    assert "untrained baseline was measured" in r.cannot_conclude


def test_position_bias_pairs_passages_with_their_own_untrained_curve():
    """Order-independence: shuffling the untrained sweep must not change the result."""
    a = registry.get("position_bias")(_with_untrained(fake_bundle()))
    c = registry.get("position_bias")(_with_untrained(fake_bundle(), shuffle=True))
    assert a.evidence["median_trained_vs_untrained_spearman"] == pytest.approx(
        c.evidence["median_trained_vs_untrained_spearman"]
    )


def test_position_bias_says_what_is_missing_without_the_baseline():
    r = registry.get("position_bias")(fake_bundle())
    assert r.evidence["untrained_baseline_measured"] is False
    assert "untrained_attention" in r.cannot_conclude


# ---------------------------------------------------------------------------------
# norm-weighted attribution (why the attention null is not yet readable)
# ---------------------------------------------------------------------------------


def test_norm_attribution_detects_an_evaporating_sink():
    b = fake_bundle()
    r = registry.get("norm_attribution")(b)
    assert r.verdict == "SUPPORTED"
    assert r.evidence["evaporates"] is True
    assert r.evidence["retained_fraction"] < 0.5
    # the sink positions are the ones the raw profile picked out
    assert C.SINK_POSITION_NEWLINE in r.evidence["raw_top2_positions"]


def test_norm_attribution_is_not_run_without_the_measurement():
    b = fake_bundle()
    b.aux.pop("norm_attn")
    r = registry.get("norm_attribution")(b)
    assert r.verdict == "NOT_RUN"
    assert "not measured" in r.cannot_conclude


def test_attention_sink_warns_that_weight_is_not_attribution():
    b = fake_bundle()
    r = registry.get("attention_sink")(b)
    assert any("norm_attribution" in c for c in r.caveats)


def test_norm_attribution_runs_before_the_head_level_probes():
    order = [p.name for p in registry.all_probes()]
    assert order.index("attention_sink") < order.index("norm_attribution")
    assert order.index("norm_attribution") < order.index("focus_directions")


# ---------------------------------------------------------------------------------
# sink stability (the frozen-profile reading of the attention null)
# ---------------------------------------------------------------------------------


def test_synthetic_attention_shift_matches_the_measured_scale():
    """The synthetic bundle must land near the measured TV distances, or sink_stability
    would be exercised without being tested."""
    b = fake_bundle()
    r = registry.get("sink_stability")(b)
    per = r.evidence["per_boundary"]
    for boundary in ((130000, 131000), (131000, 132000)):
        key = f"{boundary[0]}->{boundary[1]}"
        anchor_ref, ctrl_ref, _, _, _ = C.REFERENCE_ATTENTION_SHIFT[boundary]
        assert per[key]["anchor_tv"] == pytest.approx(anchor_ref, abs=3e-3)
        assert per[key]["control_median"] == pytest.approx(ctrl_ref, abs=6e-3)


def test_sink_stability_reproduces_the_anchor_below_every_control_result():
    b = fake_bundle()
    r = registry.get("sink_stability")(b)
    assert r.verdict == "SUPPORTED"
    assert r.evidence["frozen"] is True
    assert r.evidence["anchor_below_every_control_at_every_boundary"] is True
    for v in r.evidence["per_boundary"].values():
        assert v["p_greater"] == pytest.approx(1.0)


def test_sink_stability_refutes_when_profiles_actually_move():
    """If attention does move at this spacing, the null is a real null."""
    b = fake_bundle()
    rng = np.random.default_rng(7)
    noisy = rng.random(b.attn_last.shape).astype(np.float32)
    b.attn_last = noisy / noisy.sum(axis=-1, keepdims=True)
    r = registry.get("sink_stability")(b)
    assert r.verdict == "REFUTED"
    assert r.evidence["frozen"] is False


def test_sink_frozen_paper_is_judged_by_its_own_probe():
    assert PAP.BY_KEY["sink_frozen"].probes == ("sink_stability",)
    assert PAP.BY_KEY["attention_sinks"].probes == ("attention_sink",)
