"""Tests that run without a GPU, a model download, or network access.

They cover the three things that would silently corrupt a real run: the passage set
being something other than the K8 set, a probe's decision rule not matching the numbers
it is handed, and the bundle losing data across a save/load cycle.
"""

from __future__ import annotations

import json
import os
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
    missing = set(registry.names()) - set(PAP.PROBE_TO_PAPERS)
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
    assert set(PAP.BY_KEY["sink_frozen"].probes) == {"sink_stability"}
    assert set(PAP.BY_KEY["attention_sinks"].probes) == {"attention_sink"}


# ---------------------------------------------------------------------------------
# Pile access (offline arithmetic; the fetches are marked and skipped by default)
# ---------------------------------------------------------------------------------

from m60482 import pile  # noqa: E402

network = pytest.mark.skipif(
    not os.environ.get("M60482_NETWORK_TESTS"),
    reason="set M60482_NETWORK_TESTS=1 to hit huggingface.co",
)


def test_corpus_arithmetic_is_exact():
    """The whole 4 KB shortcut rests on this being exact, not approximate."""
    assert pile.TOTAL_BYTES == 143_000 * 1024 * pile.SEQ_LEN * 2
    assert pile.TOTAL_BYTES % pile.ROW_BYTES == 0
    assert pile.N_ROWS == 143_000 * 1024
    assert pile.ROW_BYTES == 4098


def test_step_of_matches_the_frozen_anchor_step():
    assert pile.step_of(C.ANCHOR_GLOBAL_SAMPLE_INDEX) == C.ANCHOR_TRAIN_STEP
    assert C.CHECKPOINTS[1] < C.ANCHOR_TRAIN_STEP < C.CHECKPOINTS[2]


def test_shard_boundaries_are_not_row_aligned():
    """A row can straddle two shards, which is why fetch issues two requests."""
    assert pile.SHARD_BYTES[0] % pile.ROW_BYTES != 0
    shard, local = pile._locate(pile.SHARD_BYTES[0] - 10)
    assert (shard, local) == (0, pile.SHARD_BYTES[0] - 10)
    shard, local = pile._locate(pile.SHARD_BYTES[0])
    assert (shard, local) == (1, 0)


def test_out_of_range_index_is_rejected():
    with pytest.raises(IndexError):
        pile.fetch_training_row(pile.N_ROWS)
    with pytest.raises(IndexError):
        pile.fetch_training_row(-1)


def test_check_anchor_rejects_a_wrong_row():
    row = [0] * pile.SEQ_LEN
    with pytest.raises(AssertionError):
        pile.check_anchor(row)


def test_check_anchor_accepts_a_synthetic_matching_row():
    row = [0] * pile.SEQ_LEN
    for pos, tok in zip(C.NAME_POSITIONS, C.NAME_TOKEN_IDS):
        row[pos] = tok
    row[C.ANCHOR_T] = C.TARGET_TOKEN_ID
    row[C.SINK_POSITION_NEWLINE] = 187
    pile.check_anchor(row)
    ctx, target = pile.anchor_context_and_target(row)
    assert len(ctx) == C.ANCHOR_T
    assert target == C.TARGET_TOKEN_ID


def test_the_variant_is_recorded_as_resolved():
    assert C.MODEL_VARIANT == "pythia-1.4b"
    assert "deduped" not in C.MODEL_ID
    assert C.ANCHOR_IS_PILE_ROW_PREFIX is True


@network
def test_anchor_is_the_prefix_of_the_real_pile_sample():
    ctx, target = pile.rebuild_anchor()
    assert len(ctx) == C.ANCHOR_T
    assert target == C.TARGET_TOKEN_ID
    assert tuple(ctx[124:128]) == C.NAME_TOKEN_IDS


@network
def test_the_deduped_corpus_is_the_negative_control():
    row = pile.fetch_training_row(C.ANCHOR_GLOBAL_SAMPLE_INDEX, repo=pile.DEDUPED_REPO)
    with pytest.raises(AssertionError):
        pile.check_anchor(row)


@network
def test_identify_variant_picks_exactly_one():
    assert pile.identify_variant(verbose=False) == C.MODEL_VARIANT


# ---------------------------------------------------------------------------------
# module lookup for the QKV hook (pure attribute walking; no torch needed)
# ---------------------------------------------------------------------------------

from m60482 import aux as A  # noqa: E402


class _Fake:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_decoder_layers_found_under_gpt_neox():
    m = _Fake(gpt_neox=_Fake(layers=["a", "b"]))
    assert A._decoder_layers(m) == ["a", "b"]


def test_decoder_layers_found_under_alternative_names():
    assert A._decoder_layers(_Fake(model=_Fake(layers=[1]))) == [1]
    assert A._decoder_layers(_Fake(transformer=_Fake(h=[1, 2]))) == [1, 2]


def test_decoder_layers_raises_rather_than_guessing():
    with pytest.raises(RuntimeError, match="decoder layer list"):
        A._decoder_layers(_Fake(something_else=1))


def test_qkv_module_requires_the_fused_projection():
    """A split Q/K/V layout must raise: reading the wrong tensor would produce
    plausible numbers from the query projection."""
    fused = _Fake(attention=_Fake(query_key_value="QKV"))
    assert A._qkv_module(fused, 0) == "QKV"

    split = _Fake(attention=_Fake(q_proj="Q", k_proj="K", v_proj="V"))
    with pytest.raises(RuntimeError, match="fused query_key_value"):
        A._qkv_module(split, 3)

    with pytest.raises(RuntimeError, match="no attention submodule"):
        A._qkv_module(_Fake(mlp=1), 7)


# ---------------------------------------------------------------------------------
# polarity: two papers can predict opposite outcomes of one measurement
# ---------------------------------------------------------------------------------


def test_the_name_variant_measurement_is_read_both_ways():
    """The circuit account predicts the variants differ; the reconstruction account
    predicts they do not. Same probe, opposite readings."""
    assert PAP.BY_KEY["verbatim_circuits"].probes["name_variant_equivalence"] == 1
    assert PAP.BY_KEY["reconstruction_not_recollection"].probes["name_variant_equivalence"] == -1

    b = fake_bundle()
    adj = report.adjudicate(registry.run(b, progress=False))
    by_key = {p["key"]: p for p in adj["papers"]}
    assert by_key["verbatim_circuits"]["verdict"] == "REFUTED"
    assert by_key["reconstruction_not_recollection"]["verdict"] == "SUPPORTED"


def test_softmax_artifact_is_supported_when_the_probe_refutes():
    b = fake_bundle()
    r = registry.get("softmax_renormalization")(b)
    assert r.verdict == "REFUTED"
    adj = report.adjudicate(registry.run(b, progress=False))
    by_key = {p["key"]: p for p in adj["papers"]}
    assert by_key["softmax_artifact"]["verdict"] == "SUPPORTED"


def test_scope_failed_and_not_run_never_flip():
    """'The subject matter is absent' and 'not measured' are evidence for nobody."""
    assert report._apply_polarity("SCOPE_FAILED", -1) == "SCOPE_FAILED"
    assert report._apply_polarity("NOT_RUN", -1) == "NOT_RUN"
    assert report._apply_polarity("SUPPORTED", -1) == "REFUTED"
    assert report._apply_polarity("REFUTED", -1) == "SUPPORTED"
    assert report._apply_polarity("SUPPORTED", 1) == "SUPPORTED"


def test_softmax_probe_runs_before_the_mechanistic_ones():
    order = [p.name for p in registry.all_probes()]
    assert order.index("softmax_renormalization") < order.index("attention_sink")
    assert order.index("softmax_renormalization") < order.index("rank_attribution")


# ---------------------------------------------------------------------------------
# multiplicity and rank margins
# ---------------------------------------------------------------------------------


def test_multiplicity_ledger_reports_the_uncorrectable_floor():
    """57 comparisons need alpha = 0.00088; the floor with 40 controls is 0.024. No
    result in this design survives correction, and the report has to say so."""
    r = registry.get("multiplicity_ledger")(fake_bundle())
    e = r.evidence
    assert e["n_states"] == C.N_STATES
    assert e["n_boundaries"] == C.N_BOUNDARIES
    assert e["expected_by_chance"] == pytest.approx(C.N_STATES / C.N_BOUNDARIES)
    assert e["p_at_least_one_by_chance"] > 0.99
    assert e["attainable_p_floor"] == pytest.approx(C.MIN_ATTAINABLE_P)
    assert e["floor_survives_correction"] is False
    assert "corrected alpha" in r.cannot_conclude


def test_multiplicity_ledger_runs_before_the_mechanistic_probes():
    order = [p.name for p in registry.all_probes()]
    assert order.index("multiplicity_ledger") < order.index("attention_sink")


def test_rank_margins_expose_the_fragile_checkpoint():
    """The only checkpoint where the target held rank 2 is also the only one where its
    margin was marginal."""
    r = registry.get("rank_attribution")(fake_bundle())
    by_step = {m["checkpoint"]: m for m in r.evidence["margins"]}
    assert by_step[131000]["rank"] == 2
    assert by_step[131000]["nearest_gap"] == pytest.approx(0.00122, abs=1e-5)
    assert by_step[131000]["relative_nearest_gap"] < 0.02
    for step in (130000, 132000, 133000):
        assert by_step[step]["relative_nearest_gap"] > 0.1


def test_rank_resolution_needs_a_jitter_band():
    b = fake_bundle()
    b.aux.pop("jitter_band")
    r = registry.get("rank_attribution")(b)
    assert r.evidence["jitter_band"] is None
    assert "no jitter band has been measured" in r.summary
    assert r.evidence["unresolved_checkpoints"] == []


def test_a_large_jitter_band_marks_the_rank_unresolved():
    """In fp16 the jitter would swamp the 0.00122 margin, and the probe must say so."""
    b = fake_bundle()
    b.aux["jitter_band"] = 0.01
    r = registry.get("rank_attribution")(b)
    assert 131000 in r.evidence["unresolved_checkpoints"]
    assert "UNRESOLVED" in r.summary


# ---------------------------------------------------------------------------------
# sham anchor: a control relabelled as the anchor
# ---------------------------------------------------------------------------------


def sham_bundle(control: str = "kontrolle_00"):
    """A bundle in which the anchor's rows are a control's.

    Nothing happened to this passage at the exposure boundary, so it is the false-
    positive calibration the suite would otherwise lack: how many probes say the same
    thing about a passage that was never trained on as about the one that was?
    """
    b = fake_bundle()
    a, c = b.pi(C.ANCHOR), b.pi(control)
    for name in (
        "topk_ids", "topk_probs", "topk_logprobs", "target_p", "target_logp",
        "target_rank", "target_rank_in_topk", "entropy", "top1_ids", "top1_probs",
        "ctx_nll", "attn_last", "attn_received",
    ):
        arr = getattr(b, name)
        arr[:, a] = arr[:, c]
    b.target_ids[a] = b.target_ids[c]
    b.aux["ladder"][:, a] = b.aux["ladder"][:, c]
    b.aux["norm_attn"][:, a] = b.aux["norm_attn"][:, c]
    return b


def test_no_probe_crashes_on_a_sham_anchor():
    """A probe that raises on a degenerate passage returns no verdict at all, which is
    worse than returning the wrong one."""
    b = sham_bundle()
    failures = []
    for probe in registry.all_probes():
        try:
            r = probe(b)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{probe.name}: {type(exc).__name__}: {exc}")
            continue
        if r.verdict == "ERROR":
            failures.append(f"{probe.name}: ERROR verdict -- {r.summary}")
    assert not failures, "probes that fail on a sham anchor:\n  " + "\n  ".join(failures)


def test_anchor_specific_probes_read_the_anchor_row_not_a_remembered_one():
    """A coinciding verdict is not by itself a failure -- a control is also not
    memorized, so `memorization_entry` saying SCOPE_FAILED for both is correct. What
    would be a failure is the probe reporting the *anchor's numbers* for a sham, which
    would mean it is not reading the row it was handed."""
    real = registry.get("memorization_entry")(fake_bundle())
    sham = registry.get("memorization_entry")(sham_bundle())
    real_argmax = real.tables["argmax vs target, per checkpoint"][0]["p_argmax"]
    sham_argmax = sham.tables["argmax vs target, per checkpoint"][0]["p_argmax"]
    assert real_argmax == pytest.approx(C.REFERENCE_TOP1_P[130000], abs=1e-5)
    assert sham_argmax != pytest.approx(real_argmax, abs=1e-6)


def test_closed_slot_discriminates_the_anchor_from_a_sham():
    """The near-closed unit slot is a property of this passage's continuation, so a
    control relabelled as the anchor must not reproduce it."""
    real = registry.get("closed_slot")(fake_bundle())
    sham = registry.get("closed_slot")(sham_bundle())
    assert real.verdict == "SUPPORTED"
    assert real.evidence["slot_total_mean"] == pytest.approx(0.96728, abs=1e-4)
    assert sham.verdict != "SUPPORTED" or sham.evidence["slot_total_mean"] != pytest.approx(
        real.evidence["slot_total_mean"], abs=1e-4
    )


def test_corpus_descriptive_probes_are_expected_not_to_discriminate():
    """sink_stability and position_bias describe the whole passage set, so giving the
    same answer for a sham anchor is correct -- and their `cannot_conclude` has to own
    that rather than imply anchor-specificity."""
    b = sham_bundle()
    for name in ("sink_stability", "position_bias"):
        r = registry.get(name)(b)
        assert r.verdict in registry.VERDICTS
        assert r.cannot_conclude


def test_scope_checks_are_anchor_independent_by_construction():
    """A scope check asks whether a paper's subject matter is present in the *setup*,
    which does not depend on which passage is the anchor."""
    real = {r.probe: r.verdict for r in registry.run(fake_bundle(), progress=False)}
    sham = {r.probe: r.verdict for r in registry.run(sham_bundle(), progress=False)}
    for name in ("cliff_token_scope", "self_loop_scope"):
        assert real[name] == sham[name] == "SCOPE_FAILED"


# ---------------------------------------------------------------------------------
# the run-level gate
# ---------------------------------------------------------------------------------

from m60482 import run as R  # noqa: E402


def test_the_gate_stops_a_run_whose_anchor_is_not_the_pile_sample(monkeypatch):
    """The failure this catches is a wrong model variant, which nothing else can:
    the two repos have identical configs and identical tokenizer hashes."""
    monkeypatch.setattr(
        P, "verify_anchor_against_pile",
        lambda ps, **kw: {
            "checked": True, "context_matches": False, "target_matches": True,
            "first_differing_position": 17, "note": "x",
        },
    )
    with pytest.raises(RuntimeError, match="not state 60482"):
        R._gate_passages(P.synthetic(), strict=True)


def test_an_unreachable_check_does_not_block(monkeypatch, capsys):
    """Being offline is not evidence of a wrong passage set."""
    monkeypatch.setattr(
        P, "verify_anchor_against_pile",
        lambda ps, **kw: {"checked": False, "reason": "OSError: no network"},
    )
    R._gate_passages(P.synthetic(), strict=True)
    assert "skipped" in capsys.readouterr().out


def test_a_matching_anchor_passes(monkeypatch, capsys):
    monkeypatch.setattr(
        P, "verify_anchor_against_pile",
        lambda ps, **kw: {"checked": True, "context_matches": True, "target_matches": True},
    )
    R._gate_passages(P.synthetic(), strict=True)
    assert "variant confirmed" in capsys.readouterr().out


def test_a_reproduction_mismatch_is_carried_into_the_report():
    b = fake_bundle()
    b.meta["reproduction_mismatch"] = [
        {"step": 131000, "quantity": "p_target", "expected": 0.09087, "got": 0.5, "ok": False}
    ]
    md = report.markdown(b, registry.run(b, progress=False))
    assert "does not reproduce the earlier measurement" in md


def test_full_logits_are_declared_in_the_metadata():
    """A probe reading final_logits must be able to tell 'not captured' from 'empty'."""
    b = fake_bundle()
    assert b.meta["full_logits_captured"] is False
    assert b.final_logits.size == 0
    assert C.RunConfig().capture_full_logits is True
