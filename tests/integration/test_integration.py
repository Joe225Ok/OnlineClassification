"""Integration test: every pipeline script, unchanged, on a crafted archive with the edge cases of the real data
(tests/integration/edge_case_data.py), in a temporary folder (see tests/integration/integration_support.py for the only
differences from a real run). Three kinds of checks:

  1. Hand-derived values: what the data stages must produce, worked out from the dataset's specification (packets,
     flows, segments and their labels, the time split, the conflicts Track B drops, the held-out groups), never from a
     run of the code.
  2. Invariants: rules every model stage must satisfy whatever the numbers are (targets met where met, coverage is the
     share answered, uncertainty parts add up, infer.py reproduces stage 3, ...).
  3. Reference values: model outputs compared, within a tolerance, with
     tests/integration/reference/integration_reference_<system>.json (one per operating system), recorded from the
     agreed code by tests/integration/record_reference.py. They show that results have not changed, not that they are
     right; re-record only after an agreed change to the pipeline, never to make this test pass.
     Seven of them are machine-sensitive (integration_support.MACHINE_SENSITIVE): with SKIP_MACHINE_SENSITIVE=1, as in
     CI, they are skipped with the reason shown; otherwise they are compared like the others.

Two expected failures record known behaviour of the pipeline that was deliberately not changed (see their reasons).
Takes about a minute.
"""
import json
import math
import os

import joblib
import numpy as np
import pandas as pd
import pytest
import torch

import edge_case_data as spec_data
import kitsune_lib as kl
import synthetic_kitsune
from integration_support import MACHINE_SENSITIVE, NO_TESTBED, REFERENCE, collect, isolated_pipeline

KNOWN = [c for c in kl.CAPTURES if c not in kl.OOD_CAPTURES]


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    with isolated_pipeline(tmp_path_factory.mktemp("integration"), spec_data.write_archive) as p:
        p.run_everything()
        yield p


# --- Hand-derived values, from the specification ---------------------------------------------------------------------

def chunks(labels):
    """A flow's segments as lists of labels: consecutive pieces of L packets."""
    return [labels[i:i + kl.L] for i in range(0, len(labels), kl.L)]


def expected_segments(capture, spec):
    return [seg for fl in spec[capture] for seg in chunks(fl.labels)]


def split_counts(n):
    """How many of n segments (in time order) fall in training, validation and test: rank (j + 0.5) / n against 0.6, 0.8."""
    ranks = [(j + 0.5) / n for j in range(n)]
    return {"train": sum(r < 0.6 for r in ranks), "val": sum(0.6 <= r < 0.8 for r in ranks), "test": sum(r >= 0.8 for r in ranks)}


def test_every_packet_is_parsed_with_its_label(run):
    for capture, folder in kl.CAPTURES.items():
        p = pd.read_parquet(f"data/packets/{folder}.parquet")
        labels = [lab for fl in run.spec[capture] for lab in fl.labels]
        assert len(p) == len(labels), capture
        assert p["label"].sum() == sum(labels), capture


def test_frames_cut_to_200_bytes_are_sized_from_the_ip_length(run):
    p = pd.read_parquet("data/packets/os_scan.parquet")
    tcp = p[p["ip_proto"] == 6]
    assert tcp["frame_len"].max() > spec_data.SNAP  # data packets: IP length + 14, beyond the 200 bytes kept
    ipv6 = p[p["ethertype"] == 0x86DD]
    assert len(ipv6) == 10 and (ipv6["ttl"] == 255).all()  # hop limit read as TTL
    assert set(p["ethertype"]) == {0, 0x0800, 0x86DD, 0x88CC}  # 802.3 frame and unparseable frame: EtherType 0


def test_flows_and_segments_table(run):
    table = pd.read_csv("data/flows_vs_segments.csv", index_col=0)
    for capture in kl.CAPTURES:
        flows = run.spec[capture]
        segs = expected_segments(capture, run.spec)
        row = table.loc[capture]
        attack_flows = [fl for fl in flows if max(fl.labels) == 1]
        assert row["flows"] == len(flows)
        assert row["flows, attack"] == len(attack_flows)
        assert row["flows, benign"] == len(flows) - len(attack_flows)
        assert row["largest flow, packets"] == max(len(fl.labels) for fl in flows)
        assert row["packets in first L"] == sum(min(len(fl.labels), kl.L) for fl in flows)
        assert row["attack flows benign in first L"] == sum(max(fl.labels[:kl.L]) == 0 for fl in attack_flows)
        assert row["segments"] == len(segs)
        assert row["segments, attack"] == sum(max(s) for s in segs)
        assert row["segments, benign"] == sum(max(s) == 0 for s in segs)


def test_segments_have_their_lengths_and_labels(run):
    segments = pd.read_parquet("data/segments.parquet")
    for capture in kl.CAPTURES:
        segs = expected_segments(capture, run.spec)
        got = segments[segments["capture"] == capture]
        assert sorted(got["length"]) == sorted(len(s) for s in segs)
        assert sorted(zip(got["length"], got["malicious"])) == sorted((len(s), sum(s)) for s in segs)
    X = np.load("data/sequences.npy", mmap_mode="r")
    assert (np.asarray(X[:, :, kl.MASK_IDX]).sum(axis=1) == segments["length"].to_numpy()).all()


def test_track_a_time_split(run):
    a = kl.read_track("A")
    sizes = run.result("track_sizes.csv").set_index(["track", "split"])
    total = {"train": 0, "val": 0, "test": 0}
    for capture in KNOWN:
        segs = expected_segments(capture, run.spec)
        for label in (0, 1):
            n = sum(max(s) == label for s in segs)
            got = a[(a["capture"] == capture) & (a["attack"] == label)]["split"].value_counts()
            for part, count in split_counts(n).items():
                assert got.get(part, 0) == count, (capture, label, part)
                total[part] += count
    for part, count in total.items():
        assert sizes.loc[("A", part), "segments"] == count


def test_track_b_drops_exactly_the_conflicting_probes(run):
    # Every SYN probe (attack, and the three benign copies) has the same header sequence: one sequence with both labels,
    # holding every probe segment of the seven known captures. Under "all inputs", only attack probe 0 and the copy
    # with its TTL and ports conflict (two per capture); TTL tells the other two copies apart.
    probes = sum(fl.name == "probe" for c in KNOWN for fl in run.spec[c])
    assert probes == (spec_data.ATTACK_PROBES + 3) * len(KNOWN)
    drops = run.result("track_b_drops.csv").iloc[0]
    assert drops["segments dropped from Track B"] == probes
    assert drops["of which also in conflict under all inputs"] == 2 * len(KNOWN)
    assert drops["of which TTL, destination port or IAT would separate"] == probes - 2 * len(KNOWN)
    assert drops["known segments"] == sum(len(expected_segments(c, run.spec)) for c in KNOWN)
    review = run.result("conflicts_for_review.csv")
    assert len(review) == 1
    assert review.iloc[0][["copies", "captures", "benign", "attack", "length"]].tolist() == [
        probes, len(KNOWN), 3 * len(KNOWN), spec_data.ATTACK_PROBES * len(KNOWN), 2]
    assert len(kl.read_track("B")) == len(kl.read_track("A")) - probes


def test_diagnostics_count_one_conflicting_sequence_per_key(run):
    d = run.result("diagnostics.csv").set_index(["key", "k"])
    assert d.loc[("headers", kl.L), "conflicting sequences"] == 1
    assert d.loc[("headers", kl.L), "samples in conflict"] == (spec_data.ATTACK_PROBES + 3) * len(KNOWN)
    assert d.loc[("all inputs", kl.L), "conflicting sequences"] == 1
    assert d.loc[("all inputs", kl.L), "samples in conflict"] == 2 * len(KNOWN)


def test_held_out_groups(run):
    held = pd.read_parquet("data/tracks/held_out.parquet")
    for group, capture, names in [("Mirai", "Mirai", None), ("ARP MitM (ARP)", "ARP MitM", {"arp"}),
                                  ("ARP MitM (relayed)", "ARP MitM", {"stream", "query"})]:
        segs = [s for fl in run.spec[capture] if names is None or fl.name in names for s in chunks(fl.labels)]
        got = held[held["group"] == group]
        assert len(got) == len(segs), group
        assert got["attack"].sum() == sum(max(s) for s in segs), group
    assert held.loc[held["group"] == "ARP MitM (relayed)", "attack"].sum() > 0  # relayed attack after the midpoint
    assert set(held["capture"]) == set(kl.OOD_CAPTURES)


@pytest.mark.xfail(strict=True, reason=(
    "s0_prepare.py counts a gap as 'not whole microseconds' when it is more than 1 ns from a whole number of "
    "microseconds, but timestamps near 1.5e9 s held in float64 are only resolved to about 240 ns, so whole-microsecond "
    "gaps are counted too (the real captures store microseconds: their counts of 93-97% are this artifact). The "
    "'all inputs' key itself is unaffected (it rounds to whole microseconds). Not changed in the pipeline."))
def test_iat_resolution_counts_no_gap_when_timestamps_are_whole_microseconds(run):
    iat = run.result("iat_resolution.csv").set_index("capture")
    for capture in kl.CAPTURES:
        assert iat.loc[capture, "packets"] == sum(len(fl.labels) for fl in run.spec[capture])
        assert iat.loc[capture, "gaps not whole microseconds"] == 0  # every timestamp is a whole microsecond


# --- Invariants of the model stages ----------------------------------------------------------------------------------

def test_s2_chooses_the_size_with_the_lowest_validation_loss(run):
    sizes = run.result("A_gru_size.csv", index_col=0)
    selection = run.json("A_selection.json")
    assert selection["arch"] == sizes["validation loss, all"].idxmin()
    assert not set(selection["final"][NO_TESTBED]) & set(kl.TESTBED_GROUPS)


def test_s3_units_belong_to_their_models_and_meet_their_targets(run):
    for role in ("main", NO_TESTBED):
        name = run.model_name(role)
        unit = joblib.load(kl.MODEL_DIR / f"{name}_unit.joblib")
        kl.check_unit(unit, torch.load(kl.MODEL_DIR / f"{name}.pt", weights_only=True)["state"], name)
    th = run.result("A_thresholds.csv")
    for flag, col, target, tau, never in [("high target met", "block precision", kl.TARGET_HIGH, "tau_attack", 2.0),
                                          ("low target met", "Low precision", kl.TARGET_LOW, "tau_benign", -1.0)]:
        assert (th.loc[th[flag], col] >= target).all()  # on the validation half the cut-off was tuned on
        assert (th.loc[~th[flag], tau] == never).all()  # target not met: that answer is never given


def test_s3_coverage_is_the_share_answered_high_or_low(run):
    p = run.result("A_pipeline.csv")
    answered = p[["attack: High", "attack: Low", "benign: High", "benign: Low"]].sum(axis=1) / p["segments"]
    assert np.allclose(p["coverage"], answered)
    shares = p[[f"{c}: {a}" for c in ("attack", "benign") for a in kl.ANSWERS]].sum(axis=1)
    assert (shares == p["segments"]).all()  # every segment gets exactly one answer


def test_s3_ceilings_bound_their_models_and_the_timed_path_is_the_evaluated_one(run):
    forced = run.result("A_forced_and_ceilings.csv")
    assert not forced["above own ceiling"].any()
    cost = run.json("A_deployment_cost.json")
    assert cost["answers equal to evaluation"] >= 0.99
    parts = ["GRU", "Isolation Forest", "Mahalanobis", "decision rule"]
    assert np.isclose(sum(cost[f"{x}, per packet (us)"] for x in parts), cost["inference per packet, one thread (us)"])


def test_s4_s5_cover_every_session_and_both_memories(run):
    sessions = run.result("A_sessions.csv")
    assert set(sessions["run"]) == {f"without {c}" for c in KNOWN} | {"Feb-May to Aug"}
    assert set(run.result("A_retraining.csv")["model"]) == {"before", "after"}
    assert set(run.result("A_whole_flow.csv", index_col=0).index) == {"reset", "carried"}
    stateful = run.result("A_stateful_pipeline.csv")
    assert set(stateful["memory"]) == {"reset", "carried"} and set(stateful["K"]) == set(kl.K_REPORT)


def test_export_writes_artifacts_that_belong_together(run):
    for role in ("main", "no_testbed"):
        folder = run.root / "artifacts" / "A" / role
        assert sorted(p.name for p in folder.iterdir()) == ["config.json", "model.pt", "unit.joblib"]
        config = kl.load_json(folder / "config.json")
        assert config["model weights"] == joblib.load(folder / "unit.joblib")["model weights"]
        assert set(config["thresholds per K"]) == {str(K) for K in kl.K_REPORT}


def test_infer_on_the_raw_capture_reproduces_stage_3(run):
    """infer.py on Mirai's raw capture (the deployment path) gives the answers stage 3 measured on the same traffic."""
    answers = pd.read_csv(run.root / "mirai_answers.csv")
    held = run.result("A_held_out.csv")
    held = held[(held["model"] == run.model_name(NO_TESTBED)) & (held["view"] == "all") & held["traffic"].str.startswith("Mirai")]
    assert len(answers) == held["segments"].sum()
    shares = answers["answer"].value_counts(normalize=True)
    for a in kl.ANSWERS:
        assert shares.get(a, 0.0) == pytest.approx((held[a] * held["segments"]).sum() / held["segments"].sum(), abs=1e-9)
    assert answers["attack"].sum() == held.loc[held["traffic"] == "Mirai, during infection", "segments"].sum()


def test_s6_uncertainty_parts_add_up(run):
    u = run.result("A_no_testbed_mc_uncertainty_by_answer.csv", header=[0, 1], index_col=[0, 1])
    total, aleatoric, epistemic = (u[(part, "mean")] for part in ("total", "aleatoric", "epistemic"))
    assert (epistemic >= 0).all() and (total <= 1 + 1e-9).all()  # entropy of a yes/no answer, in bits
    assert np.allclose(total, aleatoric + epistemic, atol=1e-3)  # total = aleatoric + mutual information


def test_s7_psi_is_computed_for_every_comparison(run):
    psi = run.result("A_no_testbed_psi.csv", index_col=[0, 1])
    assert set(psi.index.get_level_values(0)) == {"training -> test", "Feb-May captures -> Aug captures",
                                                  *(f"training -> held-out {c}" for c in kl.OOD_CAPTURES)}
    assert (psi.fillna(0) >= 0).all().all()


# --- Reference values ------------------------------------------------------------------------------------------------

REFERENCE_VALUES = json.loads(REFERENCE.read_text(encoding="utf-8"))["values"] if REFERENCE.exists() else {}
SKIP_MACHINE_SENSITIVE = os.environ.get("SKIP_MACHINE_SENSITIVE") == "1"


@pytest.fixture(scope="module")
def collected(run):
    return collect(run)


def test_the_reference_covers_every_collected_value(collected):
    assert REFERENCE_VALUES, f"{REFERENCE} is missing: run tests/integration/record_reference.py on the agreed code"
    assert set(collected) == set(REFERENCE_VALUES)
    assert MACHINE_SENSITIVE <= set(collected)  # a renamed value must not silently leave the list


@pytest.mark.parametrize("name", sorted(REFERENCE_VALUES))
def test_matches_the_reference(collected, name):
    if SKIP_MACHINE_SENSITIVE and name in MACHINE_SENSITIVE:
        pytest.skip("machine-sensitive: compared only where SKIP_MACHINE_SENSITIVE is not set, such as the machine that "
                    "recorded the reference (see integration_support.MACHINE_SENSITIVE)")
    ref = REFERENCE_VALUES[name]
    value, _ = collected[name]
    expected = ref["value"]
    if expected is None or isinstance(expected, str):
        assert value == expected
    else:
        assert value is not None and not math.isnan(value)
        assert value == pytest.approx(expected, rel=ref.get("rel"), abs=ref.get("abs")), \
            f"{name}: {value} against the recorded {expected}"


# --- Known behaviour: stage 3's timing path on exact ties ------------------------------------------------------------

@pytest.mark.xfail(raises=RuntimeError, strict=False, reason=(
    "s3_evaluate.py's packet-by-packet timing path sums OOD scores in float64 while evaluation stores them in float32, "
    "so a score exactly on an OOD limit can be decided differently; with identical first packets in a tiny archive, "
    "ties occur and the consistency check stops stage 3. Not changed in the pipeline (no tie on the real data: 100% "
    "agreement). Not strict: whether a tie occurs can depend on the platform's floating-point results."))
def test_s3_timing_path_on_exact_ties(tmp_path):
    write = lambda path: synthetic_kitsune.write_archive(path, distinct_first_packets=False)  # noqa: E731
    with isolated_pipeline(tmp_path, write) as p:
        for stage in ("s0_prepare", "s1_data", "s2_models", "s3_evaluate"):
            try:
                p.run(stage)
            except SystemExit as e:
                raise RuntimeError(f"{stage}: {e}") from e
        assert p.json("A_deployment_cost.json")["answers equal to evaluation"] >= 0.99
