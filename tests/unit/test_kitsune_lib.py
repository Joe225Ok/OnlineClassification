"""Unit tests of kitsune_lib on small hand-made inputs. Every expected value is worked out by hand from the definition
(the derivation is in the comment next to it), not taken from a run of the code. No data files are needed.
    python -m pytest          (from the repository root)
"""
import math

import numpy as np
import pytest

import kitsune_lib as kl


# --- Decision rule -----------------------------------------------------------------------------------------------

def test_route_answers_and_decision_packets():
    tau_attack, tau_benign, K = 0.99, 0.01, 3
    p = np.array([
        [0.2, 0.995, 0.1],    # High at packet 2 (first p >= 0.99)
        [0.995, 0.1, 0.1],    # flagged and High at packet 1: OOD comes first
        [0.2, 0.3, 0.001],    # never High; p at the last packet <= 0.01: Low at packet 3
        [0.2, 0.3, 0.5],      # never High, last p above 0.01: Medium at packet 3
        [0.2, 0.005, 0.0],    # 2 real packets: Low at its last packet, 2
        [0.2, 0.3, 0.999],    # 2 real packets: the flag and the 0.999 at packet 3 are padding, so Medium at 2
    ])
    flagged = np.zeros_like(p, bool)
    flagged[1, 0] = True
    flagged[5, 2] = True
    lengths = np.array([3, 3, 3, 3, 2, 2])
    out, at = kl.route(p, flagged, lengths, tau_attack, tau_benign, K)
    assert out.tolist() == ["High", "OOD", "Low", "Medium", "Low", "Medium"]
    assert at.tolist() == [2, 1, 3, 3, 2, 2]


def test_route_low_only_at_K():
    # K = 2: a Low answer is read at packet 2 even though the segment is longer; packet 3 is never looked at
    p = np.array([[0.2, 0.005, 0.999]])
    out, at = kl.route(p, np.zeros_like(p, bool), np.array([3]), 0.99, 0.01, 2)
    assert (out.tolist(), at.tolist()) == (["Low"], [2])


def test_tune_targets_lowest_tau_attack_and_highest_tau_benign():
    # One packet each: attacks at 0.9 and 0.95, benign at 0.6 and 0.001
    p = np.array([[0.9], [0.95], [0.6], [0.001]])
    y = np.array([1, 1, 0, 0])
    t = kl.tune_targets(p, np.zeros_like(p, bool), np.ones(4, int), y, K=1)
    # tau_attack: the lowest cut-off at which >= 99% of blocks are attacks. Any cut-off <= 0.6 also blocks the benign
    # 0.6 (2/3 precision); the first cut-off of the grid above 0.6 blocks exactly the two attacks.
    assert 0.6 < t["tau_attack"] <= 0.9
    assert (t["blocks"], t["block precision"], t["high target met"]) == (2, 1.0, True)
    # tau_benign: the highest cut-off (the grid starts at 10^-0.3) with >= 99.9% of Low answers benign. At 10^-0.3
    # only the 0.001 segment is Low (0.6 is above it, the attacks are blocked), and it is benign.
    assert t["tau_benign"] == pytest.approx(10 ** -0.3)
    assert (t["Low answers"], t["Low precision"], t["low target met"]) == (1, 1.0, True)


def test_tune_targets_not_met_never_blocks():
    # Every segment at 0.6, half of them attacks: no cut-off reaches 99% block precision
    p = np.full((4, 1), 0.6)
    t = kl.tune_targets(p, np.zeros_like(p, bool), np.ones(4, int), np.array([1, 0, 1, 0]), K=1)
    assert (t["tau_attack"], t["high target met"]) == (2.0, False)  # 2.0: above any probability, so never High


# --- Ceilings ------------------------------------------------------------------------------------------------------

def test_ceiling_majority_label_per_identical_sequence():
    # Sequence 7 appears 3 times (2 attack, 1 benign): the oracle calls all three attack. Sequence 9: benign.
    c = kl.ceiling(np.array([7, 7, 7, 9]), np.array([1, 1, 0, 0]))
    # Oracle answers [1, 1, 1, 0] against [1, 1, 0, 0]: 3 of 4 right
    assert c["MaxAcc"] == pytest.approx(0.75)
    # Attack class: precision 2/3, recall 1, F1 0.8. Benign class: precision 1, recall 1/2, F1 2/3. Mean 11/15.
    assert c["MaxF1"] == pytest.approx((0.8 + 2 / 3) / 2)
    # Minimum FPR (every mixed cluster called attack): of the 2 benign rows, 1 sits in the mixed sequence 7
    assert c["minimum FPR"] == pytest.approx(0.5)
    assert c["minimum missed attacks"] == 0.0
    assert (c["rows"], c["rows in conflict"]) == (4, 3)


def test_ceiling_tie_goes_to_attack():
    c = kl.ceiling(np.array([1, 1]), np.array([1, 0]))  # one of each: called attack, so 1 of 2 right
    assert c["MaxAcc"] == pytest.approx(0.5)


# --- Statistics ----------------------------------------------------------------------------------------------------

def test_precision_bound_clopper_pearson():
    # With all n hits, the one-sided 95% Clopper-Pearson lower bound solves p^n = 0.05, so p = 0.05^(1/n)
    assert kl.precision_bound(10, 10) == pytest.approx(0.05 ** (1 / 10))
    assert kl.precision_bound(0, 10) == 0.0  # no hits: the bound is 0
    assert kl.precision_bound(0, 0) == 0.0   # nothing answered


def test_at_rate_bayes():
    # 90% of attacks and 1% of benign blocked, 1% attacks: 0.9 x 0.01 / (0.9 x 0.01 + 0.01 x 0.99) = 0.009 / 0.0189
    assert kl.at_rate(0.9, 0.01, 0.01) == pytest.approx(0.009 / 0.0189)
    assert math.isnan(kl.at_rate(0.0, 0.0, 0.01))  # nothing blocked: undefined


def test_running_mean_over_real_packets():
    # Packets 1 and 2 are real: means 1 and (1 + 3) / 2 = 2; packet 3 is padding (-inf)
    m = kl.running_mean(np.array([[1.0, 3.0, 5.0]]), np.array([2]))
    assert m[0, :2].tolist() == [1.0, 2.0] and m[0, 2] == -np.inf


def test_psi():
    assert kl.psi([0, 0, 1, 1], [0, 0, 1, 1]) == 0.0  # same distribution
    # Reference shares (0.5, 0.5), current (0.75, 0.25): 0.25 ln(1.5) + (-0.25) ln(0.5) = 0.25 ln 3
    assert kl.psi([0, 0, 1, 1], [0, 0, 0, 1]) == pytest.approx(0.25 * math.log(3))
    # A value never seen in the reference gets its own bin, with the empty shares floored at 1e-4
    e, a, f = 0.5, 1.0, 1e-4
    expected = 2 * (f - e) * math.log(f / e) + (a - f) * math.log(a / f)
    assert kl.psi([0, 0, 1, 1], [2, 2, 2, 2]) == pytest.approx(expected)
    assert math.isnan(kl.psi([], [1]))


# --- Data rules ----------------------------------------------------------------------------------------------------

def test_time_split_60_20_20():
    import pandas as pd
    # Rank (i + 0.5) / 10: rows 0-5 below 0.6 (train), 6-7 below 0.8 (validation), 8-9 test
    s = kl.time_split(pd.DataFrame(index=range(10)))
    assert s.tolist() == ["train"] * 6 + ["val"] * 2 + ["test"] * 2


def test_track_rule_b_drops_conflicting_sequences():
    import pandas as pd
    rows = pd.DataFrame({"hash": [1, 1, 2], "attack": [0, 1, 0]})  # sequence 1 carries both labels
    assert len(kl.apply_track_rule(rows, "A")) == 3
    assert kl.apply_track_rule(rows, "B")["hash"].tolist() == [2]


# --- Cached models and their fingerprints --------------------------------------------------------------------------

def test_digest_depends_on_contents_dtype_and_shape():
    a = np.arange(6, dtype=np.float32)
    assert kl.digest(a) == kl.digest(a.copy())
    assert kl.digest(a) != kl.digest(a + 1)
    assert kl.digest(a) != kl.digest(a.astype(np.float64))
    assert kl.digest(a) != kl.digest(a.reshape(2, 3))


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    """A scratch model folder and a tiny training set (3 inputs, the last one the mask)."""
    monkeypatch.setattr(kl, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(kl, "MAX_EPOCHS", 1)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(16, 4, 3)).astype(np.float32)
    x[:, :, -1] = 1.0
    y = np.tile([0.0, 1.0], 8).astype(np.float32)
    return tmp_path, x, y


def test_fit_gru_reuses_only_a_model_trained_on_the_same_data(tiny):
    folder, x, y = tiny
    model, loss = kl.fit_gru("m", x, y, x, y, hidden=4, layers=1)
    saved = (folder / "m.pt")
    assert saved.exists()
    _, loss_again = kl.fit_gru("m", x, y, x, y, hidden=4, layers=1)  # same inputs: reused, not retrained
    assert loss_again == loss
    with pytest.raises(RuntimeError, match="other inputs"):
        kl.fit_gru("m", x + 1, y, x, y, hidden=4, layers=1)  # other training data
    with pytest.raises(RuntimeError, match="other inputs"):
        kl.fit_gru("m", x, y, x, y, hidden=4, layers=1, seed=1)  # other seed


def test_load_gru_checks_size_and_training_rows_and_reports_a_missing_file(tiny):
    _, x, y = tiny
    model, _ = kl.fit_gru("m", x, y, x, y, hidden=4, layers=1, rows="rows of track A")
    loaded = kl.load_gru("m", 3, 4, 1, rows="rows of track A")
    probe = x[:2]
    assert np.array_equal(kl.predict(loaded, probe), kl.predict(model, probe))
    with pytest.raises(RuntimeError, match="other inputs"):
        kl.load_gru("m", 3, 8, 1, rows="rows of track A")  # saved with hidden size 4
    with pytest.raises(RuntimeError, match="other inputs"):
        kl.load_gru("m", 3, 4, 1, rows="rows of a rebuilt track")  # trained on other rows
    with pytest.raises(FileNotFoundError, match="missing"):
        kl.load_gru("absent", 3, 4, 1, rows="rows of track A")


def test_rows_digest_changes_with_rows_labels_and_inputs():
    import pandas as pd
    segments = pd.DataFrame({"attack": [0, 1, 0, 1], "hash_all": np.array([10, 11, 12, 13], np.uint64)})
    stats = (np.zeros(2), np.ones(2))
    base = kl.rows_digest("A", np.array([0, 1]), np.array([2]), [0, 1], stats, segments)
    assert base == kl.rows_digest("A", np.array([0, 1]), np.array([2]), [0, 1], stats, segments)
    assert base != kl.rows_digest("B", np.array([0, 1]), np.array([2]), [0, 1], stats, segments)       # other track
    assert base != kl.rows_digest("A", np.array([0, 3]), np.array([2]), [0, 1], stats, segments)       # other rows
    relabelled = segments.assign(attack=[1, 1, 0, 1])
    assert base != kl.rows_digest("A", np.array([0, 1]), np.array([2]), [0, 1], stats, relabelled)     # other label
    rebuilt = segments.assign(hash_all=np.array([10, 99, 12, 13], np.uint64))
    assert base != kl.rows_digest("A", np.array([0, 1]), np.array([2]), [0, 1], stats, rebuilt)        # other packets


# --- Sequence keys -------------------------------------------------------------------------------------------------

def _packets(**changes):
    """Two identical packets, as stage 0 prepares them; `changes` alters the second one."""
    import pandas as pd
    row = {"direction": 0, "frame_len": 60, "protocol": 1, "tcp_flags": 2, "tcp_window": 1024, "ip_hlen": 20,
           "l4_hlen": 20, "arp_op": 0, "is_ipv6": 0, "ttl": 64, "dst_port": 80, "label": 0}
    p = pd.DataFrame([row, row])
    for column, value in changes.items():
        p.loc[1, column] = value
    return p


@pytest.mark.parametrize("change, same_headers, same_all_inputs", [
    ({}, True, True),                      # identical packets
    ({"label": 1}, True, True),            # the label is never part of a key
    ({"ttl": 128}, True, False),           # TTL: only in "all inputs"
    ({"dst_port": 443}, True, False),      # destination port: only in "all inputs"
    ({"frame_len": 61}, False, False),     # a header field: in both
])
def test_packet_keys(change, same_headers, same_all_inputs):
    header, all_inputs = kl.packet_keys(_packets(**change), np.array([0, 0]))
    assert (header[0] == header[1], all_inputs[0] == all_inputs[1]) == (same_headers, same_all_inputs)


def test_packet_keys_iat_only_in_all_inputs():
    header, all_inputs = kl.packet_keys(_packets(), np.array([0, 5]))  # gaps of 0 and 5 microseconds
    assert header[0] == header[1] and all_inputs[0] != all_inputs[1]


def test_model_key():
    every = list(kl.GROUPS)
    assert kl.model_key(every) == "all inputs"
    assert kl.model_key([g for g in every if g not in kl.TESTBED_GROUPS]) == "headers"
    with pytest.raises(ValueError, match="no sequence key"):
        kl.model_key([g for g in every if g != "TTL"])  # no key holds exactly these inputs


def test_model_saved_before_fingerprints_is_used_with_a_warning(tiny):
    import torch
    folder, x, y = tiny
    kl.fit_gru("old", x, y, x, y, hidden=4, layers=1)
    saved = torch.load(folder / "old.pt")
    del saved["fingerprint"]  # as the models saved before this change
    torch.save(saved, folder / "old.pt")
    with pytest.warns(UserWarning, match="cannot be checked"):
        kl.fit_gru("old", x, y, x, y, hidden=4, layers=1)


# --- Units tied to their model weights (stage 3, stage 7, export, inference) ----------------------------------------

def test_weights_digest_changes_with_any_weight():
    import torch
    torch.manual_seed(0)
    net = kl.OnlineGRU(3, 4, 1)
    d = kl.weights_digest(net.state_dict())
    assert d == kl.weights_digest({k: v.clone() for k, v in net.state_dict().items()})
    changed = {k: v.clone() for k, v in net.state_dict().items()}
    changed["head.bias"] += 1e-6
    assert d != kl.weights_digest(changed)


def test_check_unit_refuses_other_weights():
    import torch
    torch.manual_seed(0)
    a, b = kl.OnlineGRU(3, 4, 1), kl.OnlineGRU(3, 4, 1)  # same size, different random weights
    unit = {"model weights": kl.weights_digest(a.state_dict())}
    kl.check_unit(unit, a.state_dict(), "unit")
    with pytest.raises(RuntimeError, match="other model weights"):
        kl.check_unit(unit, b.state_dict(), "unit")
    with pytest.raises(RuntimeError, match="other model weights"):
        kl.check_unit({}, a.state_dict(), "unit")  # a unit that records no weights is refused too


@pytest.fixture
def infer_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("infer", kl.Path("pipeline") / "infer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _artifact(folder, net, unit_net, config_weights=None, versions=None):
    """An artifact folder as export_artifacts.py writes it: model.pt from `net`, a unit built for `unit_net`."""
    import joblib
    import sklearn
    import torch
    folder.mkdir()
    built_for = kl.weights_digest(unit_net.state_dict())
    torch.save({"state": net.state_dict(), "n_inputs": 3, "hidden": 4, "layers": 1, "model weights": built_for}, folder / "model.pt")
    joblib.dump({"model weights": built_for}, folder / "unit.joblib")
    kl.save_json({"model weights": config_weights or built_for,
                  "versions": versions or {"torch": torch.__version__, "scikit-learn": sklearn.__version__,
                                           "numpy": np.__version__}}, folder / "config.json")
    return folder


def test_infer_loads_a_matching_artifact(tmp_path, infer_module):
    import torch
    torch.manual_seed(0)
    net = kl.OnlineGRU(3, 4, 1)
    loaded, unit, _ = infer_module.load_artifact(_artifact(tmp_path / "ok", net, net))
    assert kl.weights_digest(loaded.state_dict()) == unit["model weights"]


def test_infer_refuses_a_model_its_unit_was_not_built_for(tmp_path, infer_module):
    import torch
    torch.manual_seed(0)
    net, other = kl.OnlineGRU(3, 4, 1), kl.OnlineGRU(3, 4, 1)
    with pytest.raises(RuntimeError, match="other model weights"):
        infer_module.load_artifact(_artifact(tmp_path / "mixed", net, other))
    with pytest.raises(RuntimeError, match="config.json"):
        infer_module.load_artifact(_artifact(tmp_path / "config", net, net, config_weights="0" * 32))


def test_infer_warns_when_a_library_version_differs(tmp_path, infer_module):
    import torch
    torch.manual_seed(0)
    net = kl.OnlineGRU(3, 4, 1)
    old = {"torch": torch.__version__, "scikit-learn": "0.0.1", "numpy": np.__version__}
    with pytest.warns(UserWarning, match="scikit-learn"):
        infer_module.load_artifact(_artifact(tmp_path / "old", net, net, versions=old))


def test_fit_gru_takes_the_model_class_without_changing_the_default(tiny):
    import torch

    class Variant(kl.OnlineGRU):
        pass

    _, x, y = tiny
    model, _ = kl.fit_gru("variant", x, y, x, y, hidden=4, layers=1, model_class=Variant)
    assert type(model) is Variant and kl.OnlineGRU is not Variant  # nothing patched in kitsune_lib
    assert torch.load(kl.MODEL_DIR / "variant.pt")["fingerprint"]["model class"] == "Variant"
    with pytest.raises(RuntimeError, match="other inputs"):  # the same file read as the default class is refused
        kl.fit_gru("variant", x, y, x, y, hidden=4, layers=1)


# --- Stage 5: where each segment of a flow starts ---------------------------------------------------------------

def test_chain_start_states_follow_each_flow():
    import torch
    torch.manual_seed(0)
    gru = torch.nn.GRU(3, 4, num_layers=2, batch_first=True)
    x = np.random.default_rng(0).normal(size=(4, 5, 3)).astype(np.float32)
    rows, flows = np.array([0, 1, 2, 3]), np.array([7, 7, 7, 9])  # flow 7: rows 0, 1, 2 in order; flow 9: row 3
    start = kl.chain_start_states(gru, x, rows, flows, layers=2, hidden=4)
    with torch.no_grad():
        _, after0 = gru(torch.from_numpy(x[[0]]))           # flow 7 run segment by segment, by hand
        _, after1 = gru(torch.from_numpy(x[[1]]), after0)
    assert torch.equal(start[:, 0], torch.zeros(2, 4))      # a flow's first segment starts empty
    assert torch.allclose(start[:, 1], after0[:, 0])        # the next one starts where the previous one ended
    assert torch.allclose(start[:, 2], after1[:, 0])        # ... through the whole flow
    assert torch.equal(start[:, 3], torch.zeros(2, 4))      # another flow starts empty again


def test_chain_start_states_use_only_the_given_rows():
    import torch
    torch.manual_seed(0)
    gru = torch.nn.GRU(3, 4, batch_first=True)
    x = np.random.default_rng(1).normal(size=(3, 5, 3)).astype(np.float32)
    # Rows 0 and 2 of one flow, row 1 left out (a Track B conflict in training): row 2 starts where row 0 ended
    start = kl.chain_start_states(gru, x, np.array([0, 2]), np.array([5, 5]), layers=1, hidden=4)
    with torch.no_grad():
        _, after0 = gru(torch.from_numpy(x[[0]]))
    assert torch.allclose(start[:, 1], after0[:, 0])


def test_a_new_training_version_refuses_models_trained_before(tiny, monkeypatch):
    _, x, y = tiny
    kl.fit_gru("versioned", x, y, x, y, hidden=4, layers=1, rows="track A")
    kl.load_gru("versioned", 3, 4, 1, rows="track A")  # same version: loads
    monkeypatch.setattr(kl, "TRAINING_VERSION", kl.TRAINING_VERSION + 1)  # the training code changed
    with pytest.raises(RuntimeError, match="training version"):
        kl.fit_gru("versioned", x, y, x, y, hidden=4, layers=1, rows="track A")
    with pytest.raises(RuntimeError, match="training version"):
        kl.load_gru("versioned", 3, 4, 1, rows="track A")
