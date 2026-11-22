"""Unit tests of the components the integration test runs together: the packet parser and segmentation in
kitsune_lib.py, the helpers of pipeline/infer.py, and pipeline/export_artifacts.py. Inputs are built by hand (frames
written with dpkt, tiny models and units); every expected value follows from how the input was built.
"""
import gzip
import importlib.util
import io
import socket

import dpkt
import joblib
import numpy as np
import pandas as pd
import pytest
import torch

import kitsune_lib as kl


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, kl.Path("pipeline") / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- Frames ----------------------------------------------------------------------------------------------------------

A_MAC, B_MAC = b"\x02\x00\x00\x00\x00\x01", b"\x02\x00\x00\x00\x00\x02"
A_IP, B_IP = socket.inet_aton("10.0.0.1"), socket.inet_aton("10.0.0.2")


def eth(etype, payload, src=A_MAC, dst=B_MAC):
    return bytes(dpkt.ethernet.Ethernet(src=src, dst=dst, type=etype, data=payload))


def tcp_frame(src, dst, sport, dport, flags, win, payload=b"", ttl=64):
    seg = dpkt.tcp.TCP(sport=sport, dport=dport, flags=flags, win=win, off=5, data=payload)
    ip = dpkt.ip.IP(src=src, dst=dst, p=dpkt.ip.IP_PROTO_TCP, ttl=ttl, data=seg)
    ip.len = 20 + len(bytes(seg))
    return eth(dpkt.ethernet.ETH_TYPE_IP, ip)


def capture(frames, pcapng=False):
    raw = io.BytesIO()
    writer = dpkt.pcapng.Writer(raw) if pcapng else dpkt.pcap.Writer(raw)
    for i, frame in enumerate(frames):
        writer.writepkt(frame, ts=1_000 + i * 0.25)
    raw.seek(0)
    return io.BufferedReader(raw)


def edge_frames():
    data = tcp_frame(B_IP, A_IP, 554, 40000, dpkt.tcp.TH_ACK | dpkt.tcp.TH_PUSH, 29200, b"d" * 860)
    udp = dpkt.udp.UDP(sport=5353, dport=5353, data=b"m" * 30)
    udp.ulen = 38
    v6 = dpkt.ip6.IP6(src=bytes(15) + b"\x01", dst=bytes(15) + b"\x02", nxt=17, hlim=255, data=udp)
    v6.plen = 38
    echo = dpkt.icmp.ICMP(type=8, code=0, data=dpkt.icmp.ICMP.Echo(id=1, seq=1, data=b"ping"))
    icmp = dpkt.ip.IP(src=A_IP, dst=B_IP, p=dpkt.ip.IP_PROTO_ICMP, ttl=64, data=echo)
    icmp.len = 20 + len(bytes(echo))
    arp = dpkt.arp.ARP(op=1, sha=A_MAC, spa=A_IP, tha=bytes(6), tpa=B_IP)
    llc = b"\x42\x42\x03" + bytes(35)
    return [
        tcp_frame(A_IP, B_IP, 40000, 554, dpkt.tcp.TH_SYN, 1024),  # 0: opens the flow
        data[:200],  # 1: the reply, cut to 200 bytes as in Kitsune: IP length 900
        eth(dpkt.ethernet.ETH_TYPE_IP6, v6),  # 2
        eth(dpkt.ethernet.ETH_TYPE_IP, icmp),  # 3
        eth(dpkt.ethernet.ETH_TYPE_ARP, arp),  # 4
        B_MAC + A_MAC + len(llc).to_bytes(2, "big") + llc,  # 5: 802.3 length frame
        b"\x01\x02\x03\x04\x05\x06\x07\x08",  # 6: too short to be an Ethernet frame
    ]


def test_parse_pcap_reads_every_kind_of_frame():
    p = kl.parse_pcap(capture(edge_frames()))
    assert len(p) == 7
    syn, reply, v6, icmp, arp, llc, junk = (p.iloc[i] for i in range(7))
    # the reply belongs to the SYN's flow, in the other direction; its size comes from the IP length, not the 200 bytes
    assert reply["flow"] == syn["flow"] and (syn["direction"], reply["direction"]) == (0, 1)
    assert reply["frame_len"] == 900 + 14 and reply["ip_hlen"] == 20 and reply["l4_hlen"] == 20
    assert (reply["tcp_flags"], reply["tcp_window"], reply["dst_port"]) == (dpkt.tcp.TH_ACK | dpkt.tcp.TH_PUSH, 29200, 40000)
    assert syn["tcp_flags"] == dpkt.tcp.TH_SYN and syn["ttl"] == 64
    assert (v6["ethertype"], v6["frame_len"], v6["ttl"], v6["ip_hlen"], v6["l4_hlen"]) == (0x86DD, 38 + 54, 255, 40, 8)
    assert (icmp["ip_proto"], icmp["l4_hlen"], icmp["icmp_type"]) == (1, 8, 8)
    assert (arp["ethertype"], arp["arp_op"]) == (0x0806, 1)
    assert llc["ethertype"] == 0  # a length field, not an EtherType
    assert (junk["ethertype"], junk["frame_len"]) == (0, 8)
    assert p["flow"].nunique() == 6  # SYN and reply share one; every other frame is its own flow


def test_pcap_and_pcapng_are_read_the_same_and_limit_stops_early():
    frames = edge_frames()
    a, b = kl.parse_pcap(capture(frames)), kl.parse_pcap(capture(frames, pcapng=True))
    pd.testing.assert_frame_equal(a, b)
    assert len(kl.parse_pcap(capture(frames), limit=3)) == 3


def test_segment_packets_cuts_flows_into_segments_of_l_packets():
    # one flow of 51 packets (segments of 50 and 1) and one of 2 packets, interleaved in the file
    rows = [{"ts": float(i), "flow": 0, "label": int(i >= 40)} for i in range(51)]
    rows += [{"ts": 0.5, "flow": 1, "label": 0}, {"ts": 1.5, "flow": 1, "label": 0}]
    p = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
    for col in ("direction", "frame_len", "ethertype", "ip_proto", "ttl", "ip_hlen", "l4_hlen", "tcp_flags",
                "tcp_window", "arp_op", "dst_port"):
        p[col] = 0
    x, meta = kl.segment_packets(p)
    assert x.shape == (3, kl.L, len(kl.COLUMNS))
    assert meta[["flow", "length", "attack"]].values.tolist() == [[0, 50, 1], [0, 1, 1], [1, 2, 0]]
    assert (x[:, :, kl.MASK_IDX].sum(axis=1) == meta["length"].to_numpy()).all()
    assert meta.loc[1, "first_packet"] == p.index[(p["flow"] == 0)][50]  # the 51st packet of flow 0, in file order


# --- infer.py --------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def infer():
    return load_script("infer")


def test_open_capture_reads_plain_and_gzipped_files_alike(tmp_path, infer):
    raw = capture(edge_frames()).read()
    (tmp_path / "c.pcap").write_bytes(raw)
    (tmp_path / "c.pcap.gz").write_bytes(gzip.compress(raw))
    with infer.open_capture(tmp_path / "c.pcap") as a, infer.open_capture(tmp_path / "c.pcap.gz") as b:
        assert a.read() == b.read() == raw


def test_read_labels_takes_the_last_column_and_skips_a_header(tmp_path, infer):
    (tmp_path / "with_header.csv").write_text(",x\n0,0\n1,1\n2,1\n")
    (tmp_path / "bare.csv.gz").write_bytes(gzip.compress(b"0\n1\n1\n"))
    for name in ("with_header.csv", "bare.csv.gz"):
        labels = infer.read_labels(tmp_path / name)
        assert labels.tolist() == [0, 1, 1] and labels.dtype == np.int8


def tiny_unit(n_inputs, hidden=4, K=kl.L):
    """A model and a unit as stage 3 would build them, with cut-offs and OOD limits set by hand."""
    torch.manual_seed(0)
    net = kl.OnlineGRU(n_inputs, hidden, 1).eval()
    rng = np.random.default_rng(0)
    states = rng.normal(size=(300, hidden)).astype(np.float32)
    detectors = kl.fit_detectors(states, (np.arange(300) % 2).astype(int))
    columns = kl.group_columns(["sizes", "TCP flags", "direction"])
    assert len(columns) == n_inputs
    stats = (np.zeros(len(kl.CONT_IDX)), np.ones(len(kl.CONT_IDX)))
    limits = {"forest": [0.55] * 4, "mahalanobis": [10.0] * 4}
    unit = {"columns": columns, "stats": stats, "temperatures": np.ones(kl.L), "detectors": detectors,
            "thresholds": {K: {"ood limits": limits, "tau_attack": 0.52, "tau_benign": 0.48}},
            "model weights": kl.weights_digest(net.state_dict())}
    return net, unit


def test_classify_is_the_evaluation_rule_in_chunks(infer, monkeypatch):
    n_inputs = len(kl.group_columns(["sizes", "TCP flags", "direction"]))
    net, unit = tiny_unit(n_inputs)
    rng = np.random.default_rng(1)
    n = rng.integers(1, kl.L + 1, 7)
    x = np.zeros((7, kl.L, len(kl.COLUMNS)), np.float32)
    x[:, :, :] = rng.normal(size=x.shape)
    x[:, :, kl.MASK_IDX] = kl.real_steps(n)
    monkeypatch.setattr(infer, "CHUNK", 3)  # three chunks: 3, 3 and 1 segments
    answers, packets, p_last = infer.classify(x, n, net, unit, kl.L)
    # the same steps in one piece, from kitsune_lib's evaluation functions
    xs = kl.load(x, np.arange(7), unit["stats"], unit["columns"])
    z, h = kl.predict(net, xs, with_hidden=True)
    t = unit["thresholds"][kl.L]
    flags = kl.ood_flags(kl.ood_running_means(unit["detectors"], h, n), t["ood limits"])
    p = kl.p_attack(z, unit["temperatures"])
    out, at = kl.route(p, flags, n, t["tau_attack"], t["tau_benign"], kl.L)
    assert answers.tolist() == out.tolist() and packets.tolist() == at.tolist()
    assert np.allclose(p_last, kl.at_prefix(p, n, kl.L))


# --- export_artifacts.py ---------------------------------------------------------------------------------------------

@pytest.fixture
def stage3_outputs(tmp_path, monkeypatch):
    """A selection and, for both final models, the weights (stage 2) and units (stage 3) the export reads."""
    models, results = tmp_path / "models", tmp_path / "results"
    models.mkdir()
    results.mkdir()
    monkeypatch.setattr(kl, "MODEL_DIR", models)
    monkeypatch.setattr(kl, "RESULT_DIR", results)
    monkeypatch.setattr(kl, "GRU_GRID", {"tiny": (4, 1)})
    n_inputs = len(kl.group_columns(["sizes", "TCP flags", "direction"]))
    kl.save_json({"arch": "tiny", "models": {"main": ["T_main"], NO_TESTBED_ROLE: ["T_nt"]}}, results / "T_selection.json")
    for name in ("T_main", "T_nt"):
        net, unit = tiny_unit(n_inputs)
        torch.save({"state": net.state_dict()}, models / f"{name}.pt")
        joblib.dump({**unit, "model": name, "groups": ["sizes", "TCP flags", "direction"],
                     "fitted temperatures": np.ones(kl.L)}, models / f"{name}_unit.joblib")
    return tmp_path


NO_TESTBED_ROLE = "without TTL, destination port and IAT"


def test_export_writes_artifacts_infer_can_load(stage3_outputs, infer):
    export = load_script("export_artifacts")
    export.export("T", out_root=stage3_outputs / "artifacts")
    for role in ("main", "no_testbed"):
        folder = stage3_outputs / "artifacts" / "T" / role
        assert sorted(p.name for p in folder.iterdir()) == ["config.json", "model.pt", "unit.joblib"]
        config = kl.load_json(folder / "config.json")
        assert config["input columns"] == [kl.COLUMNS[c] for c in kl.group_columns(["sizes", "TCP flags", "direction"])]
        assert config["thresholds per K"][str(kl.L)]["tau_attack"] == 0.52
        net, unit, _ = infer.load_artifact(folder)
        assert kl.weights_digest(net.state_dict()) == unit["model weights"] == config["model weights"]
    assert not list((stage3_outputs / "artifacts" / "T").glob("*.tmp"))  # temporary folders swapped in, none left


def test_export_refuses_a_unit_built_for_other_weights(stage3_outputs):
    torch.manual_seed(1)
    other = kl.OnlineGRU(len(kl.group_columns(["sizes", "TCP flags", "direction"])), 4, 1)
    torch.save({"state": other.state_dict()}, kl.MODEL_DIR / "T_main.pt")
    with pytest.raises(RuntimeError, match="other model weights"):
        load_script("export_artifacts").export("T", out_root=stage3_outputs / "artifacts")
    assert not (stage3_outputs / "artifacts" / "T" / "main").exists()


def test_export_stops_when_stage_3_has_not_run(stage3_outputs):
    (kl.MODEL_DIR / "T_main_unit.joblib").unlink()
    with pytest.raises(SystemExit, match="run stage 3"):
        load_script("export_artifacts").export("T", out_root=stage3_outputs / "artifacts")
