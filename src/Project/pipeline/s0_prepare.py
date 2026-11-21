"""Stage 0: data exploration, PCAP parsing, flows, segments and inputs.

Run it from anywhere (it works in the project folder), or run every stage with run_all.py in the project folder.

"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))


def show(value):
    """Print a table (or any value) left as the last expression of a block."""
    if value is None:
        return
    print(value.to_string() if hasattr(value, "to_string") else value)
    print()




# %% [markdown]
# # Kitsune: online packet-header classification
#
# Follows `Kitsune_Classification_Updated.pdf`. The classifier is **online**: its input is each packet's header fields, in
# order, plus the time since the previous packet of the flow, and it can decide after every packet. Only the PCAPs and
# label files are used; the 115-statistic `_dataset.csv` files are not, because they aggregate many packets over time.
#
# This script is stage 0: data exploration, parsing of the captures, flows, 50-packet segments and model inputs. The
# later stages are the other scripts in `pipeline/`, run once per track with shared code in `kitsune_lib.py`:
#
# | Stage | Script | Contents |
# |---|---|---|
# | 1 | `s1_data.py` | diagnostics before the split, Tracks A and B, time split |
# | 2 | `s2_models.py` | GRU size, main model and model without TTL, destination port and IAT |
# | 3 | `s3_evaluate.py` | deployable units, thresholds, views, ceilings, baselines and the real-world checks |
# | 4 | `s4_experiments.py` | retraining on OOD, unseen sessions, whole flows |
# | 5 | `s5_stateful.py` | stateful GRU against the main model |
# | 6 | `s6_uncertainty.py` | MC dropout on the chosen model: aleatoric and epistemic uncertainty |
# | 7 | `s7_monitoring.py` | drift monitoring with the population stability index (PSI) |
#
# Stages 2-5 run for Tracks A and B, stages 6 and 7 for Track A only.
#
# **Tracks.** A: labels as recorded. B: conflicting sequences removed, the best case after manual review (not every
# conflict may be resolvable). Results on real data, where conflicting labels are addressed when possible, are expected to fall between the two.
#
# **Reproducing.** Every random step uses `SEED = 0`; GRU training gives bit-identical weights on rerun with the same
# number of CPU threads. The machine, thread settings and package versions of the last run are in
# `data/results/environment.txt` (written by `run_all.py`). `python run_all.py` in the project folder
# runs every stage in order (stages 0 and 1 once, 2 to 5 and the export of the artifacts for Tracks A and B, 6 and 7 for
# Track A); `--jobs 2` runs independent steps in parallel; `--help` lists every option.
#
# Models and intermediate results are cached (`data/models_v2`, `data/results`, `data/tracks`), so a rerun reuses the
# trained models; delete them to recompute from scratch.


# %%
import gzip
import time
import zipfile
from pathlib import Path

import dpkt
import numpy as np
import pandas as pd

import kitsune_lib
from kitsune_lib import CAPTURES, COL, COLUMNS, L, OOD_CAPTURES, SEED  # noqa: F401  (used further down)

# 9 attack folders, each with <Attack>_pcap, _labels and _dataset (unused). 19 GB zipped.
DATA_DIR = Path("data")
archive = zipfile.ZipFile(DATA_DIR / "kitsune+network+attack+dataset.zip")
for m in archive.infolist():
    if "_pcap" in m.filename or "_labels" in m.filename:
        print(f"{m.file_size / 1e6:8.1f} MB  {m.filename}")


# %%
# Data exploration: what the raw data looks like, using Mirai as the example.
# Each capture is a gzipped PCAP plus a labels CSV holding one 0/1 label per packet, in packet order.
# (member() is defined further down, so the files are looked up directly here.)
pcap_name = next(n for n in archive.namelist() if n.startswith("mirai/") and "_pcap" in n)
labels_name = next(n for n in archive.namelist() if n.startswith("mirai/") and "_labels" in n)

with archive.open(labels_name) as f, gzip.open(f) as g:
    labels = pd.read_csv(g, header=None, names=["label"])["label"]
first_attack = int(labels.to_numpy().argmax())
print(f"{labels_name}: {len(labels):,} rows, no header, one label per packet")
print(f"  benign (0): {(labels == 0).sum():,}   malicious (1): {(labels == 1).sum():,}")
print(f"  first malicious packet: number {first_attack + 1:,}\n")


def describe(ts, buf):
    eth = dpkt.ethernet.Ethernet(buf)
    row = {"time": ts, "bytes": len(buf), "src": dpkt.utils.mac_to_str(eth.src),
           "dst": dpkt.utils.mac_to_str(eth.dst), "protocol": type(eth.data).__name__, "ports": ""}
    if isinstance(eth.data, bytes):  # EtherType dpkt does not decode
        row["protocol"] = f"type 0x{eth.type:04x}"
    if isinstance(eth.data, (dpkt.ip.IP, dpkt.ip6.IP6)):
        ip = eth.data
        row["src"], row["dst"] = dpkt.utils.inet_to_str(ip.src), dpkt.utils.inet_to_str(ip.dst)
        row["protocol"] = type(ip.data).__name__
        if isinstance(ip.data, (dpkt.tcp.TCP, dpkt.udp.UDP)):
            row["ports"] = f"{ip.data.sport} -> {ip.data.dport}"
    return row


# Despite its .pcap extension, Mirai's file is pcapng.
shown = set(range(5)) | set(range(first_attack - 2, first_attack + 3))
rows = []
with archive.open(pcap_name) as f, gzip.open(f) as g:
    for i, (ts, buf) in enumerate(dpkt.pcapng.Reader(g)):
        if i in shown:
            rows.append({"packet": i + 1, **describe(ts, buf), "label": labels[i]})
        if i == first_attack:
            attack_buf = buf
        if i > max(shown):
            break
packets = pd.DataFrame(rows)
packets["time"] = (packets["time"] - packets["time"].iloc[0]).round(3)  # seconds since the first packet
print(f"{pcap_name}: first packets, then the packets around the first malicious one")
print(packets.to_string(index=False))

# The table above is a selection. Every header field of one packet, layer by layer:
print(f"\nAll fields of packet {first_attack + 1:,} (the first malicious one), {len(attack_buf)} bytes captured:")
layer = dpkt.ethernet.Ethernet(attack_buf)
while isinstance(layer, dpkt.Packet):
    fields = {}
    for name in layer.__hdr_fields__:
        # Fields packed into shared bytes (e.g. IP version + header length) are split into their parts
        for part in [p for p, _ in getattr(layer, "__bit_fields__", {}).get(name, [(name, 0)]) if not p.startswith("_")]:
            value = getattr(layer, part)
            if part in ("src", "dst"):
                value = dpkt.utils.mac_to_str(value) if len(value) == 6 else dpkt.utils.inet_to_str(value)
            fields[part] = value
    print(f"  {type(layer).__name__:<9}" + ", ".join(f"{k}={v}" for k, v in fields.items()))
    layer = layer.data
print(f"  {'payload':<9}{len(layer)} bytes: {layer!r}")


# %%
# Proposal Table 1: capture -> folder and role (kitsune_lib.CAPTURES); ARP MitM and Mirai are held out
# (kitsune_lib.OOD_CAPTURES). Segments of L = 50 packets; SEED = 0 for every random step.
PACKET_DIR = DATA_DIR / "packets"
PACKET_DIR.mkdir(exist_ok=True)


def member(capture, suffix):
    return next(n for n in archive.namelist() if n.startswith(CAPTURES[capture] + "/") and suffix in n)


def read_labels(capture):
    # Mirai's file is a bare column; the others have a header and an index column.
    with archive.open(member(capture, "_labels")) as f, gzip.open(f) as g:
        labels = pd.read_csv(g, header=None, low_memory=False).iloc[:, -1]
    return pd.to_numeric(labels, errors="coerce").dropna().to_numpy(dtype=np.int8)


def parse_capture(capture):
    """One row per packet, in file order so it lines up with the labels file, parsed by kitsune_lib.parse_pcap (the
    parser infer.py uses for new captures)."""
    labels = read_labels(capture)
    with archive.open(member(capture, "_pcap")) as f, gzip.open(f) as g:
        out = kitsune_lib.parse_pcap(g)
    assert len(out) == len(labels), f"{capture}: {len(out):,} packets but {len(labels):,} labels"
    out["label"] = labels
    return out


# %%
# Parse every PCAP with dpkt (~4 min); cached after the first run.
# parse_capture asserts that the packet count equals the label count, capture by capture.
t0 = time.time()
for capture, folder in CAPTURES.items():
    path = PACKET_DIR / f"{folder}.parquet"
    if not path.exists():
        parse_capture(capture).to_parquet(path, index=False)
        print(f"{capture}: parsed, {time.time() - t0:.0f}s")

overview = []
for capture, folder in CAPTURES.items():
    p = pd.read_parquet(PACKET_DIR / f"{folder}.parquet", columns=["ts", "flow", "label", "ip_proto", "ethertype"])
    overview.append({
        "capture": capture, "packets": len(p), "malicious share": p["label"].mean(),
        "minutes": (p["ts"].max() - p["ts"].min()) / 60,
        "attack starts at (share of capture)": p["label"].to_numpy().argmax() / len(p),
        "TCP": (p["ip_proto"] == 6).mean(), "UDP": (p["ip_proto"] == 17).mean(),
        "ARP": (p["ethertype"] == 0x806).mean(),
    })
overview = pd.DataFrame(overview).set_index("capture")
show(overview.round(3))


# %%
# What the parsed packets look like: the first packets of one capture, and the first label change
p = pd.read_parquet(PACKET_DIR / "active_wiretap.parquet")
p.insert(0, "packet", np.arange(1, len(p) + 1))
shown_cols = ["packet", "ts", "flow", "direction", "frame_len", "ethertype", "ip_proto", "ttl", "l4_hlen", "tcp_flags", "tcp_window", "dst_port", "label"]
print(f"Active Wiretap: {p.shape[0]:,} packets, {p.shape[1] - 1} fields. First packets:")
print(p[shown_cols].head(5).to_string(index=False))
first_attack = int(p["label"].to_numpy().argmax())
print(f"\nFirst malicious packet is number {first_attack + 1:,}. Around it:")
print(p[shown_cols].iloc[first_attack - 2:first_attack + 3].to_string(index=False))
print("\nProtocols (ip_proto):", p["ip_proto"].value_counts().head(5).to_dict())


# %% [markdown]
# ## Week 2: flows, segments and inputs
#
# Follows proposal Section 4.2, over all nine captures:
#
# 1. **Flows.** IP packets are grouped by 5-tuple, non-IP packets by MAC pair and EtherType, both directions together.
# 2. **Flow segments.** Each flow is cut into consecutive segments of L = 50 packets and **each segment is one sample**.
#    The table below shows why: whole flows, or only their first L packets, leave few samples and most packets unseen.
#    A segment is labelled attack if any of its packets is labelled malicious. The classifier starts fresh on each, so
#    long flows are re-checked every L packets (Stage 5 carries memory across them instead).
# 3. **Inputs.** Header fields plus the inter-arrival time (IAT). Addresses group packets into flows and are then dropped.
#    TTL, destination port and IAT are inputs of the main model; because they may identify this testbed (hosts,
#    services, network timing) rather than the attack, a second model is trained without them (Stage 2).
# 4. **Diagnostics, tracks and the split** are in Stage 1 below.


# %%
# Why segments: whole flows vs L-packet segments. A sample is attack if any of its packets is malicious.
comparison = []
for capture, folder in CAPTURES.items():
    p = pd.read_parquet(PACKET_DIR / f"{folder}.parquet", columns=["ts", "flow", "label"])
    p = p.sort_values(["flow", "ts"], kind="stable")
    seq = p.groupby("flow").cumcount()
    whole = p.groupby("flow")["label"].max()
    first_l = p[seq < L].groupby("flow")["label"].max()  # label seen by a first-L-packets model
    segment = p.groupby(["flow", seq // L])["label"].max()
    comparison.append({
        "capture": capture, "flows": len(whole), "flows, benign": int((whole == 0).sum()),
        "flows, attack": int(whole.sum()), "largest flow, packets": int(p.groupby("flow").size().max()),
        "packets in first L": int((seq < L).sum()), "attack flows benign in first L": int(whole.sum() - first_l.sum()),
        "segments": len(segment), "segments, benign": int((segment == 0).sum()), "segments, attack": int(segment.sum()),
    })
comparison = pd.DataFrame(comparison).set_index("capture")
comparison.to_csv(DATA_DIR / "flows_vs_segments.csv")
print("in-distribution totals:", comparison.drop(OOD_CAPTURES)[["flows", "flows, benign", "segments", "segments, benign"]].sum().to_dict())
show(comparison)


# %% [markdown]
# Reading the table: a capture is a few cameras holding a few very long connections, plus, in OS Scan and SSDP Flood, tens of thousands of tiny attack flows. Whole flows give only a few hundred benign samples and single-digit attack samples in some captures, and most packets are never seen, so a per-capture 60/20/20 split is not possible. An idle timeout does not help because streams rarely pause. Segments keep nearly all traffic and keep the online logic unchanged.


# %%
# Per-packet inputs: kitsune_lib.COLUMNS (sizes, header lengths, TCP flags and window, protocol, direction, IAT, TTL,
# destination port, and the mask of real packets), built by kitsune_lib.packet_features, the same function infer.py
# uses. Continuous columns are log-scaled there and standardized with training statistics later.
# What makes two packets "the same": two keys, each holding exactly what one model reads (kitsune_lib.packet_keys).
# "headers": every header field except TTL and destination port (the model without TTL, destination port and IAT).
# "all inputs": the same plus TTL, destination port and the IAT in whole microseconds (the main model). The label is
# never part of a key. Stage 1 reports duplicates and conflicts under both.


SEQ_FILE, SEG_FILE, HASH_FILE = DATA_DIR / "sequences.npy", DATA_DIR / "segments.parquet", DATA_DIR / "packet_hashes.npz"
PREFIX_FILE = DATA_DIR / "prefix_hashes.npz"
if not all(f.exists() for f in (SEQ_FILE, SEG_FILE, HASH_FILE, PREFIX_FILE)):
    t0 = time.time()
    n_total = int(comparison["segments"].sum())
    X_out = np.lib.format.open_memmap(SEQ_FILE, mode="w+", dtype=np.float32, shape=(n_total, L, len(COLUMNS)))
    H_header, H_all = np.zeros((n_total, L), np.uint64), np.zeros((n_total, L), np.uint64)
    meta, offset, iat_rows = [], 0, []
    for capture, folder in CAPTURES.items():
        p = pd.read_parquet(PACKET_DIR / f"{folder}.parquet").sort_values(["flow", "ts"], kind="stable").reset_index(drop=True)
        pos, seg = kitsune_lib.flow_positions(p, L)
        seg = seg + offset
        kitsune_lib.add_protocol(p)
        gap_us = kitsune_lib.packet_iat(p, pos) * 1e6
        iat_us = np.rint(gap_us).astype(np.int64)
        X_out[seg, pos] = kitsune_lib.packet_features(p, pos)
        H_header[seg, pos], H_all[seg, pos] = kitsune_lib.packet_keys(p, iat_us)
        # The "all inputs" key holds the IAT in whole microseconds; the model reads the exact gap. Record how many gaps
        # are not whole microseconds, i.e. where the key is coarser than what the model reads.
        iat_rows.append({"capture": capture, "packets": len(p),
                         "gaps not whole microseconds": int((np.abs(gap_us - np.rint(gap_us)) > 1e-3).sum())})
        m = pd.DataFrame({"segment": seg, "label": p["label"], "ts": p["ts"], "is_arp": p["ethertype"] == 0x806})
        m = m.groupby("segment").agg(attack=("label", "max"), malicious=("label", "sum"), length=("label", "size"),
                                     start=("ts", "min"), arp_share=("is_arp", "mean")).reset_index()
        m["capture"] = capture
        meta.append(m)
        offset = seg[-1] + 1
        print(f"\r{capture:18} {time.time() - t0:4.0f}s", end="")
    assert offset == n_total
    X_out.flush()
    del X_out
    pd.concat(meta, ignore_index=True).to_parquet(SEG_FILE, index=False)
    np.savez(HASH_FILE, header=H_header, all_inputs=H_all)
    lengths = pd.concat(meta, ignore_index=True)["length"].to_numpy()
    np.savez(PREFIX_FILE, header=kitsune_lib.prefix_hashes(H_header, lengths), all_inputs=kitsune_lib.prefix_hashes(H_all, lengths))
    pd.DataFrame(iat_rows).set_index("capture").to_csv(kitsune_lib.RESULT_DIR / "iat_resolution.csv")

X = np.load(SEQ_FILE, mmap_mode="r")  # segments x L x columns, zero-padded after the last packet
segments = pd.read_parquet(SEG_FILE)
print(f"\n{X.shape[0]:,} segments, {X.shape[2]} columns, {X.nbytes / 1e9:.1f} GB")
show(segments.groupby("capture", sort=False).agg(segments=("attack", "size"), attack=("attack", "sum"), median_length=("length", "median")))


# %%
# What a sample looks like: a few segments, then the model inputs of one of them, packet by packet
print("Segments (one row per sample):")
print(segments.groupby("capture", sort=False).sample(1, random_state=SEED).to_string(index=False))

example = segments[(segments["capture"] == "SSL Renegotiation") & (segments["attack"] == 1) & (segments["length"] >= 8)].iloc[0]
values = pd.DataFrame(np.array(X[int(example["segment"]), :8]), columns=COLUMNS).round(2)
print(f"\nSegment {int(example['segment']):,}: {example['capture']}, attack={int(example['attack'])}, {int(example['length'])} packets. "
      f"First 8 packets, before standardizing (frame_len, tcp_window and iat are log-scaled):")
print(values[["frame_len", "tcp_window", "ip_hlen", "l4_hlen", "iat", "direction", "tcp", "udp", "syn", "ack", "psh", "fin", "rst", "mask"]].to_string())

short = segments[segments["length"] == 2].iloc[0]
print(f"\nPadding: segment {int(short['segment']):,} has 2 packets, so `mask` is",
      np.array(X[int(short["segment"]), :4, COL["mask"]]).astype(int).tolist(), "for its first 4 steps")
