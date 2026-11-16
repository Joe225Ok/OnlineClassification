"""Shared code for the Kitsune stages, the scripts pipeline/s0_prepare.py ... s7_monitoring.py.

Every stage imports this module, so data loading, the model, training and the metrics are defined once.
Everything a track needs is computed from that track's own data; nothing here reads another track.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn as tnn

DATA_DIR = Path("data")
TRACK_DIR = DATA_DIR / "tracks"
MODEL_DIR = DATA_DIR / "models_v2"
RESULT_DIR = DATA_DIR / "results"
for d in (TRACK_DIR, MODEL_DIR, RESULT_DIR):
    d.mkdir(parents=True, exist_ok=True)

CAPTURES = {
    "OS Scan": "os_scan", "Fuzzing": "fuzzing",
    "Video Injection": "video_injection", "Active Wiretap": "active_wiretap",
    "SSDP Flood": "ssdp_flood", "SYN DoS": "syn_dos", "SSL Renegotiation": "ssl_renegotiation",
    "ARP MitM": "arp_mitm", "Mirai": "mirai",
}
OOD_CAPTURES = ["ARP MitM", "Mirai"]  # held out: ARP MitM same network, Mirai another network
L = 50
SEED = 0
PART_CODE = {"train": 0, "val": 1, "test": 2}

# Stored columns of sequences.npy, in order (built by stage 0 and reused)
CONTINUOUS = ["frame_len", "tcp_window", "ip_hlen", "l4_hlen", "iat"]
BINARY = ["direction", "tcp", "udp", "icmp", "arp_request", "arp_reply", "other_ip", "ipv6",
          "fin", "syn", "rst", "psh", "ack", "urg", "ece", "cwr", "mask"]
EXTRA = ["ttl", "dst_port"]
COLUMNS = CONTINUOUS + BINARY + EXTRA
COL = {c: i for i, c in enumerate(COLUMNS)}
CONT_IDX = [COL[c] for c in CONTINUOUS + EXTRA]
MASK_IDX = COL["mask"]

# Feature groups (sets of columns that describe one thing); mask (real packet or padding) is always kept
GROUPS = {
    "sizes": ["frame_len"],
    "header lengths": ["ip_hlen", "l4_hlen"],
    "TCP flags": ["fin", "syn", "rst", "psh", "ack", "urg", "ece", "cwr"],
    "TCP window": ["tcp_window"],
    "protocol": ["tcp", "udp", "icmp", "arp_request", "arp_reply", "other_ip", "ipv6"],
    "direction": ["direction"],
    "IAT": ["iat"],
    "TTL": ["ttl"],
    "destination port": ["dst_port"],
}
TESTBED_GROUPS = ["TTL", "destination port", "IAT"]  # may identify the testbed (hosts, services, network timing) rather than the attack


def group_columns(groups):
    """Column indices for a set of groups, in stored order, with mask last."""
    names = {c for g in groups for c in GROUPS[g]}
    return [COL[c] for c in COLUMNS if c in names] + [MASK_IDX]


# Data -------------------------------------------------------------------------------------------------------

GOLDEN = np.uint64(0x9E3779B97F4A7C15)


def prefix_hashes(H, lengths):
    """Rolling hash: h[:, k-1] identifies the sequence of the first k packets."""
    out = np.zeros(H.shape, np.uint64)
    current = np.zeros(len(H), np.uint64)
    for k in range(L):
        current = np.where(lengths > k, (current * GOLDEN) ^ H[:, k], current)
        out[:, k] = current
    return out


def load_data():
    """Inputs (memory-mapped), one row per segment, labels, lengths and prefix hashes (headers, headers + IAT)."""
    X = np.load(DATA_DIR / "sequences.npy", mmap_mode="r")
    segments = pd.read_parquet(DATA_DIR / "segments.parquet")
    lengths = segments["length"].to_numpy()
    cache = DATA_DIR / "prefix_hashes.npz"
    if not cache.exists():
        h = np.load(DATA_DIR / "packet_hashes.npz")
        np.savez(cache, header=prefix_hashes(h["header"], lengths), timed=prefix_hashes(h["timed"], lengths))
    p = np.load(cache)
    segments["hash"] = p["header"][:, -1]
    return X, segments, segments["attack"].to_numpy(), lengths, p["header"], p["timed"]


def read_track(name):
    return pd.read_parquet(TRACK_DIR / f"{name}.parquet")


def split_index(track):
    return {part: np.sort(track.loc[track["split"] == part, "segment"].to_numpy()) for part in PART_CODE}


def unseen(eval_idx, train_idx, hashes):
    """Rows of eval_idx whose sequence never appears in training: "unseen" = "all" with those rows removed."""
    return ~np.isin(hashes[eval_idx], hashes[train_idx])


def fit_stats(X, train_idx):
    """Mean and std of continuous columns over real packets of the training segments."""
    sample = np.array(X[train_idx[::5]])
    real = sample[sample[:, :, MASK_IDX] > 0]
    return real[:, CONT_IDX].mean(axis=0), real[:, CONT_IDX].std(axis=0) + 1e-6


def load(X, idx, stats, cols=None):
    """Standardized inputs for segments idx, padding kept at exactly zero."""
    mu, sd = stats
    x = np.array(X[idx])
    x[:, :, CONT_IDX] = (x[:, :, CONT_IDX] - mu) / sd
    x *= x[:, :, [MASK_IDX]]
    return x if cols is None else x[:, :, cols].copy()


# Model ------------------------------------------------------------------------------------------------------

class OnlineGRU(tnn.Module):
    """Causal by construction: the output at packet k only depends on packets 1..k."""

    def __init__(self, n_inputs, hidden, layers):
        super().__init__()
        self.gru = tnn.GRU(n_inputs, hidden, num_layers=layers, batch_first=True)
        self.head = tnn.Linear(hidden, 1)

    def forward(self, x):
        h, _ = self.gru(x)
        return self.head(h).squeeze(-1), h  # logit of P(benign) and h_k, for every k


GRU_GRID = {"64": (64, 1), "128": (128, 1), "128x2": (128, 2)}
BATCH, MAX_EPOCHS, PATIENCE = 512, 10, 2


def masked_loss(model, xb, yb, attack_weight, mask_at):
    """Class-weighted loss over real packets."""
    logits, _ = model(xb)
    mask = xb[:, :, mask_at]
    target = (1 - yb).unsqueeze(1).expand_as(logits)
    loss = tnn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    weight = mask * (1 + (attack_weight - 1) * yb.unsqueeze(1))
    return (loss * weight).sum() / weight.sum()


@torch.no_grad()
def eval_loss(model, x, y, attack_weight, mask_at):
    model.eval()
    parts = [(masked_loss(model, torch.from_numpy(x[i:i + 4096]), torch.from_numpy(y[i:i + 4096]), attack_weight, mask_at).item(),
              len(x[i:i + 4096])) for i in range(0, len(x), 4096)]
    return sum(l * n for l, n in parts) / max(len(x), 1)


def fit_gru(name, x_train, y_train, x_val, y_val, hidden, layers, seed=SEED, log=print):
    """Early stopping on the class-weighted validation loss (all validation rows). Weights are cached by name."""
    torch.manual_seed(seed)
    mask_at = x_train.shape[-1] - 1  # mask is always the last column
    model = OnlineGRU(x_train.shape[-1], hidden, layers)
    path = MODEL_DIR / f"{name}.pt"
    if path.exists():
        saved = torch.load(path)
        model.load_state_dict(saved["state"])
        return model, saved["val_loss"]
    weight = float((y_train == 0).sum() / (y_train == 1).sum())
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed)
    best, best_state, stale = np.inf, None, 0
    t0 = time.time()
    for epoch in range(MAX_EPOCHS):
        model.train()
        for b in np.array_split(rng.permutation(len(x_train)), len(x_train) // BATCH + 1):
            n = int(x_train[b, :, mask_at].sum(axis=1).max())
            opt.zero_grad()
            loss = masked_loss(model, torch.from_numpy(x_train[b, :n]), torch.from_numpy(y_train[b]), weight, mask_at)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        val_loss = eval_loss(model, x_val, y_val, weight, mask_at)
        log(f"{name}: epoch {epoch + 1}, validation loss {val_loss:.4f}, {time.time() - t0:.0f}s")
        if val_loss < best:
            best, stale = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    model.load_state_dict(best_state)
    seconds = time.time() - t0
    torch.save({"state": best_state, "val_loss": best, "weight": weight, "epochs": epoch + 1, "seconds": seconds}, path)
    # Actual training time of every model, for the report (appended, one row per model)
    times = MODEL_DIR / "training_times.csv"
    pd.DataFrame([{"model": name, "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "epochs": epoch + 1,
                   "seconds": round(seconds), "training segments": len(x_train), "hidden": hidden, "layers": layers}]
                 ).to_csv(times, mode="a", header=not times.exists(), index=False)
    return model, best


@torch.no_grad()
def predict(model, x, with_hidden=False):
    """Logits (samples x L) and optionally hidden states (samples x L x hidden)."""
    model.eval()
    zs, hs = [], []
    for i in range(0, len(x), 4096):
        z, h = model(torch.from_numpy(x[i:i + 4096]))
        zs.append(z.numpy())
        if with_hidden:
            hs.append(h.numpy())
    return (np.concatenate(zs), np.concatenate(hs)) if with_hidden else np.concatenate(zs)


# Metrics ----------------------------------------------------------------------------------------------------

def at_prefix(values, lengths, k):
    """The value of each sample at packet k, or at its last packet if it is shorter."""
    return values[np.arange(len(values)), np.minimum(k, lengths) - 1]


def scores(y, pred):
    """Forced answers: every row gets attack (1) or benign (0)."""
    return {
        "accuracy": accuracy_score(y, pred),
        "macro F1": f1_score(y, pred, average="macro"),
        "FPR": (pred[y == 0] == 1).mean(),
        "missed attacks": (pred[y == 1] == 0).mean(),
    }


def ceiling(hashes, y):
    """Jerabek et al. Eq. 1 on these rows: an oracle that gives each cluster of identical sequences its majority label
    (attack on a tie). MaxAcc and MaxF1 are its accuracy and macro-F1; minimum FPR follows the paper (every mixed
    cluster called attack); minimum missed attacks is the oracle's."""
    share = pd.DataFrame({"h": hashes, "attack": y}).groupby("h")["attack"].transform("mean").to_numpy()
    s = scores(y, (share >= 0.5).astype(int))
    return {"rows": len(y), "MaxAcc": s["accuracy"], "MaxF1": s["macro F1"],
            "minimum FPR": (share[y == 0] > 0).mean() if (y == 0).any() else np.nan,
            "minimum missed attacks": s["missed attacks"],
            "rows in conflict": int(((share > 0) & (share < 1)).sum())}


# Re-cutting packets into segments of another length (segment-length experiment) ------------------------------

HEADER_KEY = ["direction", "frame_len", "protocol", "tcp_flags", "tcp_window", "ip_hlen", "l4_hlen", "arp_op", "is_ipv6"]


def packet_features(p, pos):
    """Stored columns for packets of one capture, ordered by flow and time (as in sequences.npy)."""
    iat = p["ts"].diff().where(pos != 0, 0).fillna(0).clip(lower=0).to_numpy()
    x = np.zeros((len(p), len(COLUMNS)), dtype=np.float32)
    x[:, COL["frame_len"]] = np.log1p(p["frame_len"])
    x[:, COL["tcp_window"]] = np.log1p(p["tcp_window"])
    x[:, COL["ip_hlen"]] = p["ip_hlen"]
    x[:, COL["l4_hlen"]] = p["l4_hlen"]
    x[:, COL["iat"]] = np.log1p(iat * 1e6)
    x[:, COL["direction"]] = p["direction"]
    for name, code in [("tcp", 1), ("udp", 2), ("icmp", 3), ("other_ip", 5)]:
        x[:, COL[name]] = p["protocol"] == code
    x[:, COL["arp_request"]] = p["arp_op"] == 1
    x[:, COL["arp_reply"]] = p["arp_op"] == 2
    x[:, COL["ipv6"]] = p["is_ipv6"]
    for bit, name in enumerate(["fin", "syn", "rst", "psh", "ack", "urg", "ece", "cwr"]):
        x[:, COL[name]] = (p["tcp_flags"].to_numpy() >> bit) & 1
    x[:, COL["mask"]] = 1
    x[:, COL["ttl"]] = p["ttl"]
    x[:, COL["dst_port"]] = np.log1p(p["dst_port"])
    return x


def cut_segments(length, captures):
    """Raw stored columns (segments x length x columns), and one row per segment with label, start and sequence hash."""
    xs, metas, offset = [], [], 0
    for order, capture in enumerate(captures):
        p = pd.read_parquet(DATA_DIR / "packets" / f"{CAPTURES[capture]}.parquet").sort_values(["flow", "ts"], kind="stable").reset_index(drop=True)
        pos = p.groupby("flow").cumcount().to_numpy() % length
        seg = np.cumsum(pos == 0) - 1
        p["protocol"] = np.select([p["ip_proto"] == 6, p["ip_proto"] == 17, p["ip_proto"].isin([1, 58]),
                                   p["ethertype"] == 0x806, p["ip_proto"] > 0], [1, 2, 3, 4, 5], 0)
        p["is_ipv6"] = (p["ethertype"] == 0x86DD).astype(np.int8)
        x = np.zeros((seg[-1] + 1, length, len(COLUMNS)), np.float32)
        x[seg, pos] = packet_features(p, pos)
        H = np.zeros((seg[-1] + 1, length), np.uint64)
        H[seg, pos] = pd.util.hash_pandas_object(p[HEADER_KEY], index=False).to_numpy()
        m = pd.DataFrame({"segment": seg, "label": p["label"], "ts": p["ts"], "flow": p["flow"]}).groupby("segment").agg(
            attack=("label", "max"), length=("label", "size"), start=("ts", "min"), flow=("flow", "first")).reset_index()
        current = np.zeros(len(m), np.uint64)
        for k in range(length):
            current = np.where(m["length"].to_numpy() > k, (current * GOLDEN) ^ H[:, k], current)
        m["hash"], m["capture"], m["capture_order"] = current, capture, order
        m["segment"] += offset
        offset += len(m)
        xs.append(x)
        metas.append(m)
    return np.concatenate(xs), pd.concat(metas, ignore_index=True)


# Parsing any capture file the way the training data was parsed (used for inference on new data) -------------

PACKET_DTYPE = np.dtype([
    ("ts", "f8"), ("flow", "u4"), ("direction", "u1"), ("frame_len", "u4"), ("ethertype", "u2"),
    ("ip_proto", "u1"), ("ttl", "u1"), ("ip_hlen", "u1"), ("l4_hlen", "u1"), ("tcp_flags", "u1"),
    ("tcp_window", "u2"), ("udp_len", "u2"), ("icmp_type", "u1"), ("icmp_code", "u1"),
    ("arp_op", "u2"), ("dst_port", "u2"),
])


def parse_pcap(stream, limit=None):
    """One row per packet of a PCAP or PCAPNG stream, in file order; the same fields and flow rule as the training data.
    IPs, MACs and ports only assign the flow id; they are not stored."""
    import dpkt
    reader = dpkt.pcapng.Reader(stream) if stream.peek(4)[:4] == b"\n\r\r\n" else dpkt.pcap.Reader(stream)
    rows, flows = [], {}  # canonical key -> (flow id, endpoint that sent the first packet)
    for i, (ts, buf) in enumerate(reader):
        if limit is not None and i >= limit:
            break
        frame_len, proto, ttl, ip_hlen, l4_hlen, flags, window, udp_len = len(buf), 0, 0, 0, 0, 0, 0, 0
        icmp_type, icmp_code, arp_op, sport, dport, ethertype = 0, 0, 0, 0, 0, 0
        src, dst = b"", b""
        try:
            eth = dpkt.ethernet.Ethernet(buf)
        except dpkt.UnpackError:
            eth = None
        if eth is not None:
            ethertype = eth.type if eth.type >= 0x600 else 0  # below 0x600 it is an 802.3 length
            ip = eth.data
            if isinstance(ip, (dpkt.ip.IP, dpkt.ip6.IP6)):
                src, dst, proto = ip.src, ip.dst, ip.p
                if isinstance(ip, dpkt.ip.IP):
                    ttl, ip_hlen = ip.ttl, ip.hl * 4
                    frame_len = (ip.len + 14) if ip.len else frame_len  # captures may cut frames short
                else:
                    ttl, ip_hlen = ip.hlim, 40
                    frame_len = ip.plen + 54
                l4 = ip.data
                if isinstance(l4, dpkt.tcp.TCP):
                    sport, dport, flags, window, l4_hlen = l4.sport, l4.dport, l4.flags, l4.win, l4.off * 4
                elif isinstance(l4, dpkt.udp.UDP):
                    sport, dport, udp_len, l4_hlen = l4.sport, l4.dport, l4.ulen, 8
                elif isinstance(l4, (dpkt.icmp.ICMP, dpkt.icmp6.ICMP6)):
                    icmp_type, icmp_code, l4_hlen = l4.type, l4.code, 8
            else:
                src, dst = eth.src, eth.dst
                if isinstance(ip, dpkt.arp.ARP):
                    arp_op = ip.op
        a, b = (src, sport), (dst, dport)
        key = (a, b, proto, ethertype) if a <= b else (b, a, proto, ethertype)
        flow_id, first_sender = flows.setdefault(key, (len(flows), a))
        rows.append((ts, flow_id, a != first_sender, frame_len, ethertype, proto, ttl, ip_hlen, l4_hlen,
                     flags, window, udp_len, icmp_type, icmp_code, arp_op, dport))
    return pd.DataFrame(np.array(rows, dtype=PACKET_DTYPE))


def segment_packets(p, length=L):
    """Stored columns (segments x length x columns) and one row per segment (flow, start, length, label if given),
    cutting each flow into consecutive segments of `length` packets, as for the training data."""
    p = p.assign(packet=np.arange(len(p))).sort_values(["flow", "ts"], kind="stable").reset_index(drop=True)
    pos = p.groupby("flow").cumcount().to_numpy() % length
    seg = np.cumsum(pos == 0) - 1
    p["protocol"] = np.select([p["ip_proto"] == 6, p["ip_proto"] == 17, p["ip_proto"].isin([1, 58]),
                               p["ethertype"] == 0x806, p["ip_proto"] > 0], [1, 2, 3, 4, 5], 0)
    p["is_ipv6"] = (p["ethertype"] == 0x86DD).astype(np.int8)
    x = np.zeros((seg[-1] + 1, length, len(COLUMNS)), np.float32)
    x[seg, pos] = packet_features(p, pos)
    agg = {"flow": ("flow", "first"), "start": ("ts", "min"), "length": ("ts", "size"), "first_packet": ("packet", "min")}
    if "label" in p:
        agg["attack"] = ("label", "max")
    meta = p.assign(segment=seg).groupby("segment").agg(**agg).reset_index()
    return x, meta


def apply_track_rule(rows, track):
    """The cleaning rule of each track, on any set of segments (rows need hash and attack, in time order)."""
    seen = rows.groupby("hash")["attack"].agg(["min", "max"])
    conflict = rows["hash"].isin(seen.index[seen["min"] != seen["max"]])
    return {"A": rows, "B": rows[~conflict]}[track]


def time_split(g, cuts=(0.6, 0.8)):
    """Earliest share train, next share validation, the rest test, within one capture and class."""
    rank = (np.arange(len(g)) + 0.5) / len(g)
    if len(cuts) == 1:
        return pd.Series(np.where(rank < cuts[0], "train", "val"), index=g.index)
    return pd.Series(np.select([rank < cuts[0], rank < cuts[1]], ["train", "val"], "test"), index=g.index)


# Calibration, OOD and the decision rule ----------------------------------------------------------------------

BUCKETS = {"1": (1, 1), "2-3": (2, 3), "4-10": (4, 10), "11+": (11, L)}
STEP_BUCKET = np.array([next(i for i, (lo, hi) in enumerate(BUCKETS.values()) if lo <= k <= hi) for k in range(1, L + 1)])
K_REPORT = [1, 3, 5, 10, 20, 50]  # decision cut-offs reported; K = L is the working one
OOD_BUDGET = 0.05                 # at most this share of known validation segments ever flagged
OOD_Q_MIN = 0.90                  # lowest per-step quantile an OOD limit may sit at
TARGET_HIGH, TARGET_LOW = 0.99, 0.999  # share of blocks that are attacks; share of Low answers that are benign
ANSWERS = ["High", "Low", "Medium", "OOD"]


def real_steps(lengths_, width=L):
    return np.arange(width)[None, :] < lengths_[:, None]


def nll(logit, y):
    """y = 1 means benign: the logit is of P(benign)."""
    return np.mean(np.logaddexp(0, logit) - y * logit)


def fit_temperatures(z, lengths_, y):
    """One temperature per prefix bucket on the first validation half, capped so it never sharpens (T >= 1)."""
    from scipy.optimize import minimize_scalar
    width = z.shape[1]
    bucket = bucket_of(width)
    real = real_steps(lengths_, width)
    benign = np.repeat((1 - y)[:, None], width, axis=1)
    fitted = np.ones(len(BUCKETS))
    for b in range(len(BUCKETS)):
        m = real & (bucket == b)[None, :]
        if m.any():
            fitted[b] = minimize_scalar(lambda t: nll(z[m] / t, benign[m]), bounds=(0.05, 20), method="bounded").x
    return np.maximum(fitted, 1.0)[bucket], fitted[bucket]


def bucket_of(width):
    """Prefix bucket of each step for segments of any length; steps past L fall in the last bucket."""
    return np.r_[STEP_BUCKET, np.full(max(0, width - L), len(BUCKETS) - 1)][:width]


def p_attack(z, T):
    return 1 / (1 + np.exp(z / T))  # z is the logit of P(benign)


def running_mean(score, lengths_):
    """Mean OOD score of each segment's packets so far; at packet 1 it is that packet's score."""
    real = real_steps(lengths_, score.shape[1])
    mean = np.cumsum(np.where(real, score, 0), axis=1) / np.arange(1, score.shape[1] + 1)
    return np.where(real, mean, -np.inf).astype(np.float32)


DETECTORS = ["forest", "mahalanobis"]  # a segment is OOD if any detector flags it


def fit_detectors(states, labels):
    """OOD detectors on training hidden states (one row per real packet, with its segment's label): an Isolation
    Forest, and Mahalanobis distance to the nearer class mean with a shared covariance (Lee et al., 2018)."""
    from sklearn.ensemble import IsolationForest
    forest = IsolationForest(n_estimators=200, random_state=SEED, n_jobs=-1).fit(states)
    means = np.stack([states[labels == c].mean(axis=0) for c in (0, 1)])
    centred = states - means[labels]
    cov = centred.T @ centred / len(states) + 1e-6 * np.eye(states.shape[1])
    return {"forest": forest, "mahalanobis": {"means": means.astype(np.float32), "precision": np.linalg.inv(cov).astype(np.float32)}}


def detector_scores(detectors, h):
    """Per-packet score of each detector for hidden states h (rows of packets); higher means more unusual."""
    m = detectors["mahalanobis"]
    dist = np.empty(len(h), np.float32)
    for i in range(0, len(h), 200_000):  # in chunks, so memory stays small
        c = h[i:i + 200_000]
        dist[i:i + len(c)] = np.min([(((c - mu) @ m["precision"]) * (c - mu)).sum(axis=1) for mu in m["means"]], axis=0)
    return {"forest": -detectors["forest"].score_samples(h), "mahalanobis": dist}


def ood_running_means(detectors, h, lengths_):
    """Running mean of each detector's score over each segment's packets (h: segments x steps x hidden)."""
    real = real_steps(lengths_, h.shape[1])
    out = {}
    for name, s in detector_scores(detectors, h[real]).items():
        full = np.full(real.shape, -np.inf, np.float32)
        full[real] = s
        out[name] = running_mean(full, lengths_)
    return out


def ood_flags(means, limits):
    """OOD flag per step: any detector's running mean over its limit for that step's bucket."""
    return np.logical_or.reduce([means[d] > np.asarray(limits[d])[STEP_BUCKET][None, :means[d].shape[1]] for d in DETECTORS])


def ood_thresholds(means, lengths_, K, budget=OOD_BUDGET):
    """Per-bucket limits for every detector at one shared quantile q, the lowest q (found by bisection) at which at most
    `budget` of these known segments are ever flagged by any detector in their first K packets."""
    real = real_steps(lengths_)
    pools = {d: [means[d][real & (STEP_BUCKET == b)[None, :]] for b in range(len(BUCKETS))] for d in DETECTORS}

    def at(q):
        limits = {d: np.array([np.quantile(v, q) for v in pools[d]]) for d in DETECTORS}
        flagged = (ood_flags({d: means[d][:, :K] for d in DETECTORS}, limits) & real[:, :K]).any(axis=1).mean()
        return limits, flagged

    lo, hi = OOD_Q_MIN, 1.0
    for _ in range(25):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if at(mid)[1] <= budget else (mid, hi)
    return at(hi)[0], hi


def route(p, flagged, lengths_, tau_attack, tau_benign, K):
    """The answer for each segment and the packet it came at. OOD and High end the decision at once (OOD first);
    Low only at packet K or the segment's last packet; anything else is Medium."""
    live = real_steps(lengths_, p.shape[1])[:, :K]
    flag = flagged[:, :K] & live
    high = (p[:, :K] >= tau_attack) & live
    event = flag | high
    early = event.any(axis=1)
    rows = np.arange(len(p))
    last = np.minimum(lengths_, K) - 1
    step = np.where(early, event.argmax(axis=1), last)
    low = p[rows, last] <= tau_benign
    outcome = np.select([early & flag[rows, step], early, low], ["OOD", "High", "Low"], "Medium")
    return outcome, step + 1


def precision_bound(hits, n, level=0.95):
    """One-sided lower confidence bound (Clopper-Pearson) on a precision of hits/n."""
    from scipy.stats import beta
    return float(beta.ppf(1 - level, hits, n - hits + 1)) if n and hits else 0.0


def tune_targets(p, flagged, lengths_, y, K, target_high=TARGET_HIGH, target_low=TARGET_LOW):
    """Precision-target thresholds on the second validation half, with the full decision rule:
    the lowest tau_attack at which at least target_high of blocks are attacks, then, given it, the highest
    tau_benign at which at least target_low of Low answers are benign."""
    tau_a, info = 2.0, {}  # target not met: never block
    for t in 1 - np.logspace(-0.3, -7, 140):  # increasing cut-offs, 0.5 to 1 - 1e-7
        out, _ = route(p, flagged, lengths_, t, -1.0, K)
        n, hits = int((out == "High").sum()), int(((out == "High") & (y == 1)).sum())
        if n and hits / n >= target_high:
            tau_a, info = float(t), {"blocks": n, "block precision": hits / n, "block precision, lower 95%": precision_bound(hits, n)}
            break
    tau_b, info_b = -1.0, {}  # target not met: never answer Low
    for t in [*np.logspace(-0.3, -7, 140), 0.0]:  # 0.0 last: models such as k-NN can output exactly 0  # decreasing cut-offs, 0.5 down to 1e-7
        out, _ = route(p, flagged, lengths_, tau_a, t, K)
        n, hits = int((out == "Low").sum()), int(((out == "Low") & (y == 0)).sum())
        if n and hits / n >= target_low:
            tau_b, info_b = float(t), {"Low answers": n, "Low precision": hits / n, "Low precision, lower 95%": precision_bound(hits, n)}
            break
    return {"tau_attack": tau_a, "tau_benign": tau_b, "high target met": bool(info), "low target met": bool(info_b),
            **info, **info_b}


def build_unit(X, model, cols, stats, parts, lengths, y_all, hidden_train_part="train sample", forest_steps=200_000):
    """Everything one deployment needs, from one track's data only: temperatures (first validation half "cal"),
    OOD detectors (training hidden states), OOD limits and High/Low cut-offs per K (second validation half "thr").
    Returns the unit plus logits, probabilities, OOD scores, flags and lengths for every part, for measuring."""
    z, n = {}, {p: lengths[i] for p, i in parts.items()}
    y = {p: y_all[i] for p, i in parts.items()}
    mean = {}
    detectors = None
    order = [hidden_train_part] + [p for p in parts if p != hidden_train_part]
    for p in order:
        z[p], h = predict(model, load(X, parts[p], stats, cols), with_hidden=True)
        if p == hidden_train_part:
            real = real_steps(n[p])
            states = h[real]
            labels = np.repeat(y[p], real.sum(axis=1)).astype(int)
            pick = np.random.default_rng(SEED).choice(len(states), min(forest_steps, len(states)), replace=False)
            detectors = fit_detectors(states[pick], labels[pick])
            del states
        else:
            mean[p] = ood_running_means(detectors, h, n[p])
        del h
    T, T_fitted = fit_temperatures(z["cal"], n["cal"], y["cal"])
    prob = {p: p_attack(z[p], T) for p in z}
    tuned = {}
    for K in K_REPORT:
        limits, q = ood_thresholds(mean["thr"], n["thr"], K)
        flags = {p: ood_flags(mean[p], limits) for p in mean}
        tuned[K] = {"ood limits": limits, "ood quantile": q, **tune_targets(prob["thr"], flags["thr"], n["thr"], y["thr"], K), "flags": flags}
    unit = {"columns": cols, "stats": stats, "temperatures": T, "fitted temperatures": T_fitted, "detectors": detectors,
            "thresholds": {K: {k: v for k, v in t.items() if k != "flags"} for K, t in tuned.items()}}
    return {**unit, "net": model, "z": z, "n": n, "y": y, "prob": prob, "mean": mean, "tuned": tuned}


def decide(u, p, K=L):
    """Route part p of a built unit at cut-off K."""
    t = u["tuned"][K]
    return route(u["prob"][p], t["flags"][p], u["n"][p], t["tau_attack"], t["tau_benign"], K)


def answer_counts(outcome, y):
    """Segments in each answer by true class: every rate below is computed from these."""
    return {f"{c}: {a}": int(((outcome == a) & (y == v)).sum()) for c, v in [("attack", 1), ("benign", 0)] for a in ANSWERS}


def known_metrics(outcome, y, packets=None):
    """Pipeline results, including "when the model says I know" (High or Low only)."""
    att, ben = y == 1, y == 0
    said = np.isin(outcome, ["High", "Low"])
    high, low = outcome == "High", outcome == "Low"
    pred = (outcome[said] == "High").astype(int)
    r = {"segments": len(y), "attack segments": int(att.sum()), "benign segments": int(ben.sum()),
         "coverage": said.mean(),
         "attacks blocked": high[att].mean() if att.any() else np.nan,
         "attacks called benign": low[att].mean() if att.any() else np.nan,
         "benign blocked": high[ben].mean() if ben.any() else np.nan,
         "benign called Low": low[ben].mean() if ben.any() else np.nan,
         "block precision": (high & att).sum() / high.sum() if high.any() else np.nan,
         "Low precision": (low & ben).sum() / low.sum() if low.any() else np.nan,
         "accuracy when answering": (pred == y[said]).mean() if said.any() else np.nan,
         "macro F1 when answering": f1_score(y[said], pred, average="macro") if said.any() and len(set(y[said])) > 1 else np.nan,
         "Medium": (outcome == "Medium").mean(), "OOD": (outcome == "OOD").mean()}
    if packets is not None:
        r["mean packets to block"] = packets[high & att].mean() if (high & att).any() else np.nan
        for k in [1, 3, 5, 10, 20, 50]:
            r[f"attacks blocked by packet {k}"] = (high & att & (packets <= k)).sum() / max(att.sum(), 1)
    return {**r, **answer_counts(outcome, y)}


def at_rate(attacks_blocked, benign_blocked, rate):
    """Block precision if attacks made up `rate` of traffic (Bayes' rule on the same per-class rates)."""
    num = attacks_blocked * rate
    return num / (num + benign_blocked * (1 - rate)) if num + benign_blocked * (1 - rate) > 0 else np.nan


def flow_bootstrap(outcome, y, flow_key, n_boot=200, seed=SEED):
    """95% intervals for the main rates, resampling whole flows (Poisson weights per flow)."""
    d = pd.DataFrame({"f": flow_key, "att": y == 1, "ben": y == 0, "high": outcome == "High", "low": outcome == "Low",
                      "said": np.isin(outcome, ["High", "Low"])})
    d = d.assign(hit_h=d["high"] & d["att"], hit_l=d["low"] & d["ben"])
    g = d.groupby("f")[["att", "ben", "high", "low", "said", "hit_h", "hit_l"]].sum().to_numpy(float)
    w = np.random.default_rng(seed).poisson(1.0, (n_boot, len(g)))
    s = w @ g  # resampled totals: att, ben, high, low, said, hit_h, hit_l
    with np.errstate(invalid="ignore", divide="ignore"):
        stats = {"attacks blocked": s[:, 5] / s[:, 0], "block precision": s[:, 5] / s[:, 2],
                 "Low precision": s[:, 6] / s[:, 3], "coverage": s[:, 4] / (s[:, 0] + s[:, 1])}
    return {f"{k}, 95% CI": (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) for k, v in stats.items()}


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, default=float))


def load_json(path):
    return json.loads(Path(path).read_text())
