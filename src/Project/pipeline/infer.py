"""Classify a new capture with a trained model from artifacts/, exactly as in evaluation; nothing is re-fitted.

    python pipeline/infer.py capture.pcap                          # Track A, model without the testbed group, K = 50
    python pipeline/infer.py capture.pcapng.gz --track B --model main --K 20
    python pipeline/infer.py capture.pcap --labels labels.csv      # also score the answers

The capture (PCAP or PCAPNG, optionally gzipped) is parsed into flows and cut into 50-packet segments as the training
data was. Each segment gets one answer: High (block), Low (benign for this segment), Medium (not sure) or OOD (unlike the
training traffic to the Isolation Forest or the Mahalanobis detector), and the packet it came at. Labels, if given, are one 0/1 per packet in file order (a header row
and index columns are allowed; the last column is used). Output: one row per segment in a CSV, and a summary.
"""
import argparse
import gzip
import io
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import torch

import kitsune_lib as kl

CHUNK = 20_000  # segments scored at a time, so memory stays small on large captures


def open_capture(path):
    raw = open(path, "rb")
    stream = gzip.open(raw) if raw.read(2) == b"\x1f\x8b" else None
    raw.seek(0)
    return io.BufferedReader(stream if stream is not None else raw)


def read_labels(path):
    opener = gzip.open if Path(path).suffix == ".gz" else open
    with opener(path, "rb") as f:
        labels = pd.read_csv(f, header=None, low_memory=False).iloc[:, -1]
    return pd.to_numeric(labels, errors="coerce").dropna().to_numpy(dtype=np.int8)


def load_artifact(folder):
    """The model, its unit and its config from one artifact folder. Stops unless the three belong together (the unit and
    the config name the hash of these weights); warns when a library differs from the versions it was exported with."""
    import sklearn
    saved = torch.load(folder / "model.pt")
    net = kl.OnlineGRU(saved["n_inputs"], saved["hidden"], saved["layers"])
    net.load_state_dict(saved["state"])
    unit, config = joblib.load(folder / "unit.joblib"), kl.load_json(folder / "config.json")
    kl.check_unit(unit, saved["state"], folder / "unit.joblib")
    if config.get("model weights") != unit["model weights"]:
        raise RuntimeError(f"{folder / 'config.json'} describes other model weights than model.pt and unit.joblib")
    now = {"torch": torch.__version__, "scikit-learn": sklearn.__version__, "numpy": np.__version__}
    for lib, version in now.items():
        if config["versions"].get(lib) != version:
            warnings.warn(f"{lib} {version} differs from {config['versions'].get(lib)}, the version this model was "
                          f"exported with; its answers may differ (the unit holds a scikit-learn Isolation Forest)",
                          stacklevel=2)
    return net.eval(), unit, config


def classify(x, n, net, unit, K):
    """Answer and decision packet for every segment, through the same steps as evaluation."""
    t = unit["thresholds"][K]
    answers, packets, p_last = [], [], []
    for i in range(0, len(x), CHUNK):
        xs = kl.load(x, np.arange(i, min(i + CHUNK, len(x))), unit["stats"], unit["columns"])
        ns = n[i:i + CHUNK]
        z, h = kl.predict(net, xs, with_hidden=True)
        flags = kl.ood_flags(kl.ood_running_means(unit["detectors"], h, ns), t["ood limits"])
        del h
        p = kl.p_attack(z, unit["temperatures"])
        out, at = kl.route(p, flags, ns, t["tau_attack"], t["tau_benign"], K)
        answers.append(out)
        packets.append(at)
        p_last.append(kl.at_prefix(p, ns, K))
    return np.concatenate(answers), np.concatenate(packets), np.concatenate(p_last)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture")
    ap.add_argument("--labels", help="per-packet 0/1 labels, in file order, to score the answers")
    ap.add_argument("--track", default="A", choices=["A", "B"])
    ap.add_argument("--model", default="no_testbed", choices=["main", "no_testbed"],
                    help="default: the model without TTL, destination port and IAT, the one chosen for deployment")
    ap.add_argument("--K", type=int, default=kl.L, choices=kl.K_REPORT, help="last packet at which a Low answer is given")
    ap.add_argument("--artifacts", default=str(ROOT / "artifacts"))
    ap.add_argument("--out", help="output CSV (default: runs/inference/<capture>_<track>_<model>_K<K>.csv)")
    args = ap.parse_args()

    folder = Path(args.artifacts) / args.track / args.model
    net, unit, config = load_artifact(folder)
    t0 = time.time()
    with open_capture(args.capture) as stream:
        p = kl.parse_pcap(stream)
    print(f"{args.capture}: {len(p):,} packets in {p['flow'].nunique():,} flows, parsed in {time.time() - t0:.0f}s")
    if args.labels:
        labels = read_labels(args.labels)
        if len(labels) != len(p):
            raise SystemExit(f"{len(labels):,} labels for {len(p):,} packets")
        p["label"] = labels
    x, meta = kl.segment_packets(p)
    answer, packet, p_attack = classify(x, meta["length"].to_numpy(), net, unit, args.K)
    meta = meta.assign(answer=answer, decided_at_packet=packet, p_attack_at_K=p_attack.round(6))
    name = Path(args.capture).name.split(".")[0]
    out = Path(args.out) if args.out else ROOT / "runs" / "inference" / f"{name}_{args.track}_{args.model}_K{args.K}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    meta.to_csv(out, index=False)

    print(f"model: Track {args.track}, {config['model']} ({', '.join(config['feature groups'])}), K = {args.K}")
    print(f"{len(meta):,} segments: " + ", ".join(f"{a} {(answer == a).mean():.1%}" for a in kl.ANSWERS))
    if args.labels:
        r = kl.known_metrics(answer, meta["attack"].to_numpy(), packet)
        for k in ["attack segments", "benign segments", "coverage", "attacks blocked", "block precision",
                  "attacks called benign", "Low precision", "accuracy when answering", "macro F1 when answering",
                  "Medium", "OOD"]:
            v = r[k]
            print(f"  {k:26s} {v:,}" if isinstance(v, (int, np.integer)) else f"  {k:26s} {v:.4f}")
    print(f"per-segment answers: {out}")


if __name__ == "__main__":
    main()
