"""Stage 1: diagnostics before the split, Tracks A and B, time split.

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
# # Stage 1: data, diagnostics and tracks
#
# Reads the parsed segments built by stage 0 (`00_prepare`: PCAP parsing, flows, 50-packet segments, inputs), measures the
# pre-split diagnostics, and builds the two tracks. Every later stage reads the tracks saved here.
#
# - **Diagnostics** (informational, before any split): the distinct share and the conflicting sequences.
# - **Tracks**, each at its own full size and split by time (per capture and class: earliest 60% train, next 20%
#   validation, last 20% test):
#
# | Track | Duplicates | Conflicts | Scenario |
# |---|---|---|---|
# | A | kept | kept | labels as recorded |
# | B | kept | dropped | best case after conflicts are reviewed by hand (not every conflict may be resolvable) |
#
# Real-world results are expected to sit between A and B.
#
# **Which key defines a conflict for Track B: "headers".** Two segments conflict when their packets agree on every header
# field except TTL and destination port, and also ignoring the inter-arrival time (IAT), yet carry different labels.
# Those three fields are left out on purpose: their values come from the testbed, not from the attack (TTL: the sending
# host's operating system and hop count; destination port: the services this testbed runs; IAT: this network's link
# speed, load and capture host). A conflict that only they could resolve would be resolved by the testbed's identity, not
# by what the attack does, which is the dependence this project measures with the model without them. The rule is also
# the stricter one: it removes every conflict either model could face, and one Track B shared by both models keeps them
# on the same rows, so their difference comes from their inputs alone. Its cost is measured below: how many dropped
# segments TTL, destination port or IAT would have separated. A Track B judged per model (on each model's own key) is
# left for future work.
#
# Mirai (another network) and ARP MitM (same network) are held out of every track.


# %%
import numpy as np
import pandas as pd

from kitsune_lib import CAPTURES, KEYS, OOD_CAPTURES, L, DATA_DIR, TRACK_DIR, RESULT_DIR, load_data

X, segments, y_all, lengths, prefix_header, prefix_all = load_data()

# Flow of every segment, numbered as when the segments were built (captures in order, flows sorted by start time);
# used to resample whole flows for confidence intervals, since segments of one flow are not independent
from kitsune_lib import L as _L
flow_rows, offset = [], 0
for capture, folder in CAPTURES.items():
    p = pd.read_parquet(DATA_DIR / "packets" / f"{folder}.parquet", columns=["flow", "ts"]).sort_values(["flow", "ts"], kind="stable")
    first = (p.groupby("flow").cumcount().to_numpy() % _L) == 0
    flow_rows.append(pd.DataFrame({"segment": np.arange(offset, offset + first.sum()), "flow": p["flow"].to_numpy()[first]}))
    offset += int(first.sum())
flows = pd.concat(flow_rows, ignore_index=True)
assert offset == len(segments)
segments = segments.merge(flows, on="segment")
print(f"{len(segments):,} segments in {segments.groupby(['capture', 'flow']).ngroups:,} flows, {X.shape[2]} stored columns")


# %% [markdown]
# ## Diagnostics before the split
#
# Informational only. A sample at k is a segment cut to its first k packets. The **distinct share** counts one copy of each
# distinct sequence, labels ignored, over all samples: below 1, copies can fall on both sides of a split, so leakage shows
# up after it as the difference between the "all" and "unseen" views. **Conflicting sequences** carry both labels; they are
# candidates for manual relabelling where the true label can be established, and the rest are genuinely ambiguous.
# Two sequence keys (kitsune_lib.packet_keys), each exactly the inputs of one model, labels never included: "headers"
# (the model without TTL, destination port and IAT) and "all inputs" (headers + TTL + destination port + IAT, the
# main model).


# %%
def diagnostics(hashes, y):
    c = pd.DataFrame({"h": hashes, "attack": y}).groupby("h")["attack"].agg(n="size", n_attack="sum")
    conflict = (c["n_attack"] > 0) & (c["n_attack"] < c["n"])
    return {"distinct share": len(c) / len(y), "conflicting sequences": int(conflict.sum()),
            "conflicting share of distinct": conflict.mean(), "samples in conflict": int(c.loc[conflict, "n"].sum())}


known = ~segments["capture"].isin(OOD_CAPTURES).to_numpy()
report = pd.concat({
    key: pd.DataFrame([diagnostics(H[known, k - 1], y_all[known]) for k in range(1, L + 1)], index=range(1, L + 1))
    for key, H in [("headers", prefix_header), ("all inputs", prefix_all)]
}, names=["key", "k"])
report.to_csv(RESULT_DIR / "diagnostics.csv")
show(report.loc[(["headers", "all inputs"], [1, 2, 3, 5, 10, 20, 50]), :].round(4))


# %%
import matplotlib.pyplot as plt

FIG_DIR = DATA_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)
# Reference palette (dataviz skill): series in fixed order, recessive chrome
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2, "legend.frameon": False, "font.size": 10,
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
})

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
for key, color in [("headers", BLUE), ("all inputs", ORANGE)]:
    g = report.loc[key]
    axes[0].plot(g.index, g["distinct share"], color=color, label=key)
    axes[1].plot(g.index, g["conflicting share of distinct"] * 100, color=color, label=key)
axes[0].set_title("Distinct share of k-packet prefixes", loc="left", fontsize=11)
axes[1].set_title("Conflicting sequences, % of distinct", loc="left", fontsize=11)
for ax in axes:
    ax.set_xlabel("Packets seen, k")
    ax.legend(loc="center right")
fig.tight_layout()
fig.savefig(FIG_DIR / "data_diagnostics.png", dpi=150)
show(plt.close(fig))


# %%
# Every conflicting whole segment for manual review, most lopsided first (a 10-to-1 split hints at a mislabel;
# a balanced one at genuine ambiguity). The ratio only orders the list, it decides nothing.
in_dist = segments[known].copy()
in_dist["capture_order"] = in_dist["capture"].map({c: i for i, c in enumerate(CAPTURES)})
in_dist = in_dist.sort_values(["capture_order", "start"], kind="stable")
copies = in_dist.groupby("hash").agg(copies=("segment", "size"), captures=("capture", "nunique"),
                                     benign=("attack", lambda a: int((a == 0).sum())), attack=("attack", "sum"),
                                     length=("length", "first"))
review = copies[(copies["benign"] > 0) & (copies["attack"] > 0)]
review = review.assign(minority_share=review[["benign", "attack"]].min(axis=1) / review["copies"]).sort_values("minority_share")
review.to_csv(RESULT_DIR / "conflicts_for_review.csv")
print(f"{len(review):,} conflicting sequences ({int(review['copies'].sum()):,} segments) written for review")
show(review.head(10))


# %% [markdown]
# ## Tracks
#
# Built from the known captures in time order. Each track keeps its own size: nothing is sampled to match the other.
# The split is by time within each capture and class.


# %%
conflict_hashes = review.index
in_conflict = in_dist["hash"].isin(conflict_hashes)
tracks = {"A": in_dist, "B": in_dist[~in_conflict]}

# The cost of judging conflicts on "headers": dropped segments that TTL, destination port or IAT would have separated
# (not in conflict under "all inputs")
by_all = in_dist.groupby("hash_all")["attack"].agg(["min", "max"])
in_conflict_all = in_dist["hash_all"].isin(by_all.index[by_all["min"] != by_all["max"]])
b_drops = pd.DataFrame([{"segments dropped from Track B": int(in_conflict.sum()),
                         "of which also in conflict under all inputs": int((in_conflict & in_conflict_all).sum()),
                         "of which TTL, destination port or IAT would separate": int((in_conflict & ~in_conflict_all).sum()),
                         "known segments": len(in_dist)}])
b_drops.to_csv(RESULT_DIR / "track_b_drops.csv", index=False)
show(b_drops.T)


def time_split(g):
    rank = (np.arange(len(g)) + 0.5) / len(g)
    return pd.Series(np.select([rank < 0.6, rank < 0.8], ["train", "val"], "test"), index=g.index)


summary = []
for name, t in tracks.items():
    t = t.sort_values(["capture_order", "attack", "start"], kind="stable").copy()
    t["split"] = pd.concat([time_split(g) for _, g in t.groupby(["capture", "attack"], sort=False)])
    for _, g in t.groupby(["capture", "attack"]):  # time order holds inside each capture and class
        s = g.groupby("split")["start"]
        assert s.max()["train"] <= s.min()["val"] and s.max()["val"] <= s.min()["test"]
    t[["segment", "capture", "flow", "attack", "length", "start", "hash", "split"]].to_parquet(TRACK_DIR / f"{name}.parquet", index=False)
    tracks[name] = t
    train_h = {key: set(t.loc[t["split"] == "train", col]) for key, col in KEYS.items()}
    for part in ["train", "val", "test"]:
        d = t[t["split"] == part]
        summary.append({"track": name, "split": part, "segments": len(d), "attack share": d["attack"].mean(),
                        **{("unseen rows" if key == "headers" else f"unseen rows, {key}"):
                           int((~d[col].isin(train_h[key])).sum()) if part != "train" else np.nan for key, col in KEYS.items()}})
assert not tracks["B"]["hash"].isin(conflict_hashes).any()
summary = pd.DataFrame(summary).set_index(["track", "split"])
summary.to_csv(RESULT_DIR / "track_sizes.csv")
show(summary)


# %%
# Held out of every track: attack segments of Mirai (another network) and ARP MitM (same network) whose sequence never
# occurs in the known captures, plus all held-out segments for the before/after views of Mirai's time-based labels
held = segments[~known].copy()
held["group"] = np.where(held["capture"] == "Mirai", "Mirai",
                         np.where(held["arp_share"] > 0.5, "ARP MitM (ARP)", "ARP MitM (relayed)"))
held["new sequence"] = ~held["hash"].isin(in_dist["hash"])
held[["segment", "capture", "flow", "attack", "length", "start", "hash", "group", "new sequence"]].to_parquet(TRACK_DIR / "held_out.parquet", index=False)
show(held.groupby(["group", "attack"]).agg(segments=("segment", "size"), new=("new sequence", "sum")))
