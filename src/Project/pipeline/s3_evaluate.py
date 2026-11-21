"""Stage 3: deployable units, thresholds, views, ceilings, baselines and the real-world checks.

Run it from anywhere (it works in the project folder), or run every stage with run_all.py in the project folder.
The track comes from the TRACK environment variable (A or B).
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
# # Stage 3: evaluation (one track per run)
#
# The track comes from `TRACK`. Each final model from stage 2 becomes one **deployable unit**, built only from this track's
# data: input statistics (training), temperatures (first validation half), two OOD detectors on training hidden states
# (an Isolation Forest, and the Mahalanobis distance to the nearer class mean with a shared covariance), OOD limits and
# the High/Low cut-offs (second validation half). Test and the held-out captures are used only to measure.
#
# **Decision rule**, per segment, packet by packet up to K (K = L = 50 is the working cut-off; shorter K are reported):
# OOD if the running mean of either detector's score is over its limit (forest OR Mahalanobis); High (block) if
# P(attack) >= tau_attack; Low (benign, this segment only) at the segment's end if P(attack) <= tau_benign; otherwise
# Medium ("I don't know").
#
# **Thresholds** are per-track precision targets on the second validation half: tau_attack is the lowest cut-off with at
# least 99% of blocks real attacks, tau_benign the highest with at least 99.9% of Low answers really benign. The two OOD
# limits sit at the same quantile of their detector's known validation scores, the lowest at which at most 5% of known
# validation segments are ever flagged by either detector.
#
# **Views**: "all" is the split as it is; "unseen" is "all" with every row whose sequence appears in this track's training
# split removed, judged on each model's own key ("all inputs" for the main model and the baselines, "headers" for the model
# without TTL, destination port and IAT). Ceilings (MaxAcc: Jerabek et al. Eq. 1; minimum FPR: their lower bound; MaxF1 and the oracle's missed attacks: our additions) are computed on
# exactly the rows each result is measured on.


# %%
import copy
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

import kitsune_lib as kl
from kitsune_lib import (BUCKETS, CAPTURES, GRU_GRID, K_REPORT, KEYS, L, SEED, STEP_BUCKET, at_prefix, at_rate, ceiling,
                         fit_stats, flow_bootstrap, group_columns, known_metrics, load, load_data, load_gru, load_json,
                         model_key, p_attack, predict, read_track, real_steps, route, rows_digest, scores, split_index,
                         tune_targets)

TRACK = os.environ.get("TRACK", "A")
OUT = kl.RESULT_DIR
HIDDEN_TRAIN, FOREST_STEPS = 30_000, 200_000
DEPLOY_RATE = 0.01  # attack share assumed for the "real network" precision line

X, segments, y_all, lengths, prefix_header, prefix_all = load_data()
track = read_track(TRACK).set_index("segment", drop=False)
part = split_index(track)
held = pd.read_parquet(kl.TRACK_DIR / "held_out.parquet").set_index("segment", drop=False)
selection = load_json(OUT / f"{TRACK}_selection.json")
arch = selection["arch"]
print(f"Track {TRACK}: size {arch}, main groups: {selection['final']['main']}")

# The two validation halves, stratified by capture and class
from sklearn.model_selection import train_test_split
v = track.loc[part["val"]]
strata = (v["capture"] + v["attack"].astype(str)).to_numpy()
counts = pd.Series(strata).map(pd.Series(strata).value_counts()).to_numpy()
strata = np.where(counts < 2, v["attack"].astype(str).to_numpy(), strata)  # a group of one: stratify by class only
cal_idx, thr_idx = train_test_split(part["val"], test_size=0.5, random_state=SEED,
                                    stratify=strata if pd.Series(strata).value_counts().min() >= 2 else None)
parts = {"train sample": np.sort(np.random.default_rng(SEED).choice(part["train"], min(HIDDEN_TRAIN, len(part["train"])), replace=False)),
         "cal": np.sort(cal_idx), "thr": np.sort(thr_idx), "val": part["val"], "test": part["test"], "held": held.index.to_numpy()}
# Unseen rows of every part under each key; each unit uses its model's key (u["unseen"])
unseen_by_key = {key: {p: ~np.isin(segments[col].to_numpy()[i], segments[col].to_numpy()[part["train"]]) for p, i in parts.items()}
                 for key, col in KEYS.items()}
for key, u_of in unseen_by_key.items():
    print(key, {p: f"{len(i):,} ({u_of[p].mean():.0%} unseen)" for p, i in parts.items()})


# %% [markdown]
# ## Deployable units
#
# `build_unit` builds everything a deployment of one model needs and saves it. The main model uses all feature
# groups; the model without TTL, destination port and IAT is built the same way.


# %%
stats = fit_stats(X, part["train"])


def unit_for(model_name, groups):
    cols = group_columns(groups)
    model = load_gru(model_name, len(cols), *GRU_GRID[arch], rows=rows_digest(TRACK, part["train"], part["val"], cols, stats, segments))
    u = kl.build_unit(X, model, cols, stats, parts, lengths, y_all, forest_steps=FOREST_STEPS)
    u.update(model=model_name, groups=groups, key=model_key(groups), unseen=unseen_by_key[model_key(groups)],
             **{"model weights": kl.weights_digest(model.state_dict())})  # the unit is valid with these weights only
    joblib.dump({k: u[k] for k in ["model", "groups", "columns", "stats", "temperatures", "fitted temperatures", "detectors",
                                   "thresholds", "model weights"]},
                kl.MODEL_DIR / f"{model_name}_unit.joblib")
    return u


units = {}
for role, names in selection["models"].items():
    for name in names:
        t0 = time.time()
        units[name] = unit_for(name, selection["final"][role])
        print(f"{name}: unit built in {time.time() - t0:.0f}s")
main = units[selection["models"]["main"][0]]
thresholds = pd.DataFrame({(name, K): {k: v for k, v in t.items() if k != "ood limits"} for name, u in units.items()
                           for K, t in u["thresholds"].items()}).T
thresholds.index.names = ["model", "K"]
thresholds.to_csv(OUT / f"{TRACK}_thresholds.csv")
show(thresholds.xs(L, level="K"))


# %% [markdown]
# ## Known traffic: pipeline results, "I know" performance, all and unseen


# %%
def decide(u, p, K):
    t = u["tuned"][K]
    return route(u["prob"][p], t["flags"][p], u["n"][p], t["tau_attack"], t["tau_benign"], K)


rows = []
for name, u in units.items():
    for p in ("val", "test"):
        for K in K_REPORT:
            out, packets = decide(u, p, K)
            for view, m in [("all", np.ones(len(out), bool)), ("unseen", u["unseen"][p])]:
                r = known_metrics(out[m], u["y"][p][m], packets[m])
                r["block precision at 1% attacks"] = at_rate(r["attacks blocked"], r["benign blocked"], DEPLOY_RATE)
                rows.append({"model": name, "split": p, "K": K, "view": view, **r})
pipeline = pd.DataFrame(rows).set_index(["model", "split", "K", "view"])
pipeline.to_csv(OUT / f"{TRACK}_pipeline.csv")
shown = ["segments", "coverage", "attacks blocked", "block precision", "block precision at 1% attacks", "attacks called benign",
         "benign blocked", "benign called Low", "Low precision", "accuracy when answering", "macro F1 when answering",
         "Medium", "OOD", "mean packets to block"]
show(pipeline.loc[(main["model"], ["val", "test"], L), shown].round(4))


# %%
# 95% intervals on test, resampling whole flows (segments of one flow are not independent)
flow_key = (track.loc[parts["test"], "capture"] + "/" + track.loc[parts["test"], "flow"].astype(str)).to_numpy()
out, _ = decide(main, "test", L)
cis = {view: flow_bootstrap(out[m], main["y"]["test"][m], flow_key[m]) for view, m in
       [("all", np.ones(len(out), bool)), ("unseen", main["unseen"]["test"])]}
pd.DataFrame(cis).to_csv(OUT / f"{TRACK}_bootstrap.csv")
show(pd.DataFrame(cis))


# %% [markdown]
# ## Per attack type, with the average across types


# %%
capture_of = track.loc[parts["test"], "capture"].to_numpy()
start_of = track.loc[parts["test"], "start"].to_numpy()
out, packets = decide(main, "test", L)
rows = []
for view, vm in [("all", np.ones(len(out), bool)), ("unseen", main["unseen"]["test"])]:
    for capture in CAPTURES:
        m = vm & (capture_of == capture)
        if not m.any():
            continue
        r = known_metrics(out[m], main["y"]["test"][m], packets[m])
        hours = max((start_of[capture_of == capture].max() - start_of[capture_of == capture].min()) / 3600, 1e-9)
        y_m, o_m = main["y"]["test"][m], out[m]
        r.update({"test hours": hours, "false blocks per hour": ((o_m == "High") & (y_m == 0)).sum() / hours,
                  "Medium per hour": (o_m == "Medium").sum() / hours,
                  "attacks passed per hour": ((o_m != "High") & (y_m == 1)).sum() / hours})
        rows.append({"view": view, "attack type": capture, **r})
per_type = pd.DataFrame(rows).set_index(["view", "attack type"])
for view in ["all", "unseen"]:  # every attack type weighs the same in the average
    per_type.loc[(view, "average across types"), :] = per_type.loc[view].mean(numeric_only=True)
per_type.to_csv(OUT / f"{TRACK}_per_attack_type.csv")
show(per_type[["attack segments", "attacks blocked", "attacks called benign", "Medium", "block precision", "Low precision",
          "false blocks per hour", "attacks passed per hour"]].round(4))


# %%
# Drift: four time slices of each capture's test period
rank = pd.Series(start_of).groupby(capture_of).rank(pct=True).to_numpy()
slice_of = np.minimum((rank * 4).astype(int), 3) + 1
drift = pd.DataFrame([{"slice": s, **known_metrics(out[slice_of == s], main["y"]["test"][slice_of == s])} for s in range(1, 5)]).set_index("slice")
drift.to_csv(OUT / f"{TRACK}_drift.csv")
show(drift[["segments", "coverage", "attacks blocked", "block precision", "Low precision", "Medium", "OOD"]].round(4))


# %% [markdown]
# ## Time to a decision
#
# How much traffic time passes before an answer: the seconds between a segment's first packet and the packet its answer
# came at (from the packets' inter-arrival times), for each decision packet K. "Blocks" are High answers on attack
# segments; "any answer" covers every segment. Compute time is separate (deployment cost below).


# %%
iat_s = np.expm1(np.array(X[parts["test"], :, kl.COL["iat"]], dtype=np.float64)) / 1e6  # seconds since the flow's previous packet
iat_s[:, 0] = 0  # counted from the segment's first packet
elapsed = np.cumsum(iat_s, axis=1)
rows = []
for name, u in units.items():
    y = u["y"]["test"]
    for K in K_REPORT:
        out, packets = decide(u, "test", K)
        t = elapsed[np.arange(len(out)), packets - 1]
        for answers, m in [("blocks", (out == "High") & (y == 1)), ("any answer", np.ones(len(out), bool))]:
            rows.append({"model": name, "K": K, "answers": answers, "segments": int(m.sum()),
                         "packets, mean": packets[m].mean() if m.any() else np.nan,
                         "seconds, median": np.median(t[m]) if m.any() else np.nan,
                         "seconds, 90th percentile": np.quantile(t[m], 0.9) if m.any() else np.nan})
decision_time = pd.DataFrame(rows).set_index(["model", "K", "answers"])
decision_time.to_csv(OUT / f"{TRACK}_decision_time.csv")
show(decision_time.round(4))


# %% [markdown]
# ## Forced answers and ceilings on the same rows
#
# A ceiling bounds a model only when identical rows are identical *in what the model reads*. Each model has its own key:
# "headers" holds exactly the inputs of the model without TTL, destination port and IAT, and "all inputs" those of the
# main model (and of the baselines, which read the same inputs), so each model's ceilings on its own key are upper bounds
# for it. Both keys are shown for every model. "above own ceiling" flags a forced accuracy above the MaxAcc of the model's
# own key on the same rows, which should not happen: the only way it can is the IAT, held in whole microseconds by the
# key (data/results/iat_resolution.csv counts the gaps that are not).


# %%
PREFIXES = [1, 2, 3, 5, 10, 20, 50]
rows = []
for name, u in units.items():
    for p in ("val", "test"):
        idx = parts[p]
        for view, m in [("all", np.ones(len(idx), bool)), ("unseen", u["unseen"][p])]:
            for k in PREFIXES:
                pred = (at_prefix(u["prob"][p], u["n"][p], k)[m] >= 0.5).astype(int)
                r = {"model": name, "split": p, "view": view, "k": k, **scores(u["y"][p][m], pred)}
                for key, H in [("all inputs", prefix_all), ("headers", prefix_header)]:
                    c = ceiling(H[idx[m], k - 1], u["y"][p][m])
                    r.update({f"{cname}, {key}": value for cname, value in c.items() if cname != "rows"})
                r["own key"] = u["key"]
                r["above own ceiling"] = bool(r["accuracy"] > r[f"MaxAcc, {u['key']}"] + 1e-12)
                r["rows"] = int(m.sum())
                rows.append(r)
forced = pd.DataFrame(rows).set_index(["model", "split", "view", "k"])
forced.to_csv(OUT / f"{TRACK}_forced_and_ceilings.csv")
if forced["above own ceiling"].any():
    print(f"WARNING: {int(forced['above own ceiling'].sum())} rows of {TRACK}_forced_and_ceilings.csv score above their own "
          f"ceiling; see iat_resolution.csv", flush=True)
no_testbed = selection["models"]["without TTL, destination port and IAT"][0]
# The model without the testbed group, against its exact ceilings (key "headers")
show(forced.loc[(no_testbed, "test"), ["rows", "accuracy", "MaxAcc, headers", "macro F1", "MaxF1, headers", "FPR",
                                  "minimum FPR, headers", "missed attacks", "minimum missed attacks, headers"]].round(4))


# %%
# The main model, against its exact ceilings (key "all inputs")
show(forced.loc[(main["model"], "test"), ["rows", "accuracy", "MaxAcc, all inputs", "macro F1", "MaxF1, all inputs", "FPR",
                                     "minimum FPR, all inputs", "missed attacks", "minimum missed attacks, all inputs"]].round(4))


# %% [markdown]
# ## Calibration on test
#
# Temperature scaling divides the logit by one temperature per range of packets seen, fitted on the first validation half
# and capped at T >= 1 so that calibration never makes the model more confident. Each model is scored with the
# temperatures it uses (capped) and, for comparison, with the fitted ones (uncapped), at decision packets 1, 3, 10 and 50.
# ECE: 15 equal-width bins of P(attack), the gap between mean P(attack) and the share of attacks in each bin, weighted by
# the bin's size. The reliability bins at K = 50 are saved for the reliability diagram.


# %%
def calibration(p, y, bins=15):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    which = np.minimum((p * bins).astype(int), bins - 1)
    ece = sum(abs(p[which == b].mean() - y[which == b].mean()) * (which == b).mean() for b in range(bins) if (which == b).any())
    return {"ECE": ece, "Brier": np.mean((p - y) ** 2), "NLL": -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))}


fitted_T = pd.DataFrame({name: dict(zip(BUCKETS, u["fitted temperatures"][[list(STEP_BUCKET).index(b) for b in range(len(BUCKETS))]]))
                         for name, u in units.items()}).T
print("fitted temperatures (before the cap), per range of packets seen:")
print(fitted_T.round(3))
rows, reliability = [], []
for name, u in units.items():
    for reading, T in [("capped (used)", u["temperatures"]), ("fitted, uncapped", u["fitted temperatures"])]:
        prob = p_attack(u["z"]["test"], T)
        for view, m in [("all", np.ones(len(u["y"]["test"]), bool)), ("unseen", u["unseen"]["test"])]:
            for k in [1, 3, 10, L]:
                p, y = at_prefix(prob, u["n"]["test"], k)[m], u["y"]["test"][m]
                rows.append({"model": name, "temperature": reading, "view": view, "k": k, **calibration(p, y)})
                if k == L and view == "all":
                    which = np.minimum((p * 15).astype(int), 14)
                    reliability += [{"model": name, "temperature": reading, "bin": b, "segments": int((which == b).sum()),
                                     "mean P(attack)": p[which == b].mean(), "attack share": y[which == b].mean()}
                                    for b in range(15) if (which == b).any()]
calib = pd.DataFrame(rows).set_index(["model", "temperature", "view", "k"])
calib.to_csv(OUT / f"{TRACK}_calibration.csv")
pd.DataFrame(reliability).to_csv(OUT / f"{TRACK}_reliability.csv", index=False)
show(calib.xs("all", level="view").round(4))


# %% [markdown]
# Does the cap change the answers? With the uncapped temperatures, the High/Low cut-offs are retuned on the second
# validation half (same precision targets, same OOD flags) and the test answers are compared with the ones in use.


# %%
rows = []
for name, u in units.items():
    prob = {p: p_attack(u["z"][p], u["fitted temperatures"]) for p in ("thr", "test")}
    for K in [1, 3, 10, L]:
        flags = u["tuned"][K]["flags"]
        t = tune_targets(prob["thr"], flags["thr"], u["n"]["thr"], u["y"]["thr"], K)
        uncapped = route(prob["test"], flags["test"], u["n"]["test"], t["tau_attack"], t["tau_benign"], K)
        for reading, (out, packets) in [("capped (used)", decide(u, "test", K)), ("fitted, uncapped", uncapped)]:
            for view, m in [("all", np.ones(len(out), bool)), ("unseen", u["unseen"]["test"])]:
                rows.append({"model": name, "temperature": reading, "K": K, "view": view,
                             **known_metrics(out[m], u["y"]["test"][m], packets[m])})
cap_answers = pd.DataFrame(rows).set_index(["model", "temperature", "K", "view"])
cap_answers.to_csv(OUT / f"{TRACK}_cap_answers.csv")
show(cap_answers.xs("all", level="view")[["coverage", "attacks blocked", "block precision", "attacks called benign", "Low precision",
                                     "Medium", "mean packets to block"]].round(4))


# %% [markdown]
# ## Held-out captures: Mirai (another network) and ARP MitM (same network)
#
# On held-out traffic the only correct answer is "I don't know" (Medium or OOD); any High or Low answer is a failure, even
# when it matches the label. Mirai's labels are time-based (every packet from 121,622 on counts as attack), so its groups
# are read as before and during infection.


# %%
group_of = held.loc[parts["held"], "group"].to_numpy()
cap_of = held.loc[parts["held"], "capture"].to_numpy()
label_of = main["y"]["held"]
rows = []
for name, u in units.items():
    out, _ = decide(u, "held", L)
    for view, vm in [("all", np.ones(len(out), bool)), ("unseen", u["unseen"]["held"])]:
        for capture in ["Mirai", "ARP MitM"]:
            for lab, desc in [(0, "before infection" if capture == "Mirai" else "benign"), (1, "during infection" if capture == "Mirai" else "attack")]:
                m = vm & (cap_of == capture) & (label_of == lab)
                o = out[m]
                rows.append({"model": name, "view": view, "traffic": f"{capture}, {desc}", "segments": int(m.sum()),
                             "failure rate": np.isin(o, ["High", "Low"]).mean(), **{a: (o == a).mean() for a in ["High", "Low", "Medium", "OOD"]}})
held_table = pd.DataFrame(rows).set_index(["model", "view", "traffic"])
held_table.to_csv(OUT / f"{TRACK}_held_out.csv")
show(held_table.loc[main["model"]].round(4))


# %%
# OOD separation: AUROC of each detector's running mean, known test packets against held-out packets, per prefix bucket
real_t, real_h = real_steps(main["n"]["test"]), real_steps(main["n"]["held"])
rows = []
for detector in kl.DETECTORS:
    s = {p: main["mean"][p][detector] for p in ("test", "held")}
    for group in pd.unique(group_of):
        for b, bucket in enumerate(BUCKETS):
            inb = (STEP_BUCKET == b)[None, :]
            known_s = s["test"][real_t & inb]
            new_s = s["held"][(group_of == group)[:, None] & real_h & inb]
            if len(new_s):
                rows.append({"detector": detector, "held-out": group, "bucket": bucket,
                             "AUROC": roc_auc_score(np.r_[np.zeros(len(known_s)), np.ones(len(new_s))], np.r_[known_s, new_s])})
ood_auc = pd.DataFrame(rows).set_index(["detector", "held-out", "bucket"])["AUROC"].unstack("bucket")[list(BUCKETS)]
ood_auc.to_csv(OUT / f"{TRACK}_ood_auroc.csv")
show(ood_auc.round(3))


# %% [markdown]
# Which detector flags what, at K = 50: the share of segments flagged by the forest only, Mahalanobis only, both, and
# either (the OOD answer). Known test traffic shows the cost, the held-out captures what each detector adds.


# %%
rows = []
for name, u in units.items():
    t = u["tuned"][L]
    for p, groups in [("test", np.where(u["y"]["test"] == 1, "known test, attack", "known test, benign")), ("held", group_of)]:
        live = real_steps(u["n"][p])
        by = {d: ((u["mean"][p][d] > np.asarray(t["ood limits"][d])[STEP_BUCKET][None, :]) & live).any(axis=1) for d in kl.DETECTORS}
        for g in pd.unique(groups):
            m = groups == g
            f, mh = by["forest"][m], by["mahalanobis"][m]
            rows.append({"model": name, "traffic": g, "segments": int(m.sum()), "forest only": (f & ~mh).mean(),
                         "Mahalanobis only": (mh & ~f).mean(), "both": (f & mh).mean(), "either": (f | mh).mean()})
detector_overlap = pd.DataFrame(rows).set_index(["model", "traffic"])
detector_overlap.to_csv(OUT / f"{TRACK}_ood_detectors.csv")
show(detector_overlap.round(4))


# %% [markdown]
# ## Baselines: k-NN and XGBoost against the GRU
#
# One section, one training per model: k-NN (L1 distance, 15 neighbours, share of attack among them) and weighted
# XGBoost, at each checkpoint k on the first k packets of the main model's inputs. Every model is read at the
# same checkpoints with the OOD gate off, gets its High/Low cut-offs from the same precision targets on the same rows of
# the second validation half, and is scored on the same 15,000 test rows. Forced answers (P(attack) >= 0.5) sit next to the
# ceilings on those rows; the H/M/L reading is the main comparison. These models read the main model's inputs, so the
# ceilings are on the "all inputs" key and are upper bounds for every one of them ("above own ceiling" flags a breach).


# %%
from sklearn.neighbors import NearestNeighbors
from xgboost import XGBClassifier

CHECKS, EVAL_ROWS, KNN_TRAIN, NEIGHBOURS = [1, 2, 3, 5, 10, 20, L], 15_000, 40_000, 15
rng = np.random.default_rng(SEED)
pick = {p: np.sort(rng.choice(len(parts[p]), min(EVAL_ROWS, len(parts[p])), replace=False)) for p in ("thr", "test")}
cols = main["columns"]
cache = OUT / f"{TRACK}_baseline_probs.npz"
# What the cached probabilities were computed from: the settings, the rows (training, evaluated), the inputs read and
# their standardisation, and the labels
baseline_fingerprint = {"training version": kl.TRAINING_VERSION, "checks": CHECKS, "evaluated rows": EVAL_ROWS, "k-NN training rows": KNN_TRAIN, "neighbours": NEIGHBOURS,
                        "xgboost": "n_estimators=200, max_depth=6, learning_rate=0.1, hist", "seed": SEED,
                        "data": kl.digest(part["train"], parts["thr"][pick["thr"]], parts["test"][pick["test"]], np.asarray(cols),
                                          stats[0], stats[1], y_all[part["train"]], segments["hash_all"].to_numpy()[part["train"]])}
if cache.exists():
    probs = dict(np.load(cache))
    stored = probs.pop("fingerprint", None)
    kl.check_fingerprint(None if stored is None else json.loads(str(stored)), baseline_fingerprint, cache)
else:
    x_tr, y_tr = load(X, part["train"], stats, cols), y_all[part["train"]]
    knn_rows = rng.permutation(len(x_tr))[:KNN_TRAIN]
    x_ev = {p: load(X, parts[p][pick[p]], stats, cols) for p in pick}
    probs = {f"{m}|{p}": np.zeros((len(pick[p]), len(CHECKS))) for m in ["k-NN", "XGBoost"] for p in pick}
    for j, k in enumerate(CHECKS):
        t0 = time.time()
        flat = x_tr[:, :k].reshape(len(x_tr), -1)
        nn = NearestNeighbors(n_neighbors=NEIGHBOURS, metric="manhattan", algorithm="brute", n_jobs=-1).fit(flat[knn_rows])
        xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, tree_method="hist", n_jobs=-1, random_state=SEED,
                            scale_pos_weight=(y_tr == 0).sum() / (y_tr == 1).sum()).fit(flat, y_tr)
        for p, x in x_ev.items():
            f = x[:, :k].reshape(len(x), -1)
            probs[f"k-NN|{p}"][:, j] = y_tr[knn_rows][nn.kneighbors(f, return_distance=False)].mean(axis=1)
            probs[f"XGBoost|{p}"][:, j] = xgb.predict_proba(f)[:, 1]
        print(f"k={k}: {time.time() - t0:.0f}s")
    np.savez(cache, fingerprint=np.array(json.dumps(baseline_fingerprint)), **probs)
for p in pick:  # the GRU read at the same checkpoints
    probs[f"GRU|{p}"] = np.stack([at_prefix(main["prob"][p][pick[p]], main["n"][p][pick[p]], k) for k in CHECKS], axis=1)

rows, forced_rows = [], []
y_ev = {p: main["y"][p][pick[p]] for p in pick}
full = {p: np.full(len(pick[p]), len(CHECKS)) for p in pick}
no_flags = {p: np.zeros((len(pick[p]), len(CHECKS)), bool) for p in pick}
test_idx = parts["test"][pick["test"]]
test_unseen = main["unseen"]["test"][pick["test"]]
for m in ["GRU", "k-NN", "XGBoost"]:
    t = tune_targets(probs[f"{m}|thr"], no_flags["thr"], full["thr"], y_ev["thr"], len(CHECKS))
    out, _ = route(probs[f"{m}|test"], no_flags["test"], full["test"], t["tau_attack"], t["tau_benign"], len(CHECKS))
    for view, vm in [("all", np.ones(len(out), bool)), ("unseen", test_unseen)]:
        rows.append({"model": m, "view": view, "tau_attack": t["tau_attack"], "tau_benign": t["tau_benign"],
                     **known_metrics(out[vm], y_ev["test"][vm])})
        for j, k in enumerate(CHECKS):
            r = {"model": m, "view": view, "k": k, **scores(y_ev["test"][vm], (probs[f"{m}|test"][vm, j] >= 0.5).astype(int))}
            c = ceiling(prefix_all[test_idx[vm], k - 1], y_ev["test"][vm])  # key "all inputs": exact for every model here
            forced_rows.append({**r, **{f"{name} (same rows)": value for name, value in c.items()},
                                "above own ceiling": bool(r["accuracy"] > c["MaxAcc"] + 1e-12)})
baselines = pd.DataFrame(rows).set_index(["model", "view"])
baselines_forced = pd.DataFrame(forced_rows).set_index(["model", "view", "k"])
baselines.to_csv(OUT / f"{TRACK}_baselines.csv")
baselines_forced.to_csv(OUT / f"{TRACK}_baselines_forced.csv")
if baselines_forced["above own ceiling"].any():
    print(f"WARNING: {int(baselines_forced['above own ceiling'].sum())} baseline rows score above their ceiling; "
          f"see iat_resolution.csv", flush=True)
show(baselines[["coverage", "attacks blocked", "block precision", "attacks called benign", "benign called Low", "Low precision",
           "accuracy when answering", "macro F1 when answering", "Medium"]].round(4))


# %%
show(baselines_forced[["accuracy", "MaxAcc (same rows)", "macro F1", "MaxF1 (same rows)", "FPR", "missed attacks"]].round(4))


# %% [markdown]
# ## Main model vs without TTL, destination port and IAT


# %%
roles = {n: role for role, names in selection["models"].items() for n in names}
compare = pd.concat({view: pipeline.xs(("test", L, view), level=["split", "K", "view"])[shown] for view in ["all", "unseen"]},
                    names=["view"]).rename(index=roles, level=1)
compare.to_csv(OUT / f"{TRACK}_testbed_comparison.csv")
show(compare.round(4))


# %% [markdown]
# ## Evasion: perturbing what an attacker controls
#
# Test attack segments with timing jitter (each inter-arrival time multiplied by a factor in [0.8, 1.2]) or padding (up to
# 32 bytes added to each frame), with no retraining. A block that turns into Medium or OOD fails safe; one that turns into
# Low is the dangerous failure.


# %%
rng = np.random.default_rng(SEED)
att = parts["test"][main["y"]["test"] == 1]
raw = np.array(X[att])
mask = raw[:, :, kl.MASK_IDX] > 0
variants = {"none": raw}
v = raw.copy(); iat = np.expm1(v[:, :, kl.COL["iat"]]) * rng.uniform(0.8, 1.2, mask.shape); v[:, :, kl.COL["iat"]] = np.where(mask, np.log1p(iat), 0); variants["IAT jitter +/-20%"] = v
v = raw.copy(); size = np.expm1(v[:, :, kl.COL["frame_len"]]) + rng.integers(0, 33, mask.shape); v[:, :, kl.COL["frame_len"]] = np.where(mask, np.log1p(size), 0); variants["padding up to 32 bytes"] = v
mu, sd = main["stats"]
n_att = lengths[att]
t = main["tuned"][L]
rows = []
for vname, raw_v in variants.items():
    x = raw_v.copy()
    x[:, :, kl.CONT_IDX] = (x[:, :, kl.CONT_IDX] - mu) / sd
    x *= x[:, :, [kl.MASK_IDX]]
    z, h = predict(main["net"], x[:, :, main["columns"]].copy(), with_hidden=True)
    flags = kl.ood_flags(kl.ood_running_means(main["detectors"], h, n_att), t["ood limits"])
    del h
    out, _ = route(p_attack(z, main["temperatures"]), flags, n_att, t["tau_attack"], t["tau_benign"], L)
    rows.append({"perturbation": vname, **{a: (out == a).mean() for a in ["High", "Low", "Medium", "OOD"]}})
evasion = pd.DataFrame(rows).set_index("perturbation")
evasion.to_csv(OUT / f"{TRACK}_evasion.csv")
print(f"{len(att):,} test attack segments; inputs used: {main['groups']}")
show(evasion.round(4))


# %% [markdown]
# ## Deployment cost
#
# Test segments of the main model are classified packet by packet, as a deployment would: the GRU step, both OOD scores
# (Isolation Forest, Mahalanobis), their running means against the OOD limits, the temperature and the High/Low cut-offs,
# stopping at the answer. Everything runs on one thread. The time per packet covers that whole path, with the share of
# each part; the answers are checked against the batch decision rule used in evaluation, so the timed code is the logic
# that was measured.


# %%
COST_SEGMENTS = 300  # test segments timed
hidden, layers = GRU_GRID[arch]
net = main["net"].eval()
t_L = main["tuned"][L]
limits = {d: np.asarray(t_L["ood limits"][d]) for d in kl.DETECTORS}
forest = copy.deepcopy(main["detectors"]["forest"]).set_params(n_jobs=1)  # one thread, like the rest
maha = main["detectors"]["mahalanobis"]
PARTS = ["GRU", "Isolation Forest", "Mahalanobis", "decision rule"]


@torch.no_grad()
def classify_stream(x, n, K=L):
    """One segment, one packet at a time. Returns its answer, the packet it came at and the seconds spent in each part."""
    spent, sums, state = dict.fromkeys(PARTS, 0.0), np.zeros(len(kl.DETECTORS)), None
    last = min(int(n), K)
    for k in range(1, last + 1):
        t0 = time.perf_counter()
        o, state = net.gru(x[None, k - 1:k], state)
        z = float(net.head(o))
        h = o[0].numpy()
        t1 = time.perf_counter()
        score = {"forest": -forest.score_samples(h)[0]}
        t2 = time.perf_counter()
        score["mahalanobis"] = min(float((((h - mu) @ maha["precision"]) * (h - mu)).sum()) for mu in maha["means"])
        t3 = time.perf_counter()
        sums += [score[d] for d in kl.DETECTORS]
        flagged = any(sums[i] / k > limits[d][STEP_BUCKET[k - 1]] for i, d in enumerate(kl.DETECTORS))
        p = 1 / (1 + np.exp(z / main["temperatures"][k - 1]))
        answer = "OOD" if flagged else "High" if p >= t_L["tau_attack"] else None
        if answer is None and k == last:
            answer = "Low" if p <= t_L["tau_benign"] else "Medium"
        t4 = time.perf_counter()
        for part_name, s in zip(PARTS, [t1 - t0, t2 - t1, t3 - t2, t4 - t3]):
            spent[part_name] += s
        if answer:
            return answer, k, spent


torch.set_num_threads(1)
timed = np.sort(np.random.default_rng(SEED).choice(len(parts["test"]), min(COST_SEGMENTS, len(parts["test"])), replace=False))
x_timed = torch.from_numpy(load(X, parts["test"][timed], stats, main["columns"]))
n_timed = main["n"]["test"][timed]
classify_stream(x_timed[0], n_timed[0])  # warm-up, not counted
runs = [classify_stream(x_timed[i], n_timed[i]) for i in range(len(timed))]
torch.set_num_threads(os.cpu_count())
batch_out, batch_packets = decide(main, "test", L)
same = np.mean([(a, k) == (batch_out[j], batch_packets[j]) for (a, k, _), j in zip(runs, timed)])
if same < 0.99:  # the timed path must be the evaluated decision rule
    raise SystemExit(f"streaming answers agree with evaluation on only {same:.1%} of the timed segments")
timed_packets = sum(k for _, k, _ in runs)
spent = {part_name: sum(s[part_name] for _, _, s in runs) for part_name in PARTS}
per_packet_us = sum(spent.values()) / timed_packets * 1e6
# What a streaming deployment keeps per open flow between packets (float32 = 4 bytes)
state_parts = {
    "GRU hidden state (bytes)": layers * hidden * 4,          # one hidden vector per GRU layer
    "OOD running sums (bytes)": len(kl.DETECTORS) * 4,        # one running sum per detector; divided by the count below
    "packet counter (bytes)": 1,                              # packets seen in the current segment, at most L = 50
}
cost = {"inference per packet, one thread (us)": per_packet_us, "packets per second, one thread": 1e6 / per_packet_us,
        **{f"{part_name}, per packet (us)": s / timed_packets * 1e6 for part_name, s in spent.items()},
        "timed segments": len(timed), "timed packets": int(timed_packets), "answers equal to evaluation": float(same),
        **state_parts, "state per flow (bytes)": sum(state_parts.values()),
        "forest trees": main["detectors"]["forest"].n_estimators}
kl.save_json(cost, OUT / f"{TRACK}_deployment_cost.json")
show(cost)
