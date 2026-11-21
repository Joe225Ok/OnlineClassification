"""Stage 6: MC dropout on the chosen model: aleatoric and epistemic uncertainty.

Run it from anywhere (it works in the project folder), or run every stage with run_all.py in the project folder.
TRACK (default A) and ROLE (main or no_testbed, default no_testbed) come from the environment.
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
# # Stage 6: uncertainty with MC dropout (chosen model only)
#
# The chosen model (track from `TRACK`, model from `ROLE`: `main` or `no_testbed`) is retrained with the same inputs, size,
# data and loss, plus dropout (p = 0.2) between the GRU layers and before the output. MC dropout keeps dropout on at
# inference and reads each segment T = 20 times; it is an approximation of a Bayesian neural network.
#
# **Entropy split**, per packet, in bits (binary entropy H, at most 1):
# - total = H(mean over passes of P(attack))
# - aleatoric = mean over passes of H(P(attack)) (noise the model expects in the data)
# - epistemic = total - aleatoric (mutual information: how much the passes disagree, a lack of knowledge)
#
# **Calibration** stays temperature scaling: one temperature per prefix bucket, refitted on the first validation half so
# that the *average over passes* is calibrated (capped, T >= 1). The OOD gate (forest OR Mahalanobis) and the decision
# rule are unchanged; the gate reads the hidden states of a normal pass without dropout. The High/Low cut-offs are
# retuned on the second validation half for the averaged probability.
#
# **Questions**: (1) does averaging over passes change the pipeline results or the calibration? (2) is the epistemic part
# higher on the held-out captures, and on confident wrong answers, than on correct ones? It is measured here; it does not
# enter the decision rule.


# %%
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import kitsune_lib as kl
from kitsune_lib import (BUCKETS, GRU_GRID, K_REPORT, L, SEED, STEP_BUCKET, at_prefix, fit_gru, fit_stats, group_columns,
                         known_metrics, load, load_data, load_json, p_attack, read_track, real_steps, route,
                         running_mean, split_index, tune_targets)

TRACK = os.environ.get("TRACK", "A")
ROLE = os.environ.get("ROLE", "no_testbed")
OUT = kl.RESULT_DIR
P_DROP, PASSES = 0.2, 20
HIDDEN_TRAIN = 30_000
ROLES = {"main": "main", "no_testbed": "without TTL, destination port and IAT"}

X, segments, y_all, lengths, _, _ = load_data()
track = read_track(TRACK).set_index("segment", drop=False)
part = split_index(track)
held = pd.read_parquet(kl.TRACK_DIR / "held_out.parquet").set_index("segment", drop=False)
selection = load_json(OUT / f"{TRACK}_selection.json")
arch = selection["arch"]
groups = selection["final"][ROLES[ROLE]]
cols = group_columns(groups)
print(f"Track {TRACK}, model {ROLE}, size {arch}, groups: {groups}")

# The same two validation halves as stage 3
v = track.loc[part["val"]]
strata = (v["capture"] + v["attack"].astype(str)).to_numpy()
counts = pd.Series(strata).map(pd.Series(strata).value_counts()).to_numpy()
strata = np.where(counts < 2, v["attack"].astype(str).to_numpy(), strata)
cal_idx, thr_idx = train_test_split(part["val"], test_size=0.5, random_state=SEED,
                                    stratify=strata if pd.Series(strata).value_counts().min() >= 2 else None)
parts = {"train sample": np.sort(np.random.default_rng(SEED).choice(part["train"], min(HIDDEN_TRAIN, len(part["train"])), replace=False)),
         "cal": np.sort(cal_idx), "thr": np.sort(thr_idx), "test": part["test"], "held": held.index.to_numpy()}
seg_hash = segments[kl.KEYS[kl.model_key(groups)]].to_numpy()  # "unseen" on this model's own key
unseen_of = {p: ~np.isin(seg_hash[i], seg_hash[part["train"]]) for p, i in parts.items()}
group_of = held.loc[parts["held"], "group"].to_numpy()


# %% [markdown]
# ## The model with dropout


# %%
class DropoutGRU(kl.OnlineGRU):
    """The same causal GRU with dropout between its layers and before the output."""

    def __init__(self, n_inputs, hidden, layers):
        super().__init__(n_inputs, hidden, layers)
        self.gru = torch.nn.GRU(n_inputs, hidden, num_layers=layers, batch_first=True, dropout=P_DROP if layers > 1 else 0.0)
        self.drop = torch.nn.Dropout(P_DROP)

    def forward(self, x):
        h, _ = self.gru(x)
        return self.head(self.drop(h)).squeeze(-1), h


stats = fit_stats(X, part["train"])
t0 = time.time()
net, _ = fit_gru(f"{TRACK}_{arch}_{ROLE}_mc_dropout", load(X, part["train"], stats, cols), y_all[part["train"]].astype(np.float32),
                 load(X, part["val"], stats, cols), y_all[part["val"]].astype(np.float32), *GRU_GRID[arch],
                 model_class=DropoutGRU)
print(f"trained or loaded in {time.time() - t0:.0f}s")

# Deterministic unit of this model (dropout off): OOD detectors and limits, temperatures, cut-offs
t0 = time.time()
u = kl.build_unit(X, net, cols, stats, parts, lengths, y_all)
print(f"unit built in {time.time() - t0:.0f}s")


# %% [markdown]
# ## T passes with dropout on, temperatures for the average, entropy split


# %%
@torch.no_grad()
def mc_logits(model, x, passes=PASSES, seed=SEED):
    """Logits of `passes` reads with dropout on (passes x segments x L)."""
    torch.manual_seed(seed)
    model.train()  # only dropout behaves differently in training mode
    out = np.empty((passes, len(x), x.shape[1]), np.float32)
    for i in range(0, len(x), 4096):
        xb = torch.from_numpy(x[i:i + 4096])
        for t in range(passes):
            out[t, i:i + len(xb)] = model(xb)[0].numpy()
    model.eval()
    return out


def fit_mc_temperatures(zs, lengths_, y):
    """One temperature per prefix bucket so that the mean over passes is calibrated (first validation half), T >= 1."""
    real = real_steps(lengths_)
    attack = np.repeat(y[:, None], L, axis=1)
    fitted = np.ones(len(BUCKETS))
    for b in range(len(BUCKETS)):
        m = real & (STEP_BUCKET == b)[None, :]
        if m.any():
            zb, yb = zs[:, m], attack[m]

            def loss(T):
                p = np.clip(p_attack(zb, T).mean(axis=0), 1e-7, 1 - 1e-7)
                return -np.mean(yb * np.log(p) + (1 - yb) * np.log(1 - p))
            fitted[b] = minimize_scalar(loss, bounds=(0.05, 20), method="bounded").x
    return np.maximum(fitted, 1.0)[STEP_BUCKET], fitted


def entropy(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


t0 = time.time()
zs = {p: mc_logits(net, load(X, parts[p], stats, cols)) for p in ("cal", "thr", "test", "held")}
T_mc, T_mc_fitted = fit_mc_temperatures(zs["cal"], u["n"]["cal"], u["y"]["cal"])
pbar, unc = {}, {}
for p, z in zs.items():
    probs = p_attack(z, T_mc)
    pbar[p] = probs.mean(axis=0)
    total, aleatoric = entropy(pbar[p]), entropy(probs).mean(axis=0)
    unc[p] = {"total": total, "aleatoric": aleatoric, "epistemic": np.maximum(total - aleatoric, 0)}
    del probs
del zs
print(f"{PASSES} passes in {time.time() - t0:.0f}s")
show(pd.DataFrame({"deterministic": u["fitted temperatures"][[list(STEP_BUCKET).index(b) for b in range(len(BUCKETS))]],
              "MC average": T_mc_fitted}, index=list(BUCKETS)).rename_axis("bucket (fitted, before the cap)").round(3))


# %% [markdown]
# ## Pipeline results: deterministic against the MC average
#
# Same OOD flags for both; the High/Low cut-offs of each are tuned on the second validation half. On held-out traffic any
# High or Low answer is a failure.


# %%
mc_tuned = {K: tune_targets(pbar["thr"], u["tuned"][K]["flags"]["thr"], u["n"]["thr"], u["y"]["thr"], K) for K in K_REPORT}
prob = {"deterministic": u["prob"], "MC average": pbar}
tuned = {"deterministic": {K: u["tuned"][K] for K in K_REPORT}, "MC average": mc_tuned}


def decide_with(reading, p, K):
    t = tuned[reading][K]
    return route(prob[reading][p], u["tuned"][K]["flags"][p], u["n"][p], t["tau_attack"], t["tau_benign"], K)


rows, held_rows, answers = [], [], {}
for reading in prob:
    for K in K_REPORT:
        out, packets = decide_with(reading, "test", K)
        for view, m in [("all", np.ones(len(out), bool)), ("unseen", unseen_of["test"])]:
            rows.append({"reading": reading, "K": K, "view": view, **known_metrics(out[m], u["y"]["test"][m], packets[m])})
        out_h, packets_h = decide_with(reading, "held", K)
        for g in pd.unique(group_of):
            o = out_h[group_of == g]
            held_rows.append({"reading": reading, "K": K, "traffic": g, "segments": len(o),
                              "failure rate": np.isin(o, ["High", "Low"]).mean(), **{a: (o == a).mean() for a in kl.ANSWERS}})
        if K == L:
            answers[reading] = {"test": (out, packets), "held": (out_h, packets_h)}
mc_pipeline = pd.DataFrame(rows).set_index(["reading", "K", "view"])
mc_held = pd.DataFrame(held_rows).set_index(["reading", "K", "traffic"])
mc_pipeline.to_csv(OUT / f"{TRACK}_{ROLE}_mc_pipeline.csv")
mc_held.to_csv(OUT / f"{TRACK}_{ROLE}_mc_held_out.csv")
show(pd.DataFrame({r: t[L] for r, t in tuned.items()}).loc[["tau_attack", "tau_benign", "high target met", "low target met"]])


# %%
shown = ["coverage", "attacks blocked", "block precision", "attacks called benign", "Low precision", "Medium", "OOD",
         "mean packets to block"]
show(mc_pipeline.xs(L, level="K")[shown].round(4))


# %%
show(mc_held.xs(L, level="K").round(4))


# %% [markdown]
# ## Calibration on test (P(attack) at each segment's last packet)
#
# ECE: 15 equal-width bins of P(attack), the gap between the mean P(attack) and the share of attacks in each bin,
# weighted by the bin's size.


# %%
def calibration(p, y, bins=15):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    which = np.minimum((p * bins).astype(int), bins - 1)
    ece = sum(abs(p[which == b].mean() - y[which == b].mean()) * (which == b).mean() for b in range(bins) if (which == b).any())
    return {"ECE": ece, "Brier": np.mean((p - y) ** 2), "NLL": -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))}


y_t, n_t = u["y"]["test"], u["n"]["test"]
rows = []
for reading in prob:
    p_last = at_prefix(prob[reading]["test"], n_t, L)
    for view, m in [("all", np.ones(len(y_t), bool)), ("unseen", unseen_of["test"])]:
        rows.append({"reading": reading, "view": view, **calibration(p_last[m], y_t[m])})
mc_calibration = pd.DataFrame(rows).set_index(["reading", "view"])
mc_calibration.to_csv(OUT / f"{TRACK}_{ROLE}_mc_calibration.csv")
show(mc_calibration.round(4))


# %% [markdown]
# ## Uncertainty by answer and traffic (MC average, K = 50)
#
# Each segment's total, aleatoric and epistemic entropy at the packet its answer came at.


# %%
def at_answer(values, packets):
    return values[np.arange(len(values)), packets - 1]


out_t, pk_t = answers["MC average"]["test"]
out_h, pk_h = answers["MC average"]["held"]
frame = pd.concat([
    pd.DataFrame({"traffic": np.where(y_t == 1, "known test, attack", "known test, benign"), "answer": out_t,
                  "unseen": unseen_of["test"], **{k: at_answer(v, pk_t) for k, v in unc["test"].items()}}),
    pd.DataFrame({"traffic": group_of, "answer": out_h, "unseen": unseen_of["held"],
                  **{k: at_answer(v, pk_h) for k, v in unc["held"].items()}})], ignore_index=True)
by_answer = frame.groupby(["traffic", "answer"])[["total", "aleatoric", "epistemic"]].agg(["mean", "median"])
by_answer.insert(0, "segments", frame.groupby(["traffic", "answer"]).size())
by_answer.to_csv(OUT / f"{TRACK}_{ROLE}_mc_uncertainty_by_answer.csv")
show(by_answer.round(4))


# %% [markdown]
# ## Does the epistemic part find what the model gets wrong?
#
# AUROC of each part (higher on the first group is better; 0.5 is no better than chance), at the answer packet, K = 50:
# - **wrong vs right, known test (unseen view)**: High or Low answers that are wrong, against those that are right
# - **held-out answered vs right**: High or Low answers on a held-out group (all failures), against right answers on test
# - **held-out vs known test, all segments**: the same question the OOD detectors answer; the detectors' running means at
#   the last packet are shown alongside
#
# "Caught at 5%" is the share of the first group above the 95th percentile of the right test answers: what a gate on that
# part would catch if it were allowed to turn 5% of right answers into "I don't know".


# %%
def separation(score_pos, score_neg):
    if not len(score_pos) or not len(score_neg):
        return {"AUROC": np.nan, "caught at 5%": np.nan, "segments": len(score_pos)}
    return {"AUROC": roc_auc_score(np.r_[np.ones(len(score_pos)), np.zeros(len(score_neg))], np.r_[score_pos, score_neg]),
            "caught at 5%": (score_pos > np.quantile(score_neg, 0.95)).mean(), "segments": len(score_pos)}


known = frame[frame["traffic"].str.startswith("known test")]
said = known[known["answer"].isin(["High", "Low"]) & known["unseen"]]
said_right = (said["answer"] == "High") == (said["traffic"] == "known test, attack")
right, wrong = said[said_right], said[~said_right]
rows = []
for part_name in ["total", "aleatoric", "epistemic"]:
    rows.append({"score": part_name, "comparison": "wrong vs right, known test (unseen)", **separation(wrong[part_name], right[part_name])})
    for g in pd.unique(group_of):
        h = frame[(frame["traffic"] == g) & frame["answer"].isin(["High", "Low"])]
        rows.append({"score": part_name, "comparison": f"{g}, answered vs right test", **separation(h[part_name], right[part_name])})

# Held-out against known test, all segments, at the last packet: running mean of epistemic entropy and of each detector
last = {p: {"epistemic": at_prefix(running_mean(np.where(real_steps(u["n"][p]), unc[p]["epistemic"], 0), u["n"][p]), u["n"][p], L),
            **{d: at_prefix(u["mean"][p][d], u["n"][p], L) for d in kl.DETECTORS}} for p in ("test", "held")}
for score in ["epistemic", *kl.DETECTORS]:
    for g in pd.unique(group_of):
        s = separation(last["held"][score][group_of == g], last["test"][score])
        rows.append({"score": f"{score} (running mean)", "comparison": f"{g} vs known test", **s})
mc_errors = pd.DataFrame(rows).set_index(["comparison", "score"]).sort_index()
mc_errors.to_csv(OUT / f"{TRACK}_{ROLE}_mc_error_detection.csv")
show(mc_errors.round(4))
