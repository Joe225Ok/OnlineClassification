"""Stage 7: drift monitoring with the population stability index (PSI).

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
# # Stage 7: drift monitoring with the population stability index (one track per run)
#
# The OOD gate judges one segment at a time. A deployment also needs to know when the traffic **as a whole** has moved,
# to trigger a review or retraining. The population stability index (PSI) compares the distribution of one variable in a
# reference window with a current window:
#
# PSI = sum over bins of (current share - reference share) x ln(current share / reference share)
#
# Bins are deciles of the reference (variables with few values use one bin per value); empty bins get a floor of 0.0001.
# Usual reading: below 0.1 stable, 0.1 to 0.25 moderate shift, above 0.25 large shift.
#
# **Variables**: every input field per real packet (raw values, before standardisation), P(attack) at each segment's last
# packet, and the mix of answers (High, Low, Medium, OOD at K = 50), from the model given by `ROLE` (default: without TTL,
# destination port and IAT).
#
# **Comparisons** (reference -> current): training -> test (the time split); Feb-May captures -> Aug captures (the same
# grouping as the stage-4 time shift); training -> each held-out capture (ARP MitM: same network; Mirai: another network).
# Each is computed on benign segments only and on all segments: an attack shifts the distribution by itself, so drift of
# normal traffic is read on the benign columns.


# %%
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import kitsune_lib as kl
from kitsune_lib import COLUMNS, GRU_GRID, L, MASK_IDX, SEED, at_prefix, fit_gru, load, load_data, load_json, p_attack, predict, read_track, route, split_index

TRACK = os.environ.get("TRACK", "A")
ROLE = os.environ.get("ROLE", "no_testbed")
OUT = kl.RESULT_DIR
QUICK = os.environ.get("QUICK")
if QUICK:  # smoke test: small samples, results kept apart
    kl.MODEL_DIR = OUT = Path(QUICK)
ROLES = {"main": "main", "no_testbed": "without TTL, destination port and IAT"}
SAMPLE = 3_000 if QUICK else 30_000  # segments per group, at most
FEB_MAY = ["Video Injection", "SSDP Flood", "SYN DoS", "SSL Renegotiation"]
AUG = ["OS Scan", "Fuzzing", "Active Wiretap"]

X, segments, y_all, lengths, _, _ = load_data()
track = read_track(TRACK).set_index("segment", drop=False)
part = split_index(track)
held = pd.read_parquet(kl.TRACK_DIR / "held_out.parquet").set_index("segment", drop=False)
selection = load_json(kl.RESULT_DIR / f"{TRACK}_selection.json")
name = selection["models"][ROLES[ROLE]][0]
unit = joblib.load(kl.MODEL_DIR / f"{name}_unit.joblib") if not QUICK else None
print(f"Track {TRACK}, model {name}")


# %% [markdown]
# ## Groups compared


# %%
rng = np.random.default_rng(SEED)


def sample(idx):
    idx = np.asarray(idx)
    return np.sort(rng.choice(idx, min(SAMPLE, len(idx)), replace=False)) if len(idx) else idx


cap = track["capture"]
groups = {
    "training": part["train"], "test": part["test"],
    "Feb-May captures": track.index[cap.isin(FEB_MAY)].to_numpy(), "Aug captures": track.index[cap.isin(AUG)].to_numpy(),
    **{f"held-out {c}": held.index[held["capture"] == c].to_numpy() for c in kl.OOD_CAPTURES},
}
groups = {g: {"all": sample(i), "benign": sample(i[y_all[i] == 0])} for g, i in groups.items()}
comparisons = [("training", "test"), ("Feb-May captures", "Aug captures")] + [("training", f"held-out {c}") for c in kl.OOD_CAPTURES]
show(pd.DataFrame({g: {k: len(v) for k, v in d.items()} for g, d in groups.items()}).T)


# %% [markdown]
# ## PSI


# %%
def psi(ref, cur, bins=10, floor=1e-4):
    ref, cur = np.asarray(ref), np.asarray(cur)
    if not len(ref) or not len(cur):
        return np.nan
    values = np.unique(ref)
    if len(values) <= 20 or ref.dtype.kind in "OUb":  # few values: one bin per value (unseen values in their own bin)
        cats = np.union1d(values, np.unique(cur))
        e = pd.Series(ref).value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
        a = pd.Series(cur).value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    else:
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1))[1:-1])
        e = np.bincount(np.searchsorted(edges, ref, side="right"), minlength=len(edges) + 1) / len(ref)
        a = np.bincount(np.searchsorted(edges, cur, side="right"), minlength=len(edges) + 1) / len(cur)
    e, a = np.maximum(e, floor), np.maximum(a, floor)
    return float(np.sum((a - e) * np.log(a / e)))


def packets(idx):
    """Raw input fields of every real packet of these segments."""
    x = np.array(X[idx])
    real = x[:, :, MASK_IDX] > 0
    return pd.DataFrame(x[real], columns=COLUMNS).drop(columns="mask")


def model_view(idx):
    """P(attack) at the last packet and the answer at K = L, from the chosen model's deployable unit."""
    if unit is None:
        return pd.DataFrame({"P(attack)": np.zeros(len(idx)), "answer": "Medium"})
    hidden, layers = GRU_GRID[selection["arch"]]
    if "net" not in model_view.__dict__:
        model_view.net, _ = fit_gru(name, np.zeros((1, 1, len(unit["columns"])), np.float32), np.zeros(1, np.float32), None, None, hidden, layers)
    z, h = predict(model_view.net, load(X, idx, unit["stats"], unit["columns"]), with_hidden=True)
    n = lengths[idx]
    p = p_attack(z, unit["temperatures"])
    t = unit["thresholds"][L]
    flags = kl.ood_flags(kl.ood_running_means(unit["detectors"], h, n), t["ood limits"])
    out, _ = route(p, flags, n, t["tau_attack"], t["tau_benign"], L)
    return pd.DataFrame({"P(attack)": at_prefix(p, n, L), "answer": out})


views = {g: {k: (packets(i), model_view(i)) for k, i in d.items()} for g, d in groups.items()}
rows = []
for ref, cur in comparisons:
    for k in ["benign", "all"]:
        (rp, rm), (cp, cm) = views[ref][k], views[cur][k]
        r = {"comparison": f"{ref} -> {cur}", "segments": k}
        r.update({c: psi(rp[c], cp[c]) for c in rp.columns})
        r.update({"P(attack)": psi(rm["P(attack)"], cm["P(attack)"]), "answer mix": psi(rm["answer"], cm["answer"])})
        rows.append(r)
table = pd.DataFrame(rows).set_index(["comparison", "segments"])
table.to_csv(OUT / f"{TRACK}_{ROLE}_psi.csv")
show(table.T.round(3))


# %% [markdown]
# Reading: below 0.1 stable, 0.1 to 0.25 moderate, above 0.25 large. Inputs not read by this model (for the model
# without the testbed group: TTL, destination port and IAT) are shown for information only.


# %%
level = table.T.apply(lambda s: pd.cut(s, [-np.inf, 0.1, 0.25, np.inf], labels=["stable", "moderate", "large"]))
show(level)


# %%
# Answer mix of each group (shares), to read the "answer mix" PSI
mix = pd.DataFrame({(g, k): v[1]["answer"].value_counts(normalize=True) for g, d in views.items() for k, v in d.items()}).T.fillna(0)
show(mix.reindex(columns=kl.ANSWERS, fill_value=0).round(4))
