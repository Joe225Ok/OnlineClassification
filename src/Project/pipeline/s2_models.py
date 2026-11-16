"""Stage 2: GRU size, main model and model without TTL, destination port and IAT.

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
# # Stage 2: models (one track per run)
#
# The track comes from the `TRACK` environment variable (A or B). Everything here uses only that track's data, and
# every choice is made on its validation split, never on test.
#
# 1. **GRU size** (64, 128, 128x2) on all feature groups, by validation loss. The chosen size model is the **main model**.
# 2. **Model without TTL, destination port and IAT**: the same size on every other feature group. These inputs can
#    identify this testbed rather than the attack (TTL: the sender's operating system and hop count; destination port:
#    the attacked service; IAT: this network's link speed, load and capture host), so comparing the two models shows how
#    much of the performance depends on them.
#
# Losses are shown on all validation rows and on the unseen ones (sequence not in training).


# %%
import os

from pathlib import Path

import numpy as np
import pandas as pd

import kitsune_lib
from kitsune_lib import (GROUPS, GRU_GRID, RESULT_DIR, SEED, TESTBED_GROUPS, eval_loss, fit_gru,
                         group_columns, load, load_data, read_track, save_json, split_index, fit_stats, unseen)

TRACK = os.environ.get("TRACK", "A")
SEEDS = [SEED]  # more seeds left for future work

X, segments, y_all, lengths, prefix_header, _ = load_data()
part = split_index(read_track(TRACK))
if os.environ.get("QUICK"):  # smoke test: tiny subsample, one epoch, scratch folders
    part = {k: v[::max(1, len(v) // 2000)] for k, v in part.items()}
    kitsune_lib.MAX_EPOCHS = 1
    kitsune_lib.MODEL_DIR = RESULT_DIR = Path(os.environ["QUICK"])
stats = fit_stats(X, part["train"])
x_train_all = load(X, part["train"], stats)
x_val_all = load(X, part["val"], stats)
y_train = y_all[part["train"]].astype(np.float32)
y_val = y_all[part["val"]].astype(np.float32)
val_unseen = unseen(part["val"], part["train"], segments["hash"].to_numpy())
weight = float((y_train == 0).sum() / (y_train == 1).sum())
print(f"Track {TRACK}: train {len(y_train):,}, validation {len(y_val):,} ({val_unseen.mean():.1%} unseen)")


def train(name, groups, arch, seed=SEED):
    cols = group_columns(groups)
    model, loss_all = fit_gru(f"{TRACK}_{name}", x_train_all[:, :, cols], y_train, x_val_all[:, :, cols], y_val,
                              *GRU_GRID[arch], seed=seed)
    loss_unseen = eval_loss(model, x_val_all[val_unseen][:, :, cols], y_val[val_unseen], weight, len(cols) - 1)
    return {"model": f"{TRACK}_{name}", "validation loss, all": loss_all, "validation loss, unseen": loss_unseen}


# %% [markdown]
# ## 1. GRU size


# %%
ALL = list(GROUPS)
size_rows = {arch: train(f"size_{arch}", ALL, arch) for arch in GRU_GRID}
sizes = pd.DataFrame(size_rows).T
arch = sizes["validation loss, all"].astype(float).idxmin()
sizes.to_csv(RESULT_DIR / f"{TRACK}_gru_size.csv")
print(f"chosen size: {arch}")
show(sizes)


# %% [markdown]
# ## 2. Main model and model without TTL, destination port and IAT


# %%
final = {"main": ALL, "without TTL, destination port and IAT":[g for g in ALL if g not in TESTBED_GROUPS]}
finals = []
for role, groups in final.items():
    tag = "all" if role == "main" else "no_testbed"
    for seed in SEEDS:
        if role == "main" and seed == SEED:
            r = {**size_rows[arch]}  # the chosen size model is the main model
        else:
            r = train(f"{arch}_{tag}" if seed == SEED else f"{arch}_{tag}_seed{seed}", groups, arch, seed)
        finals.append({"role": role, "seed": seed, "groups": ", ".join(groups), **r})
finals = pd.DataFrame(finals)
finals.to_csv(RESULT_DIR / f"{TRACK}_final_models.csv", index=False)
save_json({"track": TRACK, "arch": arch, "final": final,
           "models": {role: finals.loc[finals["role"] == role, "model"].tolist() for role in final}},
          RESULT_DIR / f"{TRACK}_selection.json")
show(finals)
