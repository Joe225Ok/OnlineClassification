"""Stage 6: copy each trained model of a track, with everything needed to use it, into artifacts/<track>/<model>/.

    TRACK=A python pipeline/export_artifacts.py      (run_all.py runs it after stage 3)

Each folder holds:
  model.pt     GRU weights and size
  unit.joblib  input standardisation, temperatures, OOD detectors (Isolation Forest, Mahalanobis), OOD limits and
               High/Low cut-offs per K
  config.json  what the model reads and how it decides, in plain values
These are everything pipeline/infer.py needs to classify a new capture; nothing is re-fitted on the new data.
The folder is built under a temporary name and swapped in when complete, so a failure leaves the old export in place.
"""
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import sklearn
import torch

import kitsune_lib as kl

ROLE_FOLDER = {"main": "main", "without TTL, destination port and IAT": "no_testbed"}


def export(track, out_root=ROOT / "artifacts"):
    model_dir, result_dir = kl.MODEL_DIR, kl.RESULT_DIR
    if os.environ.get("QUICK"):
        model_dir = result_dir = Path(os.environ["QUICK"])
        out_root = Path(os.environ["QUICK"]) / "artifacts"
    selection = kl.load_json(result_dir / f"{track}_selection.json")
    hidden, layers = kl.GRU_GRID[selection["arch"]]
    for role, names in selection["models"].items():
        name = names[0]
        unit_path = model_dir / f"{name}_unit.joblib"
        if not unit_path.exists():
            raise SystemExit(f"{unit_path} is missing: run stage 3 for Track {track} first")
        unit = joblib.load(unit_path)
        saved = torch.load(model_dir / f"{name}.pt")
        final = out_root / track / ROLE_FOLDER[role]
        out = final.with_name(final.name + ".tmp")  # written here, swapped in only when the three files are complete
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        torch.save({"state": saved["state"], "n_inputs": len(unit["columns"]), "hidden": hidden, "layers": layers}, out / "model.pt")
        joblib.dump(unit, out / "unit.joblib")
        plain = lambda v: {k: plain(w) for k, w in v.items()} if isinstance(v, dict) else v.tolist() if isinstance(v, np.ndarray) else v
        thresholds = {str(K): plain(t) for K, t in unit["thresholds"].items()}
        kl.save_json({
            "track": track, "model": role, "trained model name": name, "feature groups": unit["groups"],
            "input columns": [kl.COLUMNS[c] for c in unit["columns"]], "segment length L": kl.L,
            "gru": {"hidden": hidden, "layers": layers}, "seed": kl.SEED,
            "decision": {"answers": kl.ANSWERS, "precision targets": {"High": kl.TARGET_HIGH, "Low": kl.TARGET_LOW},
                         "OOD detectors (OOD if any flags)": kl.DETECTORS, "OOD budget": kl.OOD_BUDGET,
                         "prefix buckets": kl.BUCKETS, "default K": kl.L},
            "thresholds per K": thresholds,
            "versions": {"torch": torch.__version__, "scikit-learn": sklearn.__version__, "numpy": np.__version__,
                         "python": sys.version.split()[0]},
            "note": "unit.joblib holds a scikit-learn Isolation Forest; load it with the same scikit-learn version.",
            "exported": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, out / "config.json")
        if final.exists():
            shutil.rmtree(final)
        os.replace(out, final)
        print(f"{name} -> {final.relative_to(ROOT) if final.is_relative_to(ROOT) else final}")


if __name__ == "__main__":
    export(os.environ.get("TRACK", "A"))
