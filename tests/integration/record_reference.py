"""Record the reference values of the integration test for this operating system
(tests/integration/reference/integration_reference_<system>.json: windows, linux, ...).

    python tests/integration/record_reference.py          (from the repository root)

It runs every pipeline script on the crafted archive (tests/integration/edge_case_data.py), exactly as the integration
test does, and writes the values tests/integration/integration_support.collect() reads, with their tolerances, the
commit and the platform.

The reference is a regression baseline: it shows that results have not changed, not that they are right (the
hand-derived checks and the invariants of the integration test do that). Re-record only after a change to the
pipeline that was agreed, and say so in the commit. Re-record every platform's file from the same commit (Linux: on
the GitHub runner, with the manual workflow .github/workflows/record-reference.yml). Never re-record to make a failing test pass: a failing reference
check means the results changed, and that has to be explained first.
"""
import gc
import json
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent.parent / "src" / "Project")]

import edge_case_data  # noqa: E402
from integration_support import REFERENCE, collect, isolated_pipeline  # noqa: E402


def main():
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "src"], capture_output=True, text=True,
                                cwd=HERE).stdout.strip())
    # Windows keeps memory-mapped files open until they are collected: leftovers in the temporary folder are allowed
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        with isolated_pipeline(Path(tmp), edge_case_data.write_archive) as p:
            p.run_everything()
            values = collect(p)
        del p
        gc.collect()
    REFERENCE.parent.mkdir(exist_ok=True)
    REFERENCE.write_text(json.dumps({
        "recorded": time.strftime("%Y-%m-%d %H:%M:%S"),
        "commit": commit + (" (with uncommitted changes under src/)" if dirty else ""),
        "platform": f"{platform.system()} {platform.machine()}, Python {platform.python_version()}",
        "note": "Regression baseline: re-record only after an agreed change to the pipeline (see record_reference.py).",
        "values": {name: {"value": v, **tol} for name, (v, tol) in values.items()},
    }, indent=1) + "\n", encoding="utf-8")
    print(f"{len(values)} reference values written to {REFERENCE}")


if __name__ == "__main__":
    main()
