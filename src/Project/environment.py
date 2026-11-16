"""Describe the machine and software a run used, for reproducibility. run_all.py writes it to
data/results/environment.txt at the start of every run (python environment.py prints it).

    python environment.py                       # this process, as it is
    python environment.py --jobs 2 --threads 4  # also record run_all.py's settings

run_all.py runs this file as a child process with the same environment variables as the pipeline steps, so the thread
counts it reports (OMP_NUM_THREADS, MKL_NUM_THREADS, the thread pools torch, numpy and scikit-learn open) are those the
steps run with. Results are bit-identical between runs with the same --jobs on the same CPU; a different thread count
or CPU instruction set can change the last digits (floating-point sums are added in a different order).
"""
import argparse
import ctypes
import importlib.metadata as md
import os
import platform
import re
import shutil
import subprocess
import sys
import time

# Third-party packages the project imports (pyarrow: pandas reads and writes parquet with it)
PROJECT_IMPORTS = ["dpkt", "joblib", "matplotlib", "numpy", "pandas", "pyarrow", "scipy", "sklearn", "torch", "xgboost"]
THREAD_VARS = ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "PYTHONHASHSEED"]


# ----------------------------------------------------------------------------------------------------------- hardware
def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def cpu_name():
    if sys.platform == "win32":
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    elif sys.platform == "darwin":
        return run(["sysctl", "-n", "machdep.cpu.brand_string"])
    else:
        try:
            with open("/proc/cpuinfo") as f:
                m = re.search(r"model name\s*:\s*(.+)", f.read())
                if m:
                    return m.group(1).strip()
        except OSError:
            pass
    return platform.processor() or "unknown"


def physical_cores():
    try:
        import psutil  # optional
        return psutil.cpu_count(logical=False)
    except ImportError:
        pass
    if sys.platform == "win32":
        out = run(["powershell", "-NoProfile", "-Command",
                   "(Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum"])
    elif sys.platform == "darwin":
        out = run(["sysctl", "-n", "hw.physicalcpu"])
    else:
        out = run(["sh", "-c", "lscpu -p=core,socket | grep -v '^#' | sort -u | wc -l"])
    return int(out) if out.isdigit() else None


def memory_gb():
    if sys.platform == "win32":
        class Status(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        s = Status()
        s.dwLength = ctypes.sizeof(Status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)):
            return s.ullTotalPhys / 2**30, s.ullAvailPhys / 2**30
    elif sys.platform == "darwin":
        out = run(["sysctl", "-n", "hw.memsize"])
        return (int(out) / 2**30, None) if out.isdigit() else (None, None)
    else:
        try:
            with open("/proc/meminfo") as f:
                info = dict(re.findall(r"(\w+):\s+(\d+) kB", f.read()))
            return int(info["MemTotal"]) / 2**20, int(info["MemAvailable"]) / 2**20
        except (OSError, KeyError):
            pass
    return None, None


# ----------------------------------------------------------------------------------------------------------- software
def libraries():
    """Lines on the numeric libraries: torch build and threads, and every thread pool loaded (BLAS, OpenMP)."""
    lines = []
    import numpy  # noqa: F401  (loads its BLAS)
    import scipy.linalg  # noqa: F401
    import sklearn.ensemble  # noqa: F401  (loads its OpenMP runtime)
    import torch
    cfg = torch.__config__.show()
    blas = re.search(r"BLAS_INFO=(\w+)", cfg)
    lines.append(f"torch {torch.__version__}: CPU capability {torch.backends.cpu.get_cpu_capability()}, "
                 f"BLAS {blas.group(1) if blas else 'unknown'}, parallel backend "
                 f"{'OpenMP' if 'USE_OPENMP=ON' in cfg else 'native'}, CUDA available: {torch.cuda.is_available()}")
    lines.append(f"torch threads: intra-op {torch.get_num_threads()}, inter-op {torch.get_num_interop_threads()}")
    try:
        import xgboost  # noqa: F401
        from threadpoolctl import threadpool_info
        for p in threadpool_info():
            lines.append(f"thread pool: {p['internal_api']} {p.get('version') or ''} ({p['user_api']}, "
                         f"{p['num_threads']} threads, {os.path.basename(p['filepath'])}"
                         + (f", {p['architecture']}" if p.get("architecture") else "") + ")")
    except ImportError:
        lines.append("thread pools: threadpoolctl not installed")
    return lines


def packages():
    """The installed distributions behind PROJECT_IMPORTS and everything they require, as name==version."""
    from packaging.markers import Marker
    from packaging.requirements import Requirement
    by_module = md.packages_distributions()
    todo = {d for m in PROJECT_IMPORTS for d in by_module.get(m, [])}
    seen = {}
    while todo:
        name = todo.pop()
        try:
            dist = md.distribution(name)
        except md.PackageNotFoundError:
            continue
        key = dist.metadata["Name"].lower().replace("_", "-")
        if key in seen:
            continue
        seen[key] = f"{dist.metadata['Name']}=={dist.version}"
        for r in dist.requires or []:
            req = Requirement(r)
            if req.marker is None or Marker(str(req.marker)).evaluate({"extra": ""}):
                todo.add(req.name)
    return sorted(seen.values(), key=str.lower)


def tex_version():
    exe = shutil.which("pdflatex")
    return run([exe, "--version"]).splitlines()[0] if exe else "pdflatex not found"


# --------------------------------------------------------------------------------------------------------------- text
def describe(jobs=None, threads=None, command=None):
    total, free = memory_gb()
    phys = physical_cores()
    out = [f"# Environment of the run started {time.strftime('%Y-%m-%d %H:%M:%S')}, written by environment.py",
           "",
           "## Machine",
           f"OS: {platform.platform()} ({platform.machine()})",
           f"CPU: {cpu_name()}",
           f"cores: {phys if phys else 'unknown'} physical, {os.cpu_count()} logical",
           f"memory: {total:.1f} GB" + (f" ({free:.1f} GB free at the start)" if free else "") if total else "memory: unknown",
           "GPU: none used (every stage runs on the CPU)",
           "",
           "## Run settings that affect the results"]
    if jobs is not None:
        out.append(f"run_all.py: {jobs} step(s) at a time, {threads} CPU threads per step"
                   + (f"; command: {command}" if command else ""))
    out.append("random seed: SEED = 0 in kitsune_lib.py (numpy, torch, scikit-learn, XGBoost)")
    out += [f"{v}: {os.environ.get(v, 'not set')}" for v in THREAD_VARS]
    out.append("Rerun with the same --jobs on a CPU with the same instruction set for bit-identical results.")
    out += ["", "## Numeric libraries (as loaded in a pipeline step)"] + libraries()
    out += ["", "## Software",
            f"Python {platform.python_version()} ({sys.executable})",
            f"report and slides: {tex_version()}",
            "",
            "## Python packages",
            "# The packages the project imports and everything they depend on, at the versions the results were produced with."]
    out += packages()
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int)
    ap.add_argument("--threads", type=int)
    ap.add_argument("--command", help="the run_all.py command line, recorded as is")
    ap.add_argument("--out", help="write here instead of printing")
    args = ap.parse_args()
    text = describe(args.jobs, args.threads, args.command)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
