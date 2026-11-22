"""The orchestration and reproducibility tools: safeguard.py (what a run overwrites, the copy and the restore), run_all.py
(step planning and the run loop) and environment.py (the machine and package record).

Each test works in a temporary folder. The pipeline steps run_all.py starts are replaced by a fake process (the real
steps are tested in test_pipeline_scripts.py); the fake can write files and fail, so the backup-and-restore behaviour
is tested on real files. environment.py's queries of other operating systems (sysctl, /proc) are replaced by canned
answers, since only one operating system is present; the parsing of those answers is real.
"""
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

import environment
import run_all
import safeguard


# --- safeguard.py ----------------------------------------------------------------------------------------------------

def write(root, rel, text):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_rewrites_fills_in_the_track_and_adds_extra_globs():
    globs = safeguard.rewrites("s2_models", "B", extra=["x/*.csv"])
    assert "data/results/B_selection.json" in globs and globs[-1] == "x/*.csv"
    assert safeguard.rewrites("s0_prepare") == safeguard.REWRITES["s0_prepare"]


def test_describe_lists_existing_files_per_step_and_nothing_when_none_exist(tmp_path):
    assert safeguard.describe([("s2_models_A", safeguard.rewrites("s2_models", "A"))], tmp_path) == ""
    for name in ("A_gru_size.csv", "A_final_models.csv", "A_selection.json"):
        write(tmp_path, f"data/results/{name}", "old")
    for i in range(4):
        write(tmp_path, f"data/results/A_{i}_mc_pipeline.csv", "old")
    message = safeguard.describe([("s2_models_A", safeguard.rewrites("s2_models", "A")),
                                  ("s6_uncertainty_A", safeguard.rewrites("s6_uncertainty", "A"))], tmp_path)
    assert message.startswith("This run rewrites 7 existing files:")
    assert "s2_models_A" in message and "  3 files" in message
    assert ", ..." in message  # more than three files: the list is cut
    assert "What a rerun means" in message


def test_snapshot_and_restore_put_a_failed_step_back_as_it_was(tmp_path):
    globs = ["data/results/A_*.csv"]
    assert safeguard.snapshot(globs, tmp_path) is None  # nothing to protect
    kept = write(tmp_path, "data/results/A_pipeline.csv", "old pipeline")
    other = write(tmp_path, "data/results/B_pipeline.csv", "not this step's")
    backup = safeguard.snapshot(globs, tmp_path, backup_root=tmp_path / "backups")
    assert (backup / "data/results/A_pipeline.csv").read_text(encoding="utf-8") == "old pipeline"
    assert (backup / "manifest.txt").read_text(encoding="utf-8").split() == [str(Path("data/results/A_pipeline.csv"))]
    # the step overwrites its file, creates a new one, and fails
    kept.write_text("half-written", encoding="utf-8")
    new = write(tmp_path, "data/results/A_new.csv", "created by the failed step")
    restored, removed = safeguard.restore(backup, globs, tmp_path)
    assert kept.read_text(encoding="utf-8") == "old pipeline"
    assert not new.exists()
    assert restored == [Path("data/results/A_pipeline.csv")] and removed == [Path("data/results/A_new.csv")]
    assert other.read_text(encoding="utf-8") == "not this step's"  # files of other globs are left alone


@pytest.mark.parametrize("answer, expected", [("yes", True), ("Y", True), ("no", False), ("", False)])
def test_confirm_takes_yes_and_refuses_anything_else(answer, expected, capsys):
    assert safeguard.confirm("rewrite 3 files", ask=lambda _: answer) is expected
    assert "rewrite 3 files" in capsys.readouterr().out


def test_confirm_without_a_terminal_aborts_and_answers_given_in_advance_continue():
    def no_terminal(_):
        raise EOFError
    assert safeguard.confirm("message", ask=no_terminal) is False
    assert safeguard.confirm("message", assume_yes=True, ask=no_terminal) is True
    assert safeguard.confirm("", ask=no_terminal) is True  # nothing to rewrite: no question


def test_safeguard_as_a_script_only_lists_files(capsys):
    with pytest.raises(SystemExit) as done:
        runpy.run_path(safeguard.__file__, run_name="__main__")
    assert done.value.code == 0
    out = capsys.readouterr().out
    assert out.startswith("This run rewrites") or out.strip() == "nothing to rewrite"


# --- run_all.py ------------------------------------------------------------------------------------------------------

def test_tracks_and_plan_order_the_steps_and_their_dependencies():
    assert run_all.tracks_of(0, ["A", "B"]) == [None]
    assert run_all.tracks_of(3, ["A", "B"]) == ["A", "B"]
    assert run_all.tracks_of(7, ["A", "B"]) == ["A"]  # MC dropout: Track A only
    steps, wait = run_all.plan([2, 3, 6, 7], ["A", "B"])
    names = [n for n, _, _ in steps]
    assert names == ["s2_models_A", "s2_models_B", "s3_evaluate_A", "s3_evaluate_B", "export_artifacts_A",
                     "export_artifacts_B", "s6_uncertainty_A"]
    assert wait["s2_models_A"] is None  # stage 1 is not part of this run
    assert wait["s3_evaluate_B"] == "s2_models_B"
    assert wait["export_artifacts_A"] == "s3_evaluate_A"
    assert wait["s6_uncertainty_A"] == "s3_evaluate_A"
    assert run_all.command(3)[-1].endswith("s3_evaluate.py")


class FakeStep:
    """Stands in for a pipeline step's process: optionally writes files, then exits with the given code."""
    started = []
    behaviour = {}  # script name -> (exit code, {relative path: text to write})

    def __init__(self, cmd, cwd, stdout, stderr, env):
        script = Path(cmd[-1]).stem
        FakeStep.started.append((script, env.get("TRACK")))
        code, writes = FakeStep.behaviour.get(script, (0, {}))
        for rel, text in writes.items():
            write(Path(cwd), rel, text)
        stdout.write(f"{script} ran\n")
        self.returncode = code

    def poll(self):
        return self.returncode


@pytest.fixture
def fake_run(tmp_path, monkeypatch):
    """run_all.py with its project folder in tmp_path, fake steps, and environment.py's record faked as well."""
    FakeStep.started, FakeStep.behaviour = [], {}

    def record(cmd, cwd, env, capture_output, text):
        out = Path(cmd[cmd.index("--out") + 1])
        out.write_text("environment\n", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(run_all, "ROOT", tmp_path)
    monkeypatch.setattr(run_all.subprocess, "Popen", FakeStep)
    monkeypatch.setattr(run_all.subprocess, "run", record)
    monkeypatch.setattr(run_all.time, "sleep", lambda _: None)
    monkeypatch.setattr(run_all, "keep_awake", lambda: None)
    (tmp_path / "data" / "results").mkdir(parents=True)

    def main(*args):
        monkeypatch.setattr(sys, "argv", ["run_all.py", *args])
        return run_all.main()
    return main


def test_run_all_runs_the_steps_in_order_and_logs_them(fake_run, tmp_path):
    assert fake_run("--only", "2", "3", "--tracks", "A", "--yes") == 0
    assert FakeStep.started == [("s2_models", "A"), ("s3_evaluate", "A")]
    times = (tmp_path / "runs/logs/run_all_times.csv").read_text(encoding="utf-8").splitlines()
    assert times[0].startswith("step,") and len(times) == 3 and all(line.endswith(",ok") for line in times[1:])
    assert (tmp_path / "runs/logs/s3_evaluate_A.log").read_text(encoding="utf-8") == "s3_evaluate ran\n"
    assert (tmp_path / "data/results/environment.txt").exists()
    assert len(list((tmp_path / "runs/logs").glob("environment_*.txt"))) == 1  # the dated copy


def test_run_all_puts_back_the_files_of_a_failed_step_and_starts_nothing_after(fake_run, tmp_path):
    old = write(tmp_path, "data/results/A_thresholds.csv", "old thresholds")
    FakeStep.behaviour = {"s3_evaluate": (1, {"data/results/A_thresholds.csv": "half-written",
                                             "data/results/A_pipeline.csv": "new file"})}
    assert fake_run("--only", "3", "6", "--tracks", "A", "--yes") == 1
    assert old.read_text(encoding="utf-8") == "old thresholds"
    assert not (tmp_path / "data/results/A_pipeline.csv").exists()
    assert FakeStep.started == [("s3_evaluate", "A")]  # the export waited for stage 3 and never started
    assert list((tmp_path / "runs/backup").iterdir())  # the copy is kept


def test_run_all_stops_when_the_overwrite_question_gets_no_answer(fake_run, tmp_path):
    # pytest gives the run no terminal to answer on (reading stdin fails), as an unattended run without --yes
    write(tmp_path, "data/results/A_selection.json", "{}")
    assert fake_run("--only", "2", "--tracks", "A") == 2
    assert FakeStep.started == []


def test_run_all_keeps_an_old_style_log_and_starts_a_new_one(fake_run, tmp_path):
    write(tmp_path, "runs/logs/run_all_times.csv", "step,quick,seconds\n")
    assert fake_run("--only", "2", "--tracks", "B", "--yes") == 0
    assert len(list((tmp_path / "runs/logs").glob("run_all_times_before_*.csv"))) == 1


def test_record_environment_only_warns_when_it_fails(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run_all.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a[0], 1, "", "Traceback\nboom"))
    run_all.record_environment(tmp_path, tmp_path, {}, 1, "8")
    assert "warning" in capsys.readouterr().out and not (tmp_path / "environment.txt").exists()


# --- environment.py --------------------------------------------------------------------------------------------------

def test_describe_records_machine_settings_libraries_and_packages(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    text = environment.describe(jobs=2, threads=4, command="python run_all.py --jobs 2")
    for heading in ("## Machine", "## Run settings that affect the results", "## Numeric libraries", "## Software",
                    "## Python packages"):
        assert heading in text
    assert "run_all.py: 2 step(s) at a time, 4 CPU threads per step; command: python run_all.py --jobs 2" in text
    assert "OMP_NUM_THREADS: 4" in text
    assert any(line.lower().startswith("numpy==") for line in text.splitlines())
    assert any(line.lower().startswith("torch==") for line in text.splitlines())


def test_environment_main_writes_the_record(tmp_path, monkeypatch, capsys):
    out = tmp_path / "environment.txt"
    monkeypatch.setattr(sys, "argv", ["environment.py", "--jobs", "1", "--threads", "8", "--out", str(out)])
    environment.main()
    assert out.read_text(encoding="utf-8").startswith("# Environment of the run")
    monkeypatch.setattr(sys, "argv", ["environment.py"])
    environment.main()
    assert "## Machine" in capsys.readouterr().out


def test_run_returns_nothing_for_a_missing_command():
    assert environment.run(["no-such-command-for-this-test"]) == ""


def test_this_machine_is_described():
    assert environment.cpu_name()
    total, free = environment.memory_gb()
    assert total is None or total > 0
    cores = environment.physical_cores()
    assert cores is None or cores >= 1


def test_linux_queries_are_parsed(monkeypatch):
    files = {"/proc/cpuinfo": "processor\t: 0\nmodel name\t: Example CPU 3000\n",
             "/proc/meminfo": "MemTotal:       16384000 kB\nMemAvailable:    8192000 kB\n"}

    class Text:
        def __init__(self, path):
            self.text = files[path]

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return self.text

    monkeypatch.setattr(environment.sys, "platform", "linux")
    monkeypatch.setattr(environment, "open", Text, raising=False)
    monkeypatch.setitem(sys.modules, "psutil", None)  # not installed: the fallback query is used
    monkeypatch.setattr(environment, "run", lambda cmd: "6")
    assert environment.cpu_name() == "Example CPU 3000"
    assert environment.physical_cores() == 6
    total, free = environment.memory_gb()
    assert total == pytest.approx(16384000 / 2**20) and free == pytest.approx(8192000 / 2**20)


def test_macos_queries_are_parsed(monkeypatch):
    answers = {"machdep.cpu.brand_string": "Example M9", "hw.physicalcpu": "10", "hw.memsize": str(32 * 2**30)}
    monkeypatch.setattr(environment.sys, "platform", "darwin")
    monkeypatch.setitem(sys.modules, "psutil", None)
    monkeypatch.setattr(environment, "run", lambda cmd: answers[cmd[-1]])
    assert environment.cpu_name() == "Example M9"
    assert environment.physical_cores() == 10
    assert environment.memory_gb() == (32.0, None)


def test_unreadable_answers_are_reported_as_unknown(monkeypatch):
    monkeypatch.setattr(environment.sys, "platform", "darwin")
    monkeypatch.setitem(sys.modules, "psutil", None)
    monkeypatch.setattr(environment, "run", lambda cmd: "")
    assert environment.physical_cores() is None
    assert environment.memory_gb() == (None, None)
