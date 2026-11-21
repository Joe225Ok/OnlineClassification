# Kitsune: a trustworthy, abstaining online packet-header classifier

A causal GRU reads each packet's header fields, one packet at a time, and answers **High** (attack: block), **Low**
(benign, for this segment), **Medium** ("I don't know") or **OOD** (unlike the training traffic).

This is the deployment version: every stage is a plain script under `src/Project/`. The same stages, one after the other
for both tracks, are in the notebook `Notebook/project_code.ipynb`.

## Layout

| Path | What it holds |
|---|---|
| `kitsune_lib.py` | shared code: data loading, the GRU, calibration, OOD detectors, decision rule, metrics |
| `run_all.py` | runs every stage (`pipeline/`), both tracks, one process per step |
| `safeguard.py` | what a run overwrites: the question asked before a run, the copy taken first, the restore after a failure |
| `pipeline/` | the stages (`s0_prepare.py` ... `s7_monitoring.py`), `export_artifacts.py`, `infer.py` |
| `data/` | tracks (`tracks/`), models (`models_v2/`) and result tables (`results/`) |
| `artifacts/<track>/<model>/` | each exported model with everything its decisions need |
| `runs/` | created by a run: logs of each step (`logs/`), copies taken before a run (`backup/`), outputs of `infer.py` (`inference/`) |


## Stages

| Stage | Script | Contents |
|---|---|---|
| 0 | `s0_prepare.py` | PCAP parsing, flows, 50-packet segments, inputs |
| 1 | `s1_data.py` | diagnostics before the split, Tracks A and B, time split |
| 2 | `s2_models.py` | GRU size; main model and model without TTL, destination port and IAT |
| 3 | `s3_evaluate.py` | deployable units, thresholds, all/unseen views, ceilings, baselines, per attack type, held-out traffic, drift, evasion, cost |
| 4 | `s4_experiments.py` | retraining on OOD, leave-one-capture-out and Feb-May to Aug, whole flows |
| 5 | `s5_stateful.py` | GRU trained to carry memory across a flow's segments |
| 6 | `s6_uncertainty.py` | MC dropout on Track A, model without the testbed group: aleatoric and epistemic uncertainty |
| 7 | `s7_monitoring.py` | drift monitoring with the population stability index (PSI) |

Stages 2-5 run once per track (`TRACK=A` or `B`); stages 6-7 for Track A with `ROLE=no_testbed`. `export_artifacts.py`
runs after stage 3 for each track.

## Reproducing

Every random step uses `SEED = 0`. Dependencies are listed in `requirements.txt` at the repository root (`pip install -r requirements.txt`, Python 3.13); the full environment they were produced in is in `data/results/environment.txt`. Trained models and results
are cached in `data/models_v2`, `data/results` and `data/tracks`; delete them to recompute from scratch (about 15 hours
on an 8-thread laptop CPU, most of it GRU training).

```
python run_all.py                 # every stage, both tracks
python run_all.py --only 3 6      # selected stages (6 = export the models)
python run_all.py --jobs 2        # independent steps in parallel
python run_all.py --yes           # unattended: answer yes to the overwrite question
```

A single stage also runs on its own, with the project folder as working directory: `TRACK=A python pipeline/s3_evaluate.py`.

### What a run overwrites

Trained models (`data/models_v2/*.pt`), the parsed data of stage 0 and the baseline probabilities are caches: a stage
writes them only when they are missing, so a rerun reuses them and never overwrites them. Delete a model file and the
stage retrains it (hours). Each cache records a fingerprint of what it was computed from (model, size, training settings,
`TRAINING_VERSION` in `kitsune_lib.py` and a hash of the data; increase that version whenever the training code
changes); a cache whose fingerprint does not match the current inputs stops the run with a message, so
it is never reused for other data. A cache saved before fingerprints were recorded is reused with a warning. Everything else the selected stages produce is recomputed and replaced: results tables,
tracks, deployable units and exported artifacts.

Before the first step, `run_all.py` lists the existing files the run would rewrite, states these consequences and asks
yes or no (`--yes` answers in advance for unattended runs; with no terminal to answer on, the run aborts). On yes, the
listed files are first copied to `runs/backup/<time>/` (a few tens of MB); if a step fails, that step's files are put
back from the copy and any file it created is removed, so a failure half way never costs the old results. The copy
stays until you delete it. Artifact folders are written under a temporary name and swapped in only when complete.
`python safeguard.py` prints the list without running anything.

### Tests

`tests/` holds unit tests of `kitsune_lib.py` (decision rule, threshold tuning, ceilings, confidence bound, PSI, track
rule, time split and the model-cache fingerprints) on small hand-made inputs, with every expected value worked out by
hand. They need no data files and take a few seconds: `pip install -r requirements-dev.txt` (the pipeline's
requirements plus pytest and pyflakes), then `python -m pytest` from the repository root. On GitHub,
`.github/workflows/checks.yml` runs the same lint and tests with coverage on every push, and a SonarQube analysis
(`sonar-project.properties`) once a `SONAR_TOKEN` secret is set.

## Classifying a new capture

```
python pipeline/infer.py capture.pcap --track A --model no_testbed --K 50
python pipeline/infer.py capture.pcap --labels labels.csv      # also score the answers
```

The capture is parsed and segmented as the training data was and classified with the exported model in
`artifacts/<track>/<model>/`; nothing is re-fitted. Output: one row per segment with its answer and the packet it came at.
The unit (temperatures, OOD detectors, cut-offs) records the hash of the model weights it was built for; `infer.py`
refuses an artifact folder whose model, unit and config do not match, and warns when torch, scikit-learn or numpy differ
from the versions the model was exported with.

`Artificial Intelligence (AI) Disclosure`
During the preparation of this work, the author used Claude Opus 5.5 (Anthropic) to generate initial boilerplate code. The research methodology and design were developed by the author. All AI-generated code was reviewed, revised, and validated by the author, who takes full responsibility for the final code and the content of this work.