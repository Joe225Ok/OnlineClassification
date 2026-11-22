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
| 6 | `s6_uncertainty.py` | MC dropout on Track A, model without the testbed-specific group: aleatoric and epistemic uncertainty |
| 7 | `s7_monitoring.py` | drift monitoring with the population stability index (PSI) |

Stages 2-5 run once per track (`TRACK=A` or `B`); stages 6-7 for Track A with `ROLE=no_testbed`. `export_artifacts.py`
runs after stage 3 for each track.

## Dataset

The project uses the Kitsune Network Attack dataset (Mirsky et al., NDSS 2018) from the UCI Machine Learning
Repository. It holds nine captures, one per attack. Each capture mixes benign and malicious traffic and comes with a
label for every packet. The captures were recorded on an IP surveillance network and an IoT network.

- Seven captures are used for training, validation and test: OS Scan, Fuzzing, Video Injection, Active Wiretap,
  SSDP Flood, SYN DoS and SSL Renegotiation.
- Two captures are held out and never used in training: ARP MitM (same network) and Mirai (another network).
- Only the raw packet captures (`*_pcap`) and the packet labels (`*_labels`) are used. The feature files that come with
  the dataset (`*_dataset`) are not used.

The dataset is not in this repository (about 19 GB zipped). It is the only file an end-to-end run needs:

1. Download `kitsune+network+attack+dataset.zip` from
   https://archive.ics.uci.edu/dataset/516/kitsune+network+attack+dataset.
2. Put the zip file, as it is, in `src/Project/data/`. Stage 0 reads it directly, so there is no need to unzip it.
3. Run the pipeline from `src/Project/` (see below).

If the download does not work, please open an issue in this repository.

## Findings

Main results on Track A (labels as recorded), test split, decision at packet 50, for the chosen model (without TTL,
destination port and inter-arrival time). The full results are in `data/results/`, and the report explains them.

| Result | Value | File |
|---|---|---|
| Test attack segments blocked | 84.8% | `A_pipeline.csv` |
| Blocks that are real attacks (target 99%) | 99.2% | `A_pipeline.csv` |
| Attacks called benign | 0.2% | `A_pipeline.csv` |
| Mean packets before a block | 2.2 | `A_pipeline.csv` |
| Segments answered (the rest: "I don't know" or OOD) | 75% | `A_pipeline.csv` |

- **Data quality.** On headers alone, only 0.7% of segments are distinct at the first packet, and 47.9% at 50 packets
  (`diagnostics.csv`).
- **Main model.** With TTL, destination port and inter-arrival time, block precision is 98.0%, below the 99% target
  (`A_pipeline.csv`).
- **New header patterns.** On test rows never seen in training, the chosen model blocks only 31% of attacks, with 92%
  block precision (`A_pipeline.csv`).
- **Another network.** The main model flags 74–83% of Mirai segments as OOD (before and during infection). ARP MitM,
  from the same network, is almost never flagged (`A_held_out.csv`).
- **Drift.** On ARP MitM, the population stability index of the answer mix is 0.46, a large shift
  (`A_no_testbed_psi.csv`).
- **Uncertainty.** On Mirai failures, epistemic uncertainty from MC dropout catches 40% at a 5% cost, against 8% for
  aleatoric uncertainty (`A_no_testbed_mc_error_detection.csv`).

## Reproducing

Every random step uses `SEED = 0`. Dependencies are listed in `requirements.txt` at the repository root (`pip install -r requirements.txt`, Python 3.13). The environment the results were produced in (library versions, thread
settings and the full package list) is in `data/results/environment_published.txt`. It is a copy of the run's own record,
with the local path removed and the machine described in general terms. Trained models and results
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

`tests/` holds unit tests of each component and one integration test of the whole pipeline. None needs the dataset;
together they take about a minute and a half.

**Unit tests** check each component on its own, on inputs built by hand, with every expected value worked out from how
the input was built:
- `test_kitsune_lib.py`: decision rule, threshold tuning, ceilings, confidence bound, PSI, track rule, time split,
  model-cache fingerprints, artifact loading.
- `test_components.py`: the packet parser on every kind of frame (TCP cut to 200 bytes, IPv6, ICMP, ARP, 802.3,
  unparseable; PCAP and PCAPNG), segmentation, `infer.py`'s helpers and `export_artifacts.py`.
- `test_tools.py`: `safeguard.py` (backup and restore on real files), `run_all.py` (step order, logs, a failed step put
  back, no answer to the overwrite question) with fake pipeline steps, and `environment.py`.

**Integration test** (`test_integration.py`): every pipeline script, unchanged, in order, in a temporary folder, on a
small fully specified archive (`edge_case_data.py`: real PCAPNG and PCAP files with the edge cases of the Kitsune data,
in its layout and label formats). Only the training budget is smaller (tiny GRUs, a few epochs), and the scripts are
kept in the temporary folder so they never write to `data/` (`integration_support.py`). It checks three things:
1. hand-derived values for the data stages, worked out from the dataset's specification: packets, flows, segments and
   their labels, the time split, the 441 conflicting segments Track B drops, the held-out groups;
2. invariants of the model stages: every cut-off meets its target where it is met, coverage is the share answered,
   the uncertainty parts add up, `infer.py` on the raw capture reproduces the answers stage 3 measured, and so on;
3. reference values: 69 model and data outputs compared, within a tolerance, with
   `reference/integration_reference_<system>.json`, one file per operating system (`windows`, `linux`). They show that
   results have not changed, not that they are right. They are written only by `python tests/record_reference.py`,
   which records the commit and platform; re-record only after an agreed change to the pipeline, never to make a
   failing test pass. Trained models give slightly different numbers on Windows and Linux (different low-level code
   in PyTorch and the math libraries), so each platform has its own file, recorded from the same commit: Windows
   locally, Linux on the GitHub runner with the manual workflow `record-reference` (Actions tab > record-reference >
   Run workflow, then commit the downloaded file). The test runs the pipeline on one thread, whatever the
   machine, because the number of threads changes the order of the sums and so the last digits of the results.

**Limitation: machine-sensitive reference values.** Every random step uses a fixed seed, and on one machine the test
gives identical results on every run. On another machine, seven reference values can differ: Mirai's OOD share before
infection and the coverage of six stage 4 sessions (`MACHINE_SENSITIVE` in `integration_support.py`). They are shares of
few segments from tiny models trained for a few epochs, so the last digits of a different CPU's arithmetic are enough
to move a borderline segment. Measured on the same code and data, they moved by up to 0.10 (5 of 50 segments) across
GitHub's Linux runners and forced CPU code paths, while every other value moved by at most 0.015. PyTorch documents
this limit ([Reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html), source for 2.9.0:
[randomness.rst](https://github.com/pytorch/pytorch/blob/v2.9.0/docs/source/notes/randomness.rst)):

> Completely reproducible results are not guaranteed across PyTorch releases, individual commits, or different
> platforms.

A tolerance wide enough for this noise would make these checks meaningless, so they keep their tolerance and are
skipped in CI instead (`SKIP_MACHINE_SENSITIVE=1` in `checks.yml`; `-rs` prints the reason). Without that variable,
for example on the machine that recorded the reference, they are compared like the others. So **a change that only
moves these seven values is caught only when the tests run on that machine**: run `python -m pytest` there before
pushing a change to the pipeline. The other 62 reference values, the hand-derived values and the invariants are checked
everywhere.

Two expected failures record known behavior of the pipeline, deliberately left unchanged:
- stage 0 counts gaps as "not whole microseconds" when timestamps near 1.5e9 s, held in float64 (about 240 ns
  resolution), are whole microseconds: the real captures store microseconds, so the counts in
  `data/results/iat_resolution.csv` (93-97%) are this artifact; the "all inputs" key, which rounds to whole
  microseconds, is unaffected;
- stage 3's packet-by-packet timing path sums OOD scores in float64 while evaluation uses float32, so a score exactly
  on an OOD limit can be decided differently; with identical first packets in a tiny archive (`synthetic_kitsune.py`)
  its consistency check stops; on the real data it never occurred (100% agreement).

`pip install -r requirements-dev.txt` (the pipeline's requirements plus pytest and pyflakes), then `python -m pytest`
from the repository root. On GitHub,
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