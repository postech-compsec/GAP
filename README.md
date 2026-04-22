# GAP: Gyroscope Attack Policy

GAP is a reinforcement-learning attack framework for demonstrating an
architectural blind spot in UAV control pipelines. This repository contains
the patched PX4 and ArduPilot firmware, GAP evaluation code, analysis scripts,
and the pre-baked outputs used by the paper.
The artifact bundle is distributed through Zenodo.

Zenodo DOI: `10.5281/zenodo.19652756`

## What GAP Does

Given externally observable drone state, GAP generates an action intended to
move the drone toward a target region. The action is a gyroscope bias vector
that is added to the raw gyroscope readings before they reach the flight
controller's state estimator. The biased readings perturb the controller's
attitude estimate in a controlled way, which in turn causes the drone to drift
in the direction GAP wants. The bias is held for one second; GAP then reads
the next drone state, computes the next bias, and repeats this loop until the
drone reaches the target region or the episode times out.

## Evaluation Setup

The artifact has two evaluation geometries. Only **GAP running on
PX4-jMAVSim** uses the parallel geometry; every other configuration uses the
non-parallel one.

- **Parallel, inward (GAP on PX4-jMAVSim).** Twelve workers run in parallel
  with 4× simulation speedup. Each worker's drone spawns at one of 12
  clock-direction positions on a 220 m circle around a fixed central
  target, and attacks **inward** toward the center. Used by `RQ1` (GAP),
  `RQ2`, and `RQ6-sim`.
- **Non-parallel, outward (other simulated runs).** A single drone is
  spawned at a fixed home location, and the target is placed at one of 12
  clock-direction positions on a surrounding circle; the drone attacks
  **outward** toward that target. Simulation speedup depends on the
  simulator: PX4-jMAVSim and ArduPilot SITL run at 4×, PX4 Gazebo and the
  ci-detector run at 1×. Used by the `RQ1` baselines
  (PX4-jMAVSim), `RQ3` (PX4 Gazebo and ArduPilot SITL), and `RQ4` (ci-detector).

`RQ6-real` does not follow either geometry: the physical drone takes off,
GAP issues bias commands, and the operator directs the trial manually. It
runs in real time, one flight per trial.

In either geometry, each episode is scored against four success criteria at
once, where `Cyl.` denotes a vertical cylinder around the target and `Sph.`
denotes a sphere centered on the target:

- `Cyl. 20 m` — drone reaches within 20 m horizontal radius of the target
- `Sph. 20 m` — drone reaches within 20 m straight-line distance of the target
- `Cyl. 10 m` — drone reaches within 10 m horizontal radius (headline metric)
- `Sph. 10 m` — drone reaches within 10 m straight-line distance

The paper's headline numbers use `Cyl. 10 m`, which is what
`verify_claims.sh` also reports.

## What Each RQ Covers

- `RQ1`: GAP vs. three baselines in PX4 jMAVSim.
  Baseline 1 injects random bias every second. Baseline 2 injects a fixed
  directional bias toward the target. Baseline 3 injects an adaptive
  directional bias toward the target every second.
- `RQ2`: GAP under realistic noise conditions.
  The three conditions are tracking-noise corruption, attack delay/loss noise,
  and both together.
- `RQ3`: transfer to PX4 Gazebo and 9 ArduPilot multicopter frames.
  The ArduPilot set covers `quad`, `hexa`, `octa`, `octaquad`, `y6`,
  `dodeca-hexa`, `tri`, `singlecopter`, and `coaxcopter`.
- `RQ4`: CI-detector evasion inside a legacy VMware environment.
  This is the manual detector workflow that reproduces the published
  CI-detector implementation (Choi et al., CCS 2018).
- `RQ5`: failsafe analysis over the shared PX4 attack-flight corpus.
  This is a post-hoc analysis of built-in failsafe activation, not a fresh
  attack runner.
- `RQ6`: sim-to-real model in SITL and on a physical drone.
  The repository provides a supplementary SITL evaluation and a manual
  real-flight workflow.

`RQ4` and real-drone `RQ6` are manual workflows. See:

- `src/evaluation/ci_detector/README.md` for the RQ4 VMware workflow
- `src/evaluation/sim_to_real/README.md` for the RQ6 real-flight workflow

## Setup

The Zenodo bundle offers two equivalent entry points. Pick whichever fits your
environment.

### Path A — prepared GAP VM image (recommended)

A ready-to-run Ubuntu 22.04 VirtualBox image with the repository, Python
environment, built firmwares, model checkpoints, and QGroundControl already
in place. Import the image into VirtualBox, log in, and skip directly to
[Verify the Installation](#verify-the-installation--pre-baked-results).
Import time on a desktop-class host: about 5 minutes.

This VM image is equivalent to running Path B's `./setup.sh` on a clean
Ubuntu 22.04 install, so reviewers who prefer to rebuild from scratch can
reproduce it end-to-end via Path B.

### Path B — source snapshot

Build from scratch on a clean Ubuntu 22.04 host.

> **Side effects.** `./setup.sh` requires Internet, invokes
> `sudo apt-get install` (needs sudo), installs Python packages, and compiles
> both PX4 and ArduPilot. Expect about 30 minutes and an extra ~30 GiB of
> disk. The setup script creates a project-local `.venv/` and builds the two
> firmware submodules under `GAP-PX4-Autopilot/build/` and
> `GAP-ardupilot/build/`.

```bash
git clone --recurse-submodules https://github.com/postech-compsec/GAP.git ~/GAP
cd ~/GAP
./setup.sh
source .venv/bin/activate
```

**Download and setup GAP models**

The two RL checkpoints are distributed through Zenodo, not Git. Unpack them
into:

- `src/gap/models/gap_model/`
- `src/gap/models/sim-to-real_model/`

`gap_model/` is used by `RQ1` to `RQ5`.
`sim-to-real_model/` is used by `RQ6`.

Path A reviewers already have both checkpoints in place and can skip this
step.

## Verify the Installation & Pre-baked Results

Run this first to check that the artifact is installed correctly and that the
shipped pre-baked results are internally consistent:

```bash
./experiments/run_smoke_test.sh
bash analysis/verify_claims.sh pre-baked
bash analysis/generate_all.sh pre-baked
```

What this does:

- verifies imports, checkpoints, and representative analysis paths
- checks the paper metrics against the pre-baked data
- regenerates the shipped CSV and PNG outputs under `analysis/`

## Main Reproduction Paths

The main fresh automated reruns are `RQ1`, `RQ2`, and `RQ3`:

```bash
./experiments/run_rq1.sh --mode approx
./experiments/run_rq2.sh --mode approx
./experiments/run_rq3.sh
bash analysis/generate_all.sh fresh
bash analysis/verify_claims.sh fresh
```

Or run the same automated path through one wrapper:

```bash
./experiments/run_all.sh --mode approx
bash analysis/verify_claims.sh fresh
```

`run_all.sh` executes the fresh automated path for `RQ1`, `RQ2`, and `RQ3`,
then runs `bash analysis/generate_all.sh fresh`.

Fresh outputs are written under `results/fresh/`.

### Claim Verification Modes

`analysis/verify_claims.sh` has two modes that answer two different questions:

- **`pre-baked`** — canonical paper-reference check against the shipped
  pre-baked data. Reports **PASS** or **SKIP**. `FAIL` appears only on a
  shipped-data mismatch (should never happen in a correct distribution). Use
  this to verify the paper metrics.
- **`fresh`** — observational summary of your own reruns against the same
  paper reference. Reports **OBSERVED** or **SKIP** only.
  RL reruns of `RQ1`–`RQ3` are inherently stochastic, so the shipped
  pre-baked tree is treated as the exact paper result set and fresh reruns as
  directional corroboration.

Use pre-baked for formal PASS/SKIP. Use fresh to interpret OBSERVED values as
supplementary evidence for the same claims.

## Command Guide

Common entrypoints:

```bash
./experiments/run_smoke_test.sh
bash analysis/verify_claims.sh pre-baked
bash analysis/generate_all.sh pre-baked
bash analysis/generate_all.sh fresh
bash analysis/verify_claims.sh fresh
```

See [Claim Verification Modes](#claim-verification-modes) above for the
semantics of `pre-baked` vs `fresh`.

Per-RQ experiment wrappers:

```bash
./experiments/run_rq1.sh --mode approx
./experiments/run_rq1.sh --mode full

./experiments/run_rq2.sh --mode approx
./experiments/run_rq2.sh --mode full

./experiments/run_rq3.sh
./experiments/run_rq3.sh --frames "quad hexa octa octaquad y6 dodeca-hexa tri singlecopter coaxcopter"
./experiments/run_rq3.sh --skip-gazebo

./experiments/run_rq6.sh --mode approx
```

By default, the fresh experiment wrappers move raw flight logs into
`results/fresh/flight-logs/`. To save disk, add `--raw-logs off`.

## Analysis Guide

Use `bash analysis/generate_all.sh <pre-baked|fresh>` to regenerate all shipped
tables and figures for one result tree.
All analysis scripts aggregate all matching files under `results/<source>/`.
To run one analysis script by hand:

**RQ1 / Table 3**
```bash
python3 -m analysis.generate_rq1_table3 --source pre-baked
```

Reads `results/<source>/rq1/` and writes:
- `analysis/csv/rq1_table3_<source>.csv`
- `analysis/figures/rq1_table3_<source>.png`

**RQ1 / Figure 7**
```bash
python3 -m analysis.generate_rq1_figure7 --source pre-baked
python3 -m analysis.generate_rq1_figure7 --source pre-baked --use-ulogs
```

Default input is the shared PX4 attack-flight CSV corpus under
`results/<source>/flight-logs/px4/csv/`. Use `--use-ulogs` to read the raw
shared PX4 `.ulg` files instead if you separately have them locally. The
default shipped artifact uses the extracted CSV corpus only. Writes:

- `analysis/figures/rq1_figure7a_trajectories_<source>.png`
- `analysis/figures/rq1_figure7b_bias_<source>.png`

**RQ2 / Table 4**
```bash
python3 -m analysis.generate_rq2_table4 --source pre-baked
```

Reads `results/<source>/rq2/` and writes:
- `analysis/csv/rq2_table4_<source>.csv`
- `analysis/figures/rq2_table4_<source>.png`

**RQ3 / Table 5**
```bash
python3 -m analysis.generate_rq3_table5 --source pre-baked
```

Reads `results/<source>/rq3/` and writes:
- `analysis/csv/rq3_table5_<source>.csv`
- `analysis/figures/rq3_table5_<source>.png`

**RQ4**
```bash
python3 -m analysis.generate_rq4_analysis --source pre-baked
python3 -m analysis.generate_rq4_analysis --source pre-baked --trials 1,2,3
```

Default is the shipped trial set `1,2`. Use `--trials 1,2,3` only if
you explicitly want the extra shipped trial. Reads `results/<source>/rq4/` and
writes `rq4_analysis_<source>*.csv/.png` under `analysis/csv/` and
`analysis/figures/`.

**RQ5** (analysis-only; there is no `run_rq5.sh`)

```bash
python3 -m analysis.generate_rq5_analysis --source pre-baked
python3 -m analysis.generate_rq5_analysis --source pre-baked --use-ulogs
```

`RQ5` does not have its own experiment runner; it post-processes the same
PX4 attack-flight corpus that `RQ1` produces. Default input is the shared PX4
attack-flight CSV corpus under `results/<source>/flight-logs/px4/csv/`.
`--use-ulogs` refreshes that worker CSV corpus from
`results/<source>/flight-logs/px4/raw/` first, then reruns the analysis. The
default shipped artifact provides the extracted CSV corpus, not the full raw
PX4 log set. Writes:
- `analysis/csv/rq5_analysis_<source>.csv`

**RQ6 / Table 6**

```bash
python3 -m analysis.generate_rq6_table6 --source pre-baked
```

Reads `results/<source>/rq6/real_evaluation/` and writes:
- `analysis/csv/rq6_table6_<source>.csv`
- `analysis/figures/rq6_table6_<source>.png`

**RQ6 / Figure 8**

```bash
python3 -m analysis.generate_rq6_figure8 --source pre-baked
python3 -m analysis.generate_rq6_figure8 --source pre-baked --bias-trial 3
```

Reads `results/<source>/rq6/real_evaluation/`. The trajectory panel uses all
available real-flight trials; `--bias-trial N` chooses which trial to show in
the bias panel. Writes:

- `analysis/figures/rq6_figure8b_trajectories_<source>.png`
- `analysis/figures/rq6_figure8c_bias_<source>.png`

Most of these scripts also accept `--results-dir` to override the default
input tree. `RQ4` uses `--input-dir` instead.

## Source Layout

Top-level directories reviewers most often touch:

- `experiments/`: shell wrappers for each RQ (`run_rq1.sh`, `run_rq2.sh`,
  `run_rq3.sh`, `run_rq6.sh`), the smoke test, and `run_all.sh` for the
  one-command fresh path. These are thin drivers over `src/`.
- `analysis/`: one `generate_*.py` per paper table or figure, plus
  `verify_claims.sh` / `verify_claims.py` and `generate_all.sh`. CSV outputs go
  to `analysis/csv/`, figures to `analysis/figures/`.
- `src/gap/`: the GAP model and RLlib module definitions consumed by the
  trained checkpoints (`gap_model`, `sim-to-real_model`).
- `src/evaluation/`: reusable evaluation code shared by the experiment
  wrappers.
  - `primary/`: the main PX4-jMAVSim 12-worker evaluator used by `RQ1`
    (GAP) and `RQ2`.
  - `baseline/`: `RQ1` baselines (random / fixed-direction /
    adaptive-direction).
  - `cross_platform/`: `RQ3` drivers for PX4 Gazebo and the nine ArduPilot
    multicopter frames.
  - `ci_detector/`: `RQ4` manual VMware workflow, including
    `README.md` and `VM_SETUP.md`.
  - `sim_to_real/`: `RQ6` supplementary SITL runner and the real-flight
    workflow (see its `README.md`).
  - `common/`: shared utilities (config, logging, result writers).
- `src/tools/`: one-off helpers such as `extract_px4_attack_csvs.py`, which
  converts raw PX4 `.ulg` logs into the shared worker CSV corpus used by
  `RQ1 Figure 7` and `RQ5`.
- `GAP-PX4-Autopilot/`, `GAP-ardupilot/`: git submodules pinned to the two
  patched firmware forks. GAP's changes are preserved as separate commits on
  top of the upstream releases so reviewers can inspect them with
  `git log`.

## Shipped Data

The shipped artifact uses these pre-baked inputs:

- `results/pre-baked/rq1/`
- `results/pre-baked/rq2/`
- `results/pre-baked/rq3/`
- `results/pre-baked/rq4/`
- `results/pre-baked/rq6/`
- `results/pre-baked/flight-logs/px4/csv/worker1/` ... `worker12/`

The shared PX4 CSV corpus is used by:

- `RQ1 Figure 7`
- `RQ5`

## Results Layout

- `results/pre-baked/`: shipped paper-reference outputs consulted by
  `verify_claims.sh pre-baked`
- `results/fresh/`: outputs from your own reruns
- `results/*/rq1/`, `rq2/`, `rq3/`, `rq4/`, `rq6/`: per-RQ result trees
- `results/*/flight-logs/px4/csv/worker1/` ... `worker12/`: shared extracted
  PX4 attack-flight CSV
- `results/*/flight-logs/px4/raw/`: raw PX4 `.ulg` logs from fresh reruns or
  optional local audit data, if present
- `results/*/flight-logs/ardupilot/raw/`: raw ArduPilot `.BIN` logs from fresh
  reruns, if present
- `results/fresh/ray_results/`: fresh Ray / RLlib logs

Generated summaries and figures are written to:

- `analysis/csv/`
- `analysis/figures/`

All generated filenames are source-tagged, for example:

- `rq3_table5_pre-baked.csv`
- `rq5_analysis_fresh.csv`
- `rq1_figure7a_trajectories_fresh.png`

## Resource Summary

These are practical estimates from the authors' Ubuntu 22.04 setup:

- Pull & Setup: `~30–60 min`
- Smoke test: `<1 min`
- Pre-baked claim check: `<1 min`
- Pre-baked figure/table regeneration: `~5 min`
- Full fresh automated reruns (`RQ1` + `RQ2` + `RQ3`): `~19 h`
- Recommended CPU for the 12-worker PX4 jMAVSim path: `12 logical cores`
- RAM for setup, pre-baked verification, and analysis: `16 GiB` recommended
- RAM for the main fresh rerun path: `32 GiB` recommended
- Clean Ubuntu 22.04 VM before GAP setup: `10.9 GiB` used inside the guest
- Same VM after `./setup.sh`: `40.7 GiB` used inside the guest
- Additional guest disk consumed by GAP setup: `~30 GiB`
- Host-side VM folder size after setup: `~50 GiB`
- Recommended provisioned VM disk for evaluation: `100 GiB`
- The optional `RQ4` VMware image is a separate Zenodo download and adds
  substantial extra disk after unpacking
- GPU is optional; all shipped artifact paths run on CPU-only hosts

## Notes

- The wrappers prefer the project-local `.venv` created by `./setup.sh`.
- `--mode approx` is the recommended reviewer path for fresh reruns.
