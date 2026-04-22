# RQ4: CI-Detector Evasion

`RQ4` is a manual path that runs GAP against a legacy ArduPilot build with
the CI-detector patch applied. To faithfully reproduce the detector,
the artifact ships it as a VMware image instead of rebuilding
it from source. The VMware image (`ci-detector_apmvm.zip`) is distributed
through the project's Zenodo bundle and is
**not** included in the prepared GAP VM — download it separately and unpack. `VM_SETUP.md` in this directory walks through the one-time import.

The shipped artifact already includes the reference `RQ4` JSON outputs under
`results/pre-baked/rq4/`. Reviewers can validate the reference result from
those shipped outputs with:

```bash
python3 -m analysis.generate_rq4_analysis --source pre-baked
```

The VM path below is the optional manual rerun.

## What Is the CI-Detector?

The CI-detector in `RQ4` refers to the control-invariant detector proposed by
Hongjun Choi, Wen-Chuan Lee, Yousra Aafer, Fan Fei, Zhan Tu, Xiangyu Zhang,
Dongyan Xu, and Xinyan Deng, "Detecting Attacks against Robotic Vehicles: A
Control Invariant Approach," CCS 2018.

Original upstream repository:

- `https://github.com/hongjun9/CPS_Invariant`

In this artifact, `RQ4` preserves the released detector path in a legacy
ArduPilot 3.4 VMware environment and evaluates whether GAP can still reach the
target region before the detector reacts.

## What Runs Where

- VM: legacy ArduPilot SITL with the CI-detector patch
- Host: `ci_detector_test.py`, which sends the bias commands and records the outcome JSON

One-time VM setup is documented in `VM_SETUP.md`.
This file covers the per-trial workflow.

## Compatibility

This path is intentionally separate from the modern PX4 / ArduPilot paths in
the rest of the repository:

- legacy ArduPilot `3.4`
- MAVLink `1.0`
- gyro bias injected via `SIM_GYR_BIAS_X/Y` and `PARAM_SET`

## Per-Trial Workflow

1. Activate the repository environment on the host.

```bash
cd /path/to/GAP
source .venv/bin/activate
```

2. Start the VM SITL using the host-only network described in `VM_SETUP.md`.

```bash
SIM_HOST_IP=<HOST_ONLY_IP> sim_vehicle.sh -v ArduCopter --console --map
```

3. In that same VM terminal, initialize the simulated gyro bias and take off:

```text
param set SIM_GYR_BIAS_X 0
param set SIM_GYR_BIAS_Y 0
mode guided
arm throttle
takeoff 60
```

If `arm throttle` is rejected with a message such as
`APM: Arm: Waiting for Nav Checks`, wait until the vehicle finishes its ground
initialization checks, then arm again. Run `takeoff 60` only after the vehicle
is actually armed.

Wait until the vehicle has reached `60 m` and is stably hovering before
starting the attack trial.

4. On the host, run one `(location, trial)` pair.

```bash
python3 src/evaluation/ci_detector/ci_detector_test.py --loc 1 --trial 1
```

Typical sequence for a second trial at the same direction:

```bash
python3 src/evaluation/ci_detector/ci_detector_test.py --loc 1 --trial 2
```

Repeat the same command with `--loc 2` through `--loc 12`.

5. Stop and restart SITL before the next trial.

The full reference path is `12 locations × 2 trials = 24 runs`.

## Output

Fresh results are written under:

- `results/fresh/rq4/`

Each JSON records:

- the four success criteria
- whether the CI detector triggered
- when it triggered

After completing the full 24-run reference path, summarize it with:

```bash
python3 -m analysis.generate_rq4_analysis --source fresh
```

## Direction Index

`--loc` uses the 12 directions on the target circle:

- `1`: North
- `2`: NNE
- `3`: ENE
- `4`: East
- `5`: ESE
- `6`: SSE
- `7`: South
- `8`: SSW
- `9`: WSW
- `10`: West
- `11`: WNW
- `12`: NNW
