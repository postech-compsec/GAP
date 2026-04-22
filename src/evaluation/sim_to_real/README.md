# RQ6: Sim-to-Real Evaluation

`RQ6` evaluates the `sim-to-real_model` in two settings:

- `sim/`: hardware-free PX4 jMAVSim SITL evaluation
- `real/`: manual deployment on a physical drone

The analysis scripts for `RQ6` live under top-level `analysis/`.

For the shipped reference path, use the real-flight logs under
`results/pre-baked/rq6/real_evaluation/`. The SITL path is useful as a
hardware-free supplementary check, but the main `RQ6` result comes from the
real-flight logs.

For the real-drone path, the key firmware precondition is a GAP-PX4 build where
MAVLink gyro-bias get/set works correctly.

The `real/` path can also be adapted to a new site or target by changing the
mission parameters in `real/config.py` and rerunning the preflight checks. This
is an optional extension, not part of the claimed artifact workflow.

## Observation Path

The sim-to-real model uses onboard MAVLink topics only:

- `GPS_RAW_INT`
- `ATTITUDE_QUATERNION`
- `SCALED_IMU`

It does not use `HIL_STATE_QUATERNION`. The same observation path is used by
both the SITL and the real-drone evaluators.

## Main Files

```text
sim_to_real/
├── sim/
│   ├── evaluate_sim.py
│   ├── gym_env.py
│   └── px4_controller_multi_topic.py
├── real/
│   ├── evaluate_unified.py
│   ├── gym_env.py
│   ├── px4_controller_real.py
│   ├── mavlink_bridge.py
│   └── preflight_check.ipynb
└── common/
    ├── config.py
    └── observation.py
```

## Main Commands

Generate the shipped `RQ6` analysis from pre-baked data:

```bash
python3 -m analysis.generate_rq6_table6 --source pre-baked
python3 -m analysis.generate_rq6_figure8 --source pre-baked
```

Run the supplementary SITL evaluation:

```bash
./experiments/run_rq6.sh --mode approx
```

Direct SITL entrypoint:

```bash
python3 src/evaluation/sim_to_real/sim/evaluate_sim.py \
    --checkpoint_path src/gap/models/sim-to-real_model \
    --num_trials 2
```

For the physical-drone path, see:

- `sim/README.md`
- `real/README.md`

## Data Locations

- Pre-baked real-flight ULogs: `results/pre-baked/rq6/real_evaluation/`
- Fresh SITL JSONs: `results/fresh/rq6/sim_evaluation/`
- Fresh real-flight ULogs: `results/fresh/rq6/real_evaluation/`
