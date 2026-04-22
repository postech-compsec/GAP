# RQ6 Sim Evaluation

This path evaluates the `sim-to-real_model` in PX4 jMAVSim SITL. It is a
hardware-free supplementary check for `RQ6`, not the main reference path.

## Run

Preferred wrapper:

```bash
./experiments/run_rq6.sh --mode approx
```

Direct entrypoint:

```bash
python3 src/evaluation/sim_to_real/sim/evaluate_sim.py \
    --checkpoint_path src/gap/models/sim-to-real_model \
    --num_trials 2
```

## Inputs

- `src/gap/models/sim-to-real_model/`
- bundled PX4 + jMAVSim submodules
- target coordinates from `src/evaluation/sim_to_real/common/config.py`

Override the target with:

- `SIM_TO_REAL_TARGET_LAT`
- `SIM_TO_REAL_TARGET_LON`

## Outputs

- fresh JSON results: `results/fresh/rq6/sim_evaluation/`
- Ray logs: `results/fresh/ray_results/`

## Notes

- The evaluator uses 12 parallel workers.
- `--mode approx` runs 2 trials per worker.
- `--mode full` runs 100 trials per worker.
- CPU-only execution is supported. GPU is optional.
