# RQ6 Real Flight

This is the manual physical-drone path for `RQ6`. Use it only in a controlled
site with a tether and an operator present.

This is the path that underlies the shipped real-flight result. The artifact
already includes the corresponding pre-baked `.ulg` files under
`results/pre-baked/rq6/real_evaluation/`, so reviewers without hardware can
validate the result through the analysis scripts.

## Preconditions

- GAP-PX4 firmware on the drone
- MAVLink `get_gyro_bias` / `set_gyro_bias` must work correctly on that build
- telemetry radio link between host and drone
- QGroundControl on the host
- the physical target coordinates set in `config.py` if you are not flying at the reference site
- the drone already airborne and stably hovering before the attack begins

QGroundControl note:

- `QGC -> Application Settings -> AutoConnect to the following devices`
- leave `UDP` enabled
- disable the other auto-connect options, then restart QGroundControl

This prevents QGC from opening the telemetry serial port directly while
`mavlink_bridge.py` is using it.

## Observation Path

The real evaluator reads only:

- `GPS_RAW_INT`
- `ATTITUDE_QUATERNION`
- `SCALED_IMU`

This matches the sim-to-real SITL evaluator.

## Assumptions

- `MAV_HERTZ = 8`: assume real GPS/MAVLink updates arrive at about `8 Hz`
- `STEP_BASE = 1.0`: inject one attack action per second
- `EP_TIMEOUT = 300.0`: stop the attack after `300 s`
- `MAV_TIMEOUT = 8 / MAV_HERTZ`: tolerate about 8 missed updates
- `TARGET_LAT`, `TARGET_LON`: target coordinates
- `TAKEOFF_ALT = 10.0`: current hover altitude and target altitude
- `TARGET_DISTANCE = 60.0`: horizontal target distance
- `SUCCESS_RADIUS_M = 10.0`: `10 m` cylinder success condition
- `SAFE_TARGET_DIST`, `SAFE_ALT_MIN`, `SAFE_ALT_MAX`: horizontal/vertical geofence

For a different site, change only the mission/site values first:

- `TARGET_LAT`, `TARGET_LON`
- `TAKEOFF_ALT`
- `TARGET_DISTANCE`
- `SAFE_TARGET_DIST`
- `SAFE_ALT_MIN`, `SAFE_ALT_MAX`

Leave `MAV_HERTZ`, `STEP_BASE`, and `EP_TIMEOUT` unchanged unless you have a
clear reason to change the real-flight timing assumptions.

## Adapting to a New Site or Target

This is possible, but it is outside the claimed artifact workflow.

- change only the mission/site values in `config.py` first
- keep the observation path and timing constants unchanged unless necessary
- rerun the full preflight notebook
- rerun the dry ground sanity check before a live flight
- do not assume the shipped success rate will carry over to the new site

## Workflow

1. Activate the repository environment.

```bash
cd /path/to/GAP
source .venv/bin/activate
```

2. Find the telemetry-radio serial port.

- Linux:

```bash
ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null
```

  Typical result: `/dev/ttyUSB0`

- macOS:

```bash
ls /dev/tty.usb* /dev/cu.usb* 2>/dev/null
```

  Typical result: `/dev/tty.usbserial-0001`

- Windows:

```powershell
[System.IO.Ports.SerialPort]::GetPortNames()
```

  Typical result: `COM3`

If unclear, unplug and replug the telemetry antenna/radio and check which port
appears.

3. Start the MAVLink bridge.

```bash
python3 src/evaluation/sim_to_real/real/mavlink_bridge.py \
    --serial /dev/ttyUSB0 --baud 57600
```

4. Run the preflight notebook and check, in order:

- shell 1: verify the bridge connection is alive
- shell 2: set zero bias several times and confirm MAVLink gyro-bias get/set
  works correctly
- shell 3: list the currently received MAVLink topics and confirm
  `GPS_RAW_INT`, `ATTITUDE_QUATERNION`, and `SCALED_IMU` are present
- shell 4: check per-topic rate; if any required topic is below `8 Hz`, inspect
  the vehicle before flight

5. Before the live flight, do one dry ground sanity check with a virtual target
and assumed hover altitude. Confirm the injected bias points in the intended
direction. Example: if the target is forward in the body frame, the injected
bias should tilt the drone forward.

6. Arm, take off, and stabilize the drone at about `10 m AGL`.

7. Run one attack episode.

```bash
python3 src/evaluation/sim_to_real/real/evaluate_unified.py \
    --checkpoint_path src/gap/models/sim-to-real_model
```

8. Copy the resulting `.ulg` into:

- `results/fresh/rq6/real_evaluation/`

Then generate the analysis with:

```bash
python3 -m analysis.generate_rq6_table6 --source fresh
python3 -m analysis.generate_rq6_figure8 --source fresh
```

## Notes

- `evaluate_unified.py` runs one evaluation episode and prints the RLlib result.
- The evaluator does not write a derived JSON. The `RQ6` analysis scripts read
  the `.ulg` files directly.
- Do not proceed if the hover is unstable before the attack starts.
- The most important real-flight precondition is a GAP-PX4 build where MAVLink
  gyro-bias get/set works correctly. If shell 2 in the preflight notebook does
  not pass reliably, do not start the live attack.
- In real flights the drone may overshoot the target. One recovery option is to
  disarm it in the air after target reach and recover it manually.
- GAP tends to keep the EKF-estimated gyro bias near zero during the attack. If
  you inject zero bias after target reach, the estimated and actual gyro bias
  become aligned again, so the drone often recovers almost immediately. In that
  case, manual control or RTL may recover the vehicle.
- Be careful with automatic RTL after attack recovery: the altitude estimate may
  still be slightly degraded.
