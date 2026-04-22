# Custom MAVLink Messages for GAP

GAP extends the MAVLink protocol with custom messages for sensor bias injection
and privileged state observation. These messages enable the reinforcement learning
agent to interact with the autopilot's sensor pipeline during simulation.

## Message Definitions

All custom messages are defined in the `GAP-mavlink/` submodule (branch:
`custom_message`) under `message_definitions/v1.0/common.xml`.

### Sensor Bias Messages

| Message ID | Name           | Fields                              | Purpose                               |
|------------|----------------|-------------------------------------|---------------------------------------|
| 31016      | GET_GYRO_BIAS  | gyro_bias_{x,y,z} (float, rad/s)   | Read current injected gyroscope bias  |
| 31017      | SET_GYRO_BIAS  | gyro_bias_{x,y,z} (float, rad/s)   | Inject gyroscope bias into autopilot  |
| 31018      | GET_ACCEL_BIAS | accel_bias_{x,y,z} (float, m/s^2)  | Read current accelerometer bias       |
| 31019      | SET_ACCEL_BIAS | accel_bias_{x,y,z} (float, m/s^2)  | Inject accelerometer bias             |

Each bias message carries three float fields corresponding to the X, Y, and Z
axes of the respective sensor.

### PRIVILEGED_INFO Message

The `PRIVILEGED_INFO` message provides the full internal state of
the autopilot to the RL training framework. This includes:

- Estimated and true attitude (quaternion and Euler angles)
- Angular rates and accelerations
- Position and velocity in local and global frames
- Estimator internal states (covariances, innovation sequences)
- Control outputs (actuator commands, PID terms)

This message is used exclusively during **critic training** in the asymmetric
actor-critic architecture. The actor (attack policy) never observes privileged
information; only the critic uses it to improve value estimation during training.

## How Bias Injection Works

### PX4 Firmware

1. The `SET_GYRO_BIAS` MAVLink message is received by a custom MAVLink handler.
2. The handler publishes the bias values to a uORB topic via a `GyroBias.msg`
   message definition.
3. The `PX4Gyroscope` driver subscribes to this topic and adds the bias to raw
   gyroscope readings before they enter the estimator pipeline.
4. The Extended Kalman Filter (EKF2) processes the corrupted gyroscope data,
   causing the attitude estimate to drift according to the injected bias.

### ArduPilot Firmware

1. The `SET_GYRO_BIAS` MAVLink message is received by a custom GCS_MAVLink
   handler.
2. The bias values are passed to `AP_InertialSensor` through an external bias
   interface.
3. The INS layer applies the bias to gyroscope samples before they reach the
   EKF (Extended Kalman Filter).
4. The corrupted sensor data propagates through the attitude estimator, producing
   the same drift effect as in PX4.

In both autopilots, the injection point is **before** the estimator, which means
the EKF treats the biased readings as genuine sensor data. This is what makes the
attack effective: the estimator cannot distinguish injected bias from real sensor
drift.

## Regenerating MAVLink Python Bindings

If you modify the custom message definitions, you must regenerate the Python
bindings:

```bash
cd GAP-mavlink

# Install pymavlink dependencies
python3 -m pip install -r pymavlink/requirements.txt

# Generate the Python dialect module
python3 -m pymavlink.tools.mavgen \
    --wire-protocol=2.0 \
    -o common \
    message_definitions/v1.0/common.xml

# Install the updated bindings
cp common.py pymavlink/dialects/v20/common.py
cd pymavlink
python3 -m pip install --editable .
```

### Using Custom Messages in Python

```python
import os
os.environ['MAVLINK20'] = '1'

from pymavlink import mavutil

# Connect to the autopilot
master = mavutil.mavlink_connection('udpin:0.0.0.0:14540')
master.wait_heartbeat()

# Inject a gyroscope bias
master.mav.set_gyro_bias_send(
    gyro_bias_x=0.1,   # rad/s
    gyro_bias_y=0.0,
    gyro_bias_z=0.0
)

# Read current bias
msg = master.recv_match(type="GET_GYRO_BIAS", blocking=True, timeout=5)
print(f"Current gyro bias: x={msg.gyro_bias_x}, y={msg.gyro_bias_y}, z={msg.gyro_bias_z}")
```

## References

- [MAVLink Developer Guide](https://mavlink.io/en/)
- [MAVLink Message Definition XML](https://mavlink.io/en/guide/define_xml_element.html)
- [PX4 Custom MAVLink Messages](https://docs.px4.io/main/en/mavlink/custom_messages.html)
- [pymavlink Python Library](https://mavlink.io/en/mavgen_python/)
