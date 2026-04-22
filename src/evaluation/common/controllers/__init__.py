"""Controller modules for different platforms.

Avoid eager imports here. The modern PX4/current-ArduPilot controller modules
set ``MAVLINK20=1`` at import time, which breaks the legacy CI-detector path
that must stay on MAVLink 1 in the same Python process.
"""

__all__ = [
    "PX4GazeboController",
    "PX4JMAVSimController",
    "ArdupilotController",
    "ArdupilotLegacyController",
]


def __getattr__(name):
    if name == "PX4GazeboController":
        from .px4_gazebo_controller import PX4GazeboController
        return PX4GazeboController
    if name == "PX4JMAVSimController":
        from .px4_jmavsim_controller import PX4JMAVSimController
        return PX4JMAVSimController
    if name == "ArdupilotController":
        from .ardupilot_controller import ArdupilotController
        return ArdupilotController
    if name == "ArdupilotLegacyController":
        from .ardupilot_legacy_controller import ArdupilotLegacyController
        return ArdupilotLegacyController
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
