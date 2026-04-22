"""Serial↔UDP router for a physical drone link (SiK telemetry radio → host).

Forwards MAVLink between the serial port and three UDP clients:
    - QGroundControl          (udpout → 127.0.0.1:14550)
    - evaluate_unified.py     (udpin  ← 0.0.0.0:17000)
    - preflight_check.ipynb   (udpout → 127.0.0.1:17001)

Serial device and baud are environment-dependent; pass them as CLI args.

Usage:
    python3 mavlink_bridge.py                                     # /dev/ttyUSB0 @ 57600
    python3 mavlink_bridge.py --serial /dev/ttyUSB1 --baud 115200
    python3 mavlink_bridge.py --dry-run                           # log only; don't forward to drone
"""

import argparse
import os
import time

os.environ['MAVLINK20'] = '1'
from pymavlink import mavutil


def main():
    parser = argparse.ArgumentParser(description="Serial↔UDP MAVLink router")
    parser.add_argument("--serial", default="/dev/ttyUSB0",
                        help="Serial device (Linux: /dev/ttyUSB*; macOS: /dev/tty.usbserial-*)")
    parser.add_argument("--baud", type=int, default=57600,
                        help="Baud rate. 100 mW SiK: 57600; 500 mW SiK: 115200")
    parser.add_argument("--qgc-port", type=int, default=14550)
    parser.add_argument("--app-port", type=int, default=17000,
                        help="Port evaluate_unified.py binds (udpin)")
    parser.add_argument("--test-port", type=int, default=17001,
                        help="Port preflight_check.ipynb binds (udpin)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Log incoming MAVLink but do NOT forward app/test commands to drone")
    args = parser.parse_args()

    serial = mavutil.mavlink_connection(args.serial, baud=args.baud)
    qgc = mavutil.mavlink_connection(f"udpout:127.0.0.1:{args.qgc_port}")
    app = mavutil.mavlink_connection(f"udpin:0.0.0.0:{args.app_port}")
    test = mavutil.mavlink_connection(f"udpout:127.0.0.1:{args.test_port}")

    print(f"Waiting for heartbeat on {args.serial} @ {args.baud}...")
    serial.wait_heartbeat()
    print(f"Drone connected: sys={serial.target_system}, comp={serial.target_component}")
    print(f"  Serial:    {args.serial} @ {args.baud}")
    print(f"  QGC:       udpout → 127.0.0.1:{args.qgc_port}")
    print(f"  evaluator: udpin  ← 0.0.0.0:{args.app_port}")
    print(f"  preflight: udpout → 127.0.0.1:{args.test_port}")
    if args.dry_run:
        print("  DRY RUN: app/test MAVLink commands will NOT be forwarded to drone.")

    try:
        while True:
            msg = serial.recv_match(blocking=False)
            if msg:
                buf = msg.get_msgbuf()
                qgc.write(buf)
                app.write(buf)
                test.write(buf)

            msg = qgc.recv_match(blocking=False)
            if msg:
                serial.write(msg.get_msgbuf())

            msg = app.recv_match(blocking=False)
            if msg:
                if not args.dry_run:
                    serial.write(msg.get_msgbuf())
                print(f"evaluator → drone: {msg.get_type()}")

            msg = test.recv_match(blocking=False)
            if msg:
                if not args.dry_run:
                    serial.write(msg.get_msgbuf())
                print(f"preflight → drone: {msg.get_type()}")

            time.sleep(0.0001)
    except KeyboardInterrupt:
        print("\nBridge stopped.")


if __name__ == "__main__":
    main()
