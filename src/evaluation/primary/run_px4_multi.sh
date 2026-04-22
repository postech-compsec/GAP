#!/bin/bash

set -e

if [ "$#" -ne 1 ]; then
    echo "Error: Instance ID must be provided as an argument."
    echo "Usage: $0 [instance_id]"
    exit 1
fi

INSTANCE_ID=$1
if [ -z "${PX4_ROOT:-}" ]; then
    echo "Error: PX4_ROOT is not set. Source experiments/common.sh first." >&2
    exit 1
fi
BUILD_PATH="$PX4_ROOT/build/px4_sitl_default"

if [ ! -d "$BUILD_PATH" ]; then
    echo "Error: PX4 build directory not found: $BUILD_PATH"
    echo "Please make sure you have run 'make px4_sitl_default'."
    exit 1
fi

WORKING_DIR="$BUILD_PATH/instance_$INSTANCE_ID"
mkdir -p "$WORKING_DIR"

export PX4_SIM_MODEL=jmavsim_iris

pushd "$WORKING_DIR" >/dev/null

echo "PX4 instance $INSTANCE_ID starting... (Working directory: $(pwd))"
exec $BUILD_PATH/bin/px4 -i $INSTANCE_ID -d "$BUILD_PATH/etc" >px4.log 2>px4.err

popd >/dev/null
