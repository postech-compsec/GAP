#!/bin/bash
set -euo pipefail
# Bootstrap the artifact on a fresh Ubuntu host.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BLUE='\033[0;34m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${BLUE}[setup]${NC} $*"; }
ok()    { echo -e "${GREEN}[setup]${NC} $*"; }
warn()  { echo -e "${YELLOW}[setup]${NC} $*"; }
fail()  { echo -e "${RED}[setup]${NC} $*"; exit 1; }

require_file() {
    [ -f "$1" ] || fail "${2:-$1} is missing."
}
require_exec() {
    [ -x "$1" ] || fail "${2:-$1} is missing or not executable."
}

# Run a submodule setup step once; keep the marker outside the submodule.
run_once_in() {
    local submodule="$1" cmd="$2"
    local marker="$SCRIPT_DIR/.gap_${submodule//[^A-Za-z0-9]/_}_prereqs_done"
    if [ -f "$marker" ]; then
        return 0
    fi
    (cd "$submodule" && bash -c "$cmd") || fail "$submodule: $cmd failed"
    touch "$marker"
}

# Tolerate broken nested refs; the firmware build scripts fetch what they need.
init_submodules_in() {
    local parent="$1"
    (
        cd "$parent"
        [ -f .gitmodules ] || return 0
        git config -f .gitmodules --get-regexp '^submodule\..*\.path$' | \
        awk '{print $2}' | while read -r path; do
            if ! git submodule update --init --force --recursive -- "$path" 2>/dev/null; then
                warn "  $parent/$path: skipped (upstream broken ref or missing)"
            fi
        done
    )
}

info "Initializing submodules"
git submodule update --init GAP-PX4-Autopilot GAP-ardupilot GAP-mavlink
for sub in GAP-mavlink GAP-ardupilot GAP-PX4-Autopilot; do
    init_submodules_in "$sub"
done

require_file GAP-ardupilot/modules/waf/waf-light "GAP-ardupilot/modules/waf"
require_file GAP-mavlink/pymavlink/requirements.txt "GAP-mavlink/pymavlink"

# Must run before the venv because the upstream installers use --user.
info "Installing ArduPilot system prereqs"
run_once_in GAP-ardupilot "Tools/environment_install/install-prereqs-ubuntu.sh -y"
# Refresh PATH so later waf calls find the new toolchain.
[ -f "$HOME/.profile" ] && . "$HOME/.profile" || true

info "Installing PX4 system prereqs"
run_once_in GAP-PX4-Autopilot "bash Tools/setup/ubuntu.sh"
ok "System prerequisites installed"

if ! python3 -c "import ensurepip" >/dev/null 2>&1; then
    info "Installing python3-venv"
    sudo apt-get update
    sudo apt-get install -y python3-venv
fi

if [ ! -d .venv ]; then
    info "Creating Python virtualenv at .venv/"
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip wheel

# Reinstall the firmware Python deps into the venv; wxPython needs distro wheels.
WXPY_WHEEL_REPO=""
if [ -f /etc/lsb-release ]; then
    . /etc/lsb-release
    case "${DISTRIB_CODENAME:-}" in
        focal)  WXPY_WHEEL_REPO="https://extras.wxpython.org/wxPython4/extras/linux/gtk3/ubuntu-20.04" ;;
        jammy)  WXPY_WHEEL_REPO="https://extras.wxpython.org/wxPython4/extras/linux/gtk3/ubuntu-22.04" ;;
        noble)  WXPY_WHEEL_REPO="https://extras.wxpython.org/wxPython4/extras/linux/gtk3/ubuntu-24.04" ;;
    esac
fi

info "Installing Python dependencies into the venv"
pip install \
    ${WXPY_WHEEL_REPO:+-f "$WXPY_WHEEL_REPO"} \
    -r requirements.txt \
    -r GAP-PX4-Autopilot/Tools/setup/requirements.txt \
    pexpect future lxml pyserial MAVProxy geocoder empy==3.3.4 \
    ptyprocess dronecan flake8 junitparser wsproto tabulate attrdict3 \
    wxpython opencv-python

if [ -f requirements-cuda.txt ] && [ "${GAP_INSTALL_CUDA:-0}" = "1" ]; then
    info "Installing CUDA dependencies"
    pip install -r requirements-cuda.txt
fi

# Install the custom dialect shipped in GAP-mavlink.
info "Installing custom pymavlink from GAP-mavlink/"
(
    cd GAP-mavlink
    pip install -r pymavlink/requirements.txt
    if [ ! -f common.py ]; then
        warn "Pre-built common.py missing; regenerating from XML."
        python3 -m pymavlink.tools.mavgen \
            --wire-protocol=2.0 -o common message_definitions/v1.0/common.xml
    fi
    cp common.py pymavlink/dialects/v20/common.py

    # Regenerate explicitly so editable reinstalls also refresh merged dialects.
    for dialect in common ardupilotmega all; do
        xml="message_definitions/v1.0/${dialect}.xml"
        [ -f "$xml" ] || continue
        python3 -m pymavlink.tools.mavgen --wire-protocol=2.0 \
            -o "pymavlink/dialects/v20/${dialect}" "$xml" > /dev/null
        python3 -m pymavlink.tools.mavgen --wire-protocol=1.0 \
            -o "pymavlink/dialects/v10/${dialect}" "$xml" > /dev/null
    done

    pip install --editable ./pymavlink
)
python3 - <<'PY' || fail "Custom pymavlink verification failed"
import os; os.environ['MAVLINK20'] = '1'
from pymavlink.dialects.v20 import common as m
for name in ('MAVLINK_MSG_ID_SET_GYRO_BIAS', 'MAVLINK_MSG_ID_GET_GYRO_BIAS'):
    assert hasattr(m, name), f'Missing {name}'
PY
ok "Custom pymavlink installed"

info "Building ArduPilot SITL"
(
    cd GAP-ardupilot
    ./waf configure --board sitl
    ./waf copter
)
require_exec GAP-ardupilot/build/sitl/bin/arducopter "ArduPilot binary"
ok "ArduPilot SITL built"

info "Building PX4 SITL"
# Build only; do not launch a simulator during setup.
(cd GAP-PX4-Autopilot && make px4_sitl_default)
require_exec GAP-PX4-Autopilot/build/px4_sitl_default/bin/px4 "PX4 binary"
ok "PX4 SITL built"

echo ""
ok "Setup complete."
echo ""
ok "Next:"
ok "   source .venv/bin/activate"
ok "   ./experiments/run_all.sh"
ok ""
ok "Note: the prereq installers added your user to 'dialout' and"
ok "installed udev rules. These are only needed for physical USB"
ok "serial access (not SITL); reboot only if you plan to connect"
ok "real flight hardware."
