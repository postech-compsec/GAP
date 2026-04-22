# CI-Detector VM — One-Time Setup

The CI detector was published against ArduPilot 3.4 on Ubuntu 16.04.
The artifact ships the environment as a VMware image.

This VM is only needed if you want to rerun `RQ4` manually. It is not required
for the normal shipped-output verification path.

## Prerequisites

- **VMware Player or Workstation** (Linux/Windows)
- ~35 GB free disk; ~4 GB RAM allocated to the VM

## Download

The VM is ~32 GB so it's hosted on Zenodo separately from the code.

The path `src/evaluation/ci_detector/apmvm/` is gitignored, so
it won't pollute the repo.

## Import

In VMware: **File → Open → `src/evaluation/ci_detector/apmvm/apmvm.vmx`**.

Choose "I Copied It" if prompted about UUID.

## Recommended VMware Settings

Use the VM in the simplest possible configuration:

1. Power the VM off before changing settings.
2. Set the network adapter to **Host-only**.
3. Keep the VM on a single host-only NIC for the artifact run.
4. Allocate at least `4 GB` RAM.
5. Boot the VM and log in as:
   - user: `apm`
   - password: `apm`

## Network

Use a **host-only** VMware network for this artifact. After the VM boots:

- Find the host-only adapter IP in VMware's Virtual Network Editor, if your
  VMware edition provides it.
- On the authors' machine, `vmnet1` is host-only and the host-side IP is
  `172.16.56.1`.
- The VM-side IP will typically be another address on that subnet, e.g.
  `172.16.56.x`.
- MAVProxy inside the VM should forward to the host-side IP, with the attack
  path using host port `17000`.

Notes:

- VMware Player may not expose the Virtual Network Editor UI. That is fine as
  long as the VM is attached to a host-only NIC and the guest gets a
  `172.16.56.x` address.
- If `ip a` shows no host-only `172.*` address and the VM NIC is `DOWN`,
  enable networking from the guest's top-right network menu first.
- Do not hardcode the host IP. On many VMware host-only networks the host ends
  up at the first usable address (for example `.1`), but this is a common
  default, not a guarantee.

The host scripts in `src/evaluation/ci_detector/` already force the legacy VM
path to use **MAVLink 1**. This VMware workflow uses `SIM_GYR_BIAS_X/Y`, not
the modern custom `SET/GET_GYRO_BIAS` messages.

## One-Time Network Check

After the VM has booted:

1. In VMware's Virtual Network Editor, note the host-only host IP.
   If VMware Player does not provide that UI, check the host OS directly:
   - Linux host:
     ```bash
     ip -4 addr show vmnet1
     ```
   - Windows host:
     ```powershell
     ipconfig
     ```
     Look for `VMware Network Adapter VMnet1`.
2. Inside the VM, confirm the guest IP:
   ```bash
   ip a
   ```
   If the host-only `172.*` address is missing, enable networking in the VM
   first and run `ip a` again.
3. From the VM, confirm the host-only link is up:
   ```bash
   ping -c 3 <HOST_ONLY_IP>
   ```
   If you only know the guest subnet and need a first guess, try the first
   usable address on that subnet (for example, guest `172.16.56.129` often
   implies host `172.16.56.1`) and keep it only if the ping succeeds.

If this succeeds, use that same `<HOST_ONLY_IP>` when starting SITL with
`SIM_HOST_IP=<HOST_ONLY_IP>`.

## Credentials

- User: `apm`
- Password: `apm`

## Where the Detector Lives

Inside the VM:

- ArduPilot 3.4 source: `/home/apm/ardupilot/`
- CI-detector patch applied to `ArduCopter/AP_Motors` (already compiled)
- SITL launcher: `/home/apm/ardupilot/ArduCopter/sim_vehicle.sh`

## Verify the VM

Inside the VM, one smoke check:

```bash
cd /home/apm/ardupilot
SIM_HOST_IP=<HOST_ONLY_IP> sim_vehicle.sh -v ArduCopter --console --map
```

MAVProxy should come up with `GPS Lock`, altitude reading, and a map window.
Close it with `Ctrl-C` before running the real attack workflow.

After this one-time setup, return to `README.md` in the same directory for the
per-trial `RQ4` workflow.
