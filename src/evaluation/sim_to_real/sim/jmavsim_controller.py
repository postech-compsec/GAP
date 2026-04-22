import os, signal, subprocess, time, socket
from config import JMAVSIM_SCRIPT, logger

class JmavsimController:
    def __init__(self, instance_id: int, speed_factor="1", headless="0"):
        self.instance_id = instance_id
        self.port = 4560 + instance_id  # Default port is 4560, each instance gets a unique port
        self.proc = None  # Will hold the process handle once started
        self.speed_factor = speed_factor
        self.headless = headless
        logger.debug(f"[Jmavsim-{self.instance_id}] Controller initialized for port {self.port}.")

    def start(self, latitude=None, longitude=None):
        if self.proc and self.proc.poll() is None:
            logger.warning(f"[Jmavsim-{self.instance_id}] Process is already running.")
            return False

        # Clean up any stale processes on this port before starting
        try:
            subprocess.run(["pkill", "-9", "-f", f":{self.port}"],
                          capture_output=True, timeout=3)
            time.sleep(0.5)  # Let OS clean up ports
            logger.debug(f"[Jmavsim-{self.instance_id}] Cleaned up stale processes on port {self.port}")
        except Exception as e:
            logger.debug(f"[Jmavsim-{self.instance_id}] Port cleanup error (non-fatal): {e}")

        env = dict(os.environ, PX4_SIM_SPEED_FACTOR=self.speed_factor, HEADLESS=self.headless)
        if latitude is not None and longitude is not None:
            env['PX4_HOME_LAT'] = str(latitude)
            env['PX4_HOME_LON'] = str(longitude)
            logger.debug(f"[Jmavsim-{self.instance_id}] Modifying start location: LAT={latitude}, LON={longitude}")

        cmd = ["/usr/bin/env", "bash", JMAVSIM_SCRIPT, '-p', str(self.port), '-l']

        self.proc = subprocess.Popen(cmd,
                                    env=env,
                                    cwd=os.path.dirname(JMAVSIM_SCRIPT),
                                    preexec_fn=os.setsid,  # Create a new process group
                                    stdout=subprocess.DEVNULL,  # Suppress stdout
                                    stderr=subprocess.DEVNULL)  # Suppress stderr
        
        logger.info(f"[Jmavsim-{self.instance_id}] Process started with PID {self.proc.pid}.")

        return True
    
    def wait_for_ready(self, timeout=20):
        """Waits for the JMAVSIM TCP port to become active."""
        if not self.proc or self.proc.poll() is not None:
            logger.error(f"[Jmavsim-{self.instance_id}] Process not running, cannot wait.")
            return False

        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                with socket.create_connection(("localhost", self.port), timeout=1):
                    logger.info(f"[Jmavsim-{self.instance_id}] Simulator is ready on port {self.port}.")
                    return True
            except (socket.timeout, ConnectionRefusedError):
                time.sleep(0.5)
        
        logger.error(f"[Jmavsim-{self.instance_id}] Timeout waiting for simulator on port {self.port}.")
        return False


    def stop(self):
        if self.proc and self.proc.poll() is None:
            pid = self.proc.pid
            pgid = os.getpgid(pid)
            logger.info(f"[{self.__class__.__name__}-{self.instance_id}] Stopping process group PGID={pgid}")

            # 1. Graceful shutdown
            try:
                os.killpg(pgid, signal.SIGINT)
                self.proc.wait(timeout=10)
                logger.info(f"[{self.__class__.__name__}-{self.instance_id}] Process group PGID={pgid} stopped gracefully.")
                self.proc = None
                return
            except (subprocess.TimeoutExpired, ProcessLookupError):
                pass # Continue to force kill

            # 2. Force kill
            logger.warning(f"[{self.__class__.__name__}-{self.instance_id}] Graceful shutdown failed for PGID={pgid}. Killing.")
            try:
                os.killpg(pgid, signal.SIGKILL)
                subprocess.run(["pkill", "-9", "-f", f":{4560 + self.instance_id}"], capture_output=True, timeout=3)
            except ProcessLookupError:
                pass # Already gone

            # 3. Wait until the process fully disappears
            t0 = time.time()
            while time.time() - t0 < 10: # Check for up to 10 seconds
                if self.proc.poll() is not None:
                    logger.info(f"[{self.__class__.__name__}-{self.instance_id}] Process group PGID={pgid} confirmed terminated.")
                    self.proc = None
                    return
                time.sleep(0.1)

            logger.error(f"[{self.__class__.__name__}-{self.instance_id}] FAILED to confirm termination of PGID={pgid}.")

        self.proc = None