import atexit
import subprocess
import os

import psutil
import requests


class PhosphoClient:
    BASE_URL = "http://localhost:80"

    def __init__(self, auto_stop: bool | None = None):
        self._process = None
        if auto_stop is None:
            auto_stop = os.getenv("PHOSPHOBOT_AUTOSTOP", "0") in ("1", "true", "True")
        self._auto_stop = auto_stop
        if self._auto_stop:
            atexit.register(self.stop)

    def start(self):
        if self._process is not None and self._process.poll() is None:
            print("[phosphobot] Already started (PID={})".format(self._process.pid))
            return
        print("[phosphobot] Starting...")
        self._process = subprocess.Popen(["phosphobot", "run"])
        print(f"[phosphobot] Started with PID={self._process.pid}")

    def stop(self):
        try:
            print("phosphobot: calling .stop()")
        except Exception:
            pass
        found = False
        if self._process and self._process.poll() is None:
            try:
                proc = psutil.Process(self._process.pid)
                for child in proc.children(recursive=True):
                    try:
                        print(f"[phosphobot] Killing child PID={child.pid}")
                    except Exception:
                        pass
                    child.kill()
                try:
                    print(f"[phosphobot] Killing main PID={proc.pid}")
                except Exception:
                    pass
                proc.kill()
                found = True
            except psutil.NoSuchProcess:
                pass
            self._process = None

        # Fallback: kill any remaining phosphobot processes by name
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmd = proc.info['cmdline']
                if cmd and any("phosphobot" in c for c in cmd):
                    try:
                        print(f"[phosphobot] Force killing stray PID={proc.pid}")
                    except Exception:
                        pass
                    proc.kill()
                    found = True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        if not found:
            try:
                print("[phosphobot] No process found to kill.")
            except Exception:
                pass

    def post(
        self,
        endpoint: str,
        json: dict | None = None,
        params: dict | None = None,
        return_response: bool = False,
    ):
        # if self._process is None or self._process.poll() is not None:
        #     self.start()
        url = self.BASE_URL + endpoint
        try:
            response = requests.post(url, json=json, params=params)
            if return_response:
                return response
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"[phosphobot] POST request failed: {e}")
            return {"status": "error", "error": str(e)}

    def get(self, endpoint: str, params: dict | None=None, return_response: bool = False):
        # if self._process is None or self._process.poll() is not None:
        #     self.start()
        url = self.BASE_URL + endpoint
        try:
            response = requests.get(url, params=params)
            if return_response:
                return response
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"[phosphobot] GET request failed: {e}")
            return {"status": "error", "error": str(e)}

    def hello(self, name: str | None = None, robot_id: int | None = None, return_response: bool = False):
        payload = {"name": name} if name is not None else {}
        params = {"robot_id": robot_id} if robot_id is not None else None
        return self.post("/move/hello", json=payload, params=params, return_response=return_response)
