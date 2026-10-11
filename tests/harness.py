"""Shared plumbing for the suite: reporting, and disposable servers.

Every test gets its own ports and its own data directory. Nothing here touches
the library or config of a real install, and two runs cannot collide.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
MOCK = Path(__file__).resolve().parent / "mock_comfy.py"
sys.path.insert(0, str(ROOT))
# The suite talks to servers on this machine; a proxy in the environment would
# swallow every request.
os.environ["NO_PROXY"] = os.environ["no_proxy"] = "*"


class Suite:
    """Collects checks so one failure does not hide the rest."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.passed = 0
        self.failures: list[str] = []
        print(f"\n== {name}")

    def check(self, what: str, ok: bool, detail: str = "") -> bool:
        if ok:
            self.passed += 1
            print(f"  ok   {what}")
        else:
            self.failures.append(what)
            print(f"  FAIL {what}" + (f" — {detail}" if detail else ""))
        return ok


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_http(url: str, timeout: float = 30) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        try:
            requests.get(url, timeout=2)
            return True
        except requests.RequestException:
            time.sleep(0.2)
    return False


# Every model file a set-up install has, as safetensors headers that declare
# their own (tiny) length — complete as far as the app can tell.
MODEL_FILES = [
    "diffusion_models/minimax_music3_dit_int8_convrot.safetensors",
    "text_encoders/minimax_music3_text_encoder_pruned_int8_convrot.safetensors",
    "vae/minimax_music3_dav.safetensors",
]


def tiny_safetensors(path: Path, payload: int = 16) -> None:
    head = json.dumps({"w": {"dtype": "U8", "shape": [payload],
                             "data_offsets": [0, payload]}}).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(len(head).to_bytes(8, "little") + head + b"\0" * payload)


class Stack:
    """A mock ComfyUI and an M3 Studio pointed at it, torn down on exit."""

    def __init__(self, mock_env: dict | None = None, models: bool = True,
                 setup_complete: bool = True, config: dict | None = None,
                 app_env: dict | None = None) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="m3test-"))
        self.data = self.tmp / "data"
        self.models = self.tmp / "models"
        self.models.mkdir(parents=True)
        if models:
            for rel in MODEL_FILES:
                tiny_safetensors(self.models / rel)
        self.comfy_port = free_port()
        self.app_port = free_port()
        self.prompt_log = self.tmp / "prompts.jsonl"
        env = {**os.environ, "MOCK_DELAY": "1",
               "MOCK_PROMPT_LOG": str(self.prompt_log), **(mock_env or {})}
        self.mock = subprocess.Popen([sys.executable, str(MOCK),
                                      str(self.comfy_port)], env=env)
        self.data.mkdir(parents=True)
        cfg = {"comfy_url": f"http://127.0.0.1:{self.comfy_port}",
               "models_dir": str(self.models), "comfy_dir": "",
               "python": "", "managed": False, "auto_start_comfy": False,
               "setup_complete": setup_complete, **(config or {})}
        (self.data / "config.json").write_text(json.dumps(cfg))
        app_env = {**os.environ, "M3_STUDIO_DATA": str(self.data),
                   "M3_STUDIO_PORT": str(self.app_port),
                   "M3_STUDIO_NO_BROWSER": "1", "M3_STUDIO_NO_SEARCH": "1",
                   **(app_env or {})}
        self.log = open(self.tmp / "server.log", "w")
        self.app = subprocess.Popen([sys.executable, str(ROOT / "server.py")],
                                    env=app_env, stdout=self.log,
                                    stderr=subprocess.STDOUT, cwd=str(ROOT))
        self.url = f"http://127.0.0.1:{self.app_port}"
        wait_http(f"http://127.0.0.1:{self.comfy_port}/system_stats")
        if not wait_http(self.url + "/api/config"):
            raise RuntimeError("M3 Studio did not start: "
                               + (self.tmp / "server.log").read_text()[-2000:])

    def get(self, path: str, **kw):
        return requests.get(self.url + path, timeout=30, **kw)

    def post(self, path: str, body=None, **kw):
        return requests.post(self.url + path, json=body, timeout=60, **kw)

    def prompts(self) -> list[dict]:
        if not self.prompt_log.exists():
            return []
        return [json.loads(x) for x in self.prompt_log.read_text().splitlines()]

    def wait_job(self, job_id: str, timeout: float = 60) -> dict:
        end = time.time() + timeout
        while time.time() < end:
            for j in self.get("/api/jobs").json():
                if j["id"] == job_id and j["status"] != "running":
                    return j
            time.sleep(0.4)
        raise TimeoutError(f"job {job_id} still running")

    def wait_ready(self, timeout: float = 30) -> dict:
        end = time.time() + timeout
        st = {}
        while time.time() < end:
            st = self.get("/api/status").json()
            if st.get("ready"):
                return st
            time.sleep(0.4)
        return st

    def close(self) -> None:
        for p in (self.app, self.mock):
            p.terminate()
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
        self.log.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
