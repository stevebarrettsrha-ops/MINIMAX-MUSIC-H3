"""A stand-in ComfyUI: /object_info captured from a real one, plus a queue.

Run:  python tests/mock_comfy.py PORT

The schema is not written by hand: object_info.json is what a real ComfyUI
(v0.39) answers /object_info with, trimmed to the nodes the Music 3 graph uses
(tests/capture_object_info.py makes it). A hand-written stand-in drifts from
the thing it stands in for, and is then worth very little.

Execution plays the real graph's shape: the loaders, then the text encoder
counting frames (25 a second) until the song "ends", then the sampler counting
steps, then the decoder — each announced over /ws exactly as ComfyUI does.

Knobs (environment):
  MOCK_DELAY        seconds a render takes, roughly (default 2)
  MOCK_VRAM         bytes of video memory to report (default 8 GB)
  MOCK_FAIL_NODE    class_type to fail inside (with MOCK_FAIL_MSG)
  MOCK_CUT          1 = the song runs to its max_duration (cut off)
  MOCK_BLANK_MODELS N = the first N /object_info answers list no model files
  MOCK_NO_ARGV      1 = /system_stats names no main.py
  MOCK_COMFY_ROOT   folder /system_stats reports the engine runs from
"""
import base64
import copy
import hashlib
import io
import json
import os
import pathlib
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

OBJECT_INFO = json.loads(
    (pathlib.Path(__file__).with_name("object_info.json")).read_text())

BLANK_CALLS = int(os.environ.get("MOCK_BLANK_MODELS", "0"))
CALLS = 0
HISTORY = {}
PROMPTS = {}
QUEUE_PENDING = []
QUEUE_RUNNING = []
INTERRUPTS = []
DELETES = []
LOCK = threading.Lock()
RUN_LOCK = threading.Lock()
WS_CLIENTS = []
DELAY = float(os.environ.get("MOCK_DELAY", "2"))
LOADER_FIELDS = {"UNETLoader": "unet_name", "CLIPLoader": "clip_name",
                 "VAELoader": "vae_name"}


def object_info():
    """The schema, with the model lists withheld for the first N calls —
    ComfyUI scans its folders once, so files that landed later are unseen."""
    global CALLS
    with LOCK:
        CALLS += 1
        withhold = CALLS <= BLANK_CALLS
    if not withhold:
        return OBJECT_INFO
    out = copy.deepcopy(OBJECT_INFO)
    for node, field in LOADER_FIELDS.items():
        spec = out[node]["input"]["required"][field]
        spec[0] = []
    return out


def wav_bytes(seconds=3.0, rate=8000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(rate * seconds))
    return buf.getvalue()


_FLAC: dict = {}


def flac_bytes(seconds: float) -> bytes:
    """A FLAC file `seconds` long. A real one, made with ffmpeg, when there
    is an ffmpeg — the join of a long song decodes the parts — and otherwise
    a header the app can at least read the length from."""
    key = round(seconds, 2)
    if key in _FLAC:
        return _FLAC[key]
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        with tempfile.TemporaryDirectory() as tmp:
            dst = f"{tmp}/a.flac"
            done = subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-f", "lavfi",
                                   "-i", f"anullsrc=r=8000:cl=mono:d={seconds}",
                                   dst], capture_output=True)
            if done.returncode == 0 and os.path.getsize(dst):
                _FLAC[key] = open(dst, "rb").read()
                return _FLAC[key]
    _FLAC[key] = fake_flac(seconds)
    return _FLAC[key]


def fake_flac(seconds: float) -> bytes:
    rate, total = 32000, int(32000 * seconds)
    packed = (rate << 44) | (1 << 41) | (15 << 36) | total
    info = (b"\x00\x10\x00\x10" + b"\x00\x00\x00" + b"\x00\x00\x00"
            + packed.to_bytes(8, "big") + b"\x00" * 16)
    return b"fLaC" + bytes([0x80, 0, 0, len(info)]) + info + b"\x00" * 64


def ws_send(obj):
    """One unmasked text frame to every connected /ws client."""
    data = json.dumps(obj).encode()
    head = bytearray([0x81])
    if len(data) < 126:
        head.append(len(data))
    else:
        head += bytes([126]) + struct.pack(">H", len(data))
    frame = bytes(head) + data
    with LOCK:
        for sock in WS_CLIENTS[:]:
            try:
                sock.sendall(frame)
            except OSError:
                WS_CLIENTS.remove(sock)


def node_of(graph, class_type):
    return next(((nid, n) for nid, n in graph.items()
                 if n["class_type"] == class_type), (None, None))


def execute(pid, graph):
    with RUN_LOCK:                      # one at a time, like the real queue
        with LOCK:
            if pid not in QUEUE_PENDING:   # deleted while waiting
                return
            QUEUE_PENDING.remove(pid)
            QUEUE_RUNNING.append(pid)
        _execute(pid, graph)


def interrupted(pid):
    with LOCK:
        return pid in INTERRUPTS


def finish(pid, status, outputs=None, messages=None):
    with LOCK:
        if pid in QUEUE_RUNNING:
            QUEUE_RUNNING.remove(pid)
        HISTORY[pid] = {"status": {"status_str": status, "messages": messages or []},
                        "outputs": outputs or {}}


UPLOADS: dict = {}          # name -> seconds, for LoadAudio


def _join(pid, graph, save_id):
    """A join graph: LoadAudio parts into AudioConcat, saved. The result is
    as long as its parts together."""
    total = 0.0
    for nid, node in graph.items():
        if node["class_type"] == "LoadAudio":
            name = node["inputs"].get("audio", "")
            if name not in UPLOADS:
                finish(pid, "error", messages=[["execution_error", {
                    "node_type": "LoadAudio",
                    "exception_message": f"Invalid audio file: {name}"}]])
                return
            total += UPLOADS[name]
    time.sleep(0.3)
    fmt = (graph[save_id]["inputs"].get("format") or "flac")
    name = f"{pid}.{fmt}"
    PROMPTS[name] = total
    ws_send({"type": "execution_success", "data": {"prompt_id": pid}})
    finish(pid, "success", outputs={save_id: {"audio": [
        {"filename": name, "subfolder": "audio", "type": "output"}]}})


def _execute(pid, graph):
    ws_send({"type": "execution_start", "data": {"prompt_id": pid}})
    fail = os.environ.get("MOCK_FAIL_NODE", "")
    enc_id, enc = node_of(graph, "MiniMaxMusic3TextEncode")
    if enc_id is None:
        save_id, _ = node_of(graph, "SaveAudioAdvanced")
        if save_id is None:
            save_id, _ = node_of(graph, "SaveAudio")
        _join(pid, graph, save_id)
        return
    ks_id, ks = node_of(graph, "KSampler")
    save_id, _ = node_of(graph, "SaveAudioAdvanced")
    if save_id is None:
        save_id, _ = node_of(graph, "SaveAudio")
    ceiling = float(enc["inputs"].get("max_duration", 120))
    frames = int(ceiling * 25)
    # A song ends on its own at 70% of the ceiling, unless asked to run out.
    sung = frames if os.environ.get("MOCK_CUT") else int(frames * 0.7)
    order = [nid for nid, n in graph.items()
             if n["class_type"] in LOADER_FIELDS] + [enc_id, ks_id]
    ticks = max(2, int(DELAY * 2))
    for nid in order:
        ws_send({"type": "executing", "data": {"node": nid, "prompt_id": pid}})
        if graph[nid]["class_type"] == fail:
            finish(pid, "error", messages=[["execution_error", {
                "node_type": fail, "exception_message":
                    os.environ.get("MOCK_FAIL_MSG", "CUDA out of memory. "
                                   "Tried to allocate 2.00 GiB")}]])
            return
        if nid == enc_id:
            for i in range(ticks):
                time.sleep(0.5)
                ws_send({"type": "progress", "data": {
                    "value": int(sung * (i + 1) / ticks), "max": frames,
                    "prompt_id": pid, "node": nid}})
                if interrupted(pid):
                    break
        elif nid == ks_id:
            steps = int(ks["inputs"].get("steps", 30))
            for i in range(ticks):
                time.sleep(0.25)
                ws_send({"type": "progress", "data": {
                    "value": int(steps * (i + 1) / ticks), "max": steps,
                    "prompt_id": pid, "node": nid}})
                if interrupted(pid):
                    break
        if interrupted(pid):
            finish(pid, "error", messages=[["execution_interrupted", {
                "node_type": graph[nid]["class_type"]}]])
            return
    seconds = sung / 25
    fmt = (graph[save_id]["inputs"].get("format") or "flac")
    name = f"{pid}.{fmt}"
    PROMPTS[name] = seconds
    ws_send({"type": "executing", "data": {"node": None, "prompt_id": pid}})
    ws_send({"type": "execution_success", "data": {"prompt_id": pid}})
    finish(pid, "success", outputs={save_id: {"audio": [
        {"filename": name, "subfolder": "audio", "type": "output"}]}})


def validate(graph):
    """Reject what ComfyUI itself would reject: unknown nodes or inputs, bad
    enums, wrong scalar types, dangling links, missing required inputs."""
    errs = {}
    for nid, node in graph.items():
        cls = node["class_type"]
        info = OBJECT_INFO.get(cls)
        if not info:
            errs[nid] = {"class_type": cls, "errors": [
                {"message": "Node not found", "details": cls}]}
            continue
        spec = dict(info["input"].get("required", {}))
        spec.update(info["input"].get("optional", {}))
        required = dict(info["input"].get("required", {}))
        # A dynamic combo's chosen option brings its own inputs along, named
        # after their parent ("format.quality") as ComfyUI v0.39 names them.
        # The bare name is tolerated, as the real server ignores it.
        for name, definition in list(spec.items()):
            if (not isinstance(definition, list) or len(definition) < 2
                    or isinstance(definition[0], list)
                    or not isinstance(definition[1], dict)):
                continue
            opts = definition[1].get("options") or []
            if not (opts and isinstance(opts[0], dict)):
                continue
            for option in opts:
                if option.get("key") != node["inputs"].get(name):
                    continue
                for group in ("required", "optional"):
                    nested = (option.get("inputs") or {}).get(group) or {}
                    dotted = {f"{name}.{k}": v for k, v in nested.items()}
                    spec.update(dotted)
                    spec.update(nested)
                    if group == "required":
                        required.update(dotted)
        node_errs = []
        for name, value in node["inputs"].items():
            if name not in spec:
                node_errs.append({"message": "Unknown input", "details": name})
                continue
            definition = spec[name]
            kind = definition[0]
            opts = definition[1] if len(definition) > 1 and isinstance(definition[1], dict) else {}
            if isinstance(value, list) and len(value) == 2 and isinstance(value[1], int):
                if str(value[0]) not in graph:
                    node_errs.append({"message": "Link to a node that is not in "
                                                 "the prompt", "details": name})
                continue
            if isinstance(kind, list):
                if value not in kind:
                    node_errs.append({"message": "Value not in list",
                                      "details": f"{name}: {value!r}"})
            elif str(kind).startswith(("COMBO", "COMFY_DYNAMICCOMBO")):
                choices = opts.get("options") or []
                keys = [c["key"] if isinstance(c, dict) else c for c in choices]
                if keys and value not in keys:
                    node_errs.append({"message": "Value not in list",
                                      "details": f"{name}: {value!r}"})
            elif kind == "INT" and (not isinstance(value, int) or isinstance(value, bool)):
                node_errs.append({"message": "Wrong type", "details": f"{name} INT"})
            elif kind == "FLOAT" and not isinstance(value, (int, float)):
                node_errs.append({"message": "Wrong type", "details": f"{name} FLOAT"})
            elif kind == "STRING" and not isinstance(value, str):
                node_errs.append({"message": "Wrong type", "details": f"{name} STRING"})
            elif kind in ("INT", "FLOAT"):
                if "min" in opts and value < opts["min"]:
                    node_errs.append({"message": "Value smaller than min",
                                      "details": f"{name}: {value}"})
                if "max" in opts and value > opts["max"]:
                    node_errs.append({"message": "Value bigger than max",
                                      "details": f"{name}: {value}"})
        for name in required:
            if name not in node["inputs"]:
                node_errs.append({"message": "Required input is missing",
                                  "details": name})
        if node_errs:
            errs[nid] = {"class_type": cls, "errors": node_errs}
    return errs


def probe_seconds(data: bytes) -> float:
    """How long an uploaded file is, from its own header (flac) or ffprobe."""
    if data[:4] == b"fLaC":
        packed = int.from_bytes(data[18:26], "big")
        rate, total = packed >> 44, packed & ((1 << 36) - 1)
        if rate and total:
            return total / rate
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as fh:
            fh.write(data)
        try:
            out = subprocess.run([ffprobe, "-v", "error", "-show_entries",
                                  "format=duration", "-of",
                                  "default=noprint_wrappers=1:nokey=1", fh.name],
                                 capture_output=True, text=True, timeout=20).stdout
            return float(out.strip() or 0)
        finally:
            os.unlink(fh.name)
    return 0.0


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        raw = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/ws":
            key = self.headers.get("Sec-WebSocket-Key", "")
            accept = base64.b64encode(hashlib.sha1(
                (key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()
            ).digest()).decode()
            self.send_response(101, "Switching Protocols")
            self.send_header("Upgrade", "websocket")
            self.send_header("Connection", "Upgrade")
            self.send_header("Sec-WebSocket-Accept", accept)
            self.end_headers()
            with LOCK:
                WS_CLIENTS.append(self.connection)
            try:
                while self.connection.recv(1024):
                    pass
            except OSError:
                pass
            self.close_connection = True
            return
        if p == "/object_info":
            self._send(200, object_info())
        elif p == "/system_stats":
            system = {"comfyui_version": "0.39.0"}
            if not os.environ.get("MOCK_NO_ARGV"):
                root = os.environ.get("MOCK_COMFY_ROOT", "/opt/ComfyUI")
                system["argv"] = [f"{root}/main.py"]
            vram = int(os.environ.get("MOCK_VRAM", str(8 * 1024 ** 3)))
            self._send(200, {"system": system, "devices": [
                {"name": "cuda:0 NVIDIA GeForce RTX 4060", "type": "cuda",
                 "index": 0, "vram_total": vram}]})
        elif p.startswith("/history/"):
            pid = p.rsplit("/", 1)[-1]
            with LOCK:
                self._send(200, {pid: HISTORY[pid]} if pid in HISTORY else {})
        elif p == "/queue":
            with LOCK:
                self._send(200, {
                    "queue_running": [[0, pid, {}, {}, []] for pid in QUEUE_RUNNING],
                    "queue_pending": [[0, pid, {}, {}, []] for pid in QUEUE_PENDING]})
        elif p == "/view":
            name = self.path.split("filename=")[-1].split("&")[0]
            seconds = PROMPTS.get(name, 3.0)
            if name.endswith(".flac"):
                self._send(200, flac_bytes(seconds), "audio/flac")
            else:
                self._send(200, wav_bytes(seconds), "audio/mpeg")
        else:
            self._send(404, {"error": "no route " + p})

    def do_POST(self):
        p = self.path.split("?")[0]
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n) if n else b""
        if p == "/prompt":
            graph = json.loads(raw)["prompt"]
            bad = validate(graph)
            if bad:
                self._send(400, {"error": {"type": "prompt_outputs_failed_validation",
                                           "message": "Prompt outputs failed validation",
                                           "details": ""},
                                 "node_errors": bad})
                return
            with LOCK:
                pid = f"pid{len(HISTORY) + len(QUEUE_PENDING) + len(QUEUE_RUNNING) + 1}"
                QUEUE_PENDING.append(pid)
                pathlib.Path(os.environ.get("MOCK_PROMPT_LOG", os.devnull)).open(
                    "a").write(json.dumps({"pid": pid, "prompt": graph}) + "\n")
            threading.Thread(target=execute, args=(pid, graph), daemon=True).start()
            self._send(200, {"prompt_id": pid, "number": 1, "node_errors": {}})
        elif p == "/upload/image":
            # A multipart upload: keep the file's name and its length, which
            # is what a later LoadAudio of it needs to answer with.
            import email
            from email import policy
            msg = email.message_from_bytes(
                b"Content-Type: " + self.headers.get("Content-Type", "").encode()
                + b"\r\n\r\n" + raw, policy=policy.default)
            name, seconds = "", 0.0
            for part in msg.iter_parts():
                if part.get_param("name", header="content-disposition") == "image":
                    name = part.get_filename() or "upload"
                    data = part.get_payload(decode=True) or b""
                    seconds = probe_seconds(data)
            UPLOADS[name] = seconds
            self._send(200, {"name": name, "subfolder": "", "type": "input"})
        elif p == "/interrupt":
            with LOCK:
                INTERRUPTS.extend(QUEUE_RUNNING)
            self._send(200, {})
        elif p == "/queue":
            body = json.loads(raw) if raw else {}
            with LOCK:
                for pid in body.get("delete", []):
                    DELETES.append(pid)
                    if pid in QUEUE_PENDING:
                        QUEUE_PENDING.remove(pid)
                        HISTORY[pid] = {"status": {"status_str": "error", "messages": [
                            ["execution_interrupted", {"node_type": "-"}]]},
                            "outputs": {}}
            self._send(200, {})
        else:
            self._send(404, {"error": "no route " + p})


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8188
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
