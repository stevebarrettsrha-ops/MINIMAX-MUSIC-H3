"""
server.py - M3 Studio backend: a local, Suno-style front end for MiniMax
Music 3, driving ComfyUI.

Run:  python server.py            (opens http://127.0.0.1:7789)
"""

from __future__ import annotations

import json
import mimetypes
import re
import os
import random
import subprocess
import threading
import time
import uuid
import webbrowser
from pathlib import Path
from urllib.parse import urlsplit

import requests
from flask import Flask, jsonify, request, send_file, send_from_directory

import bootstrap
import comfy
import lyricist
import parts
import manager
from bootstrap import (ComfyProcess, Progress, clean_url,
                       comfy_online, comfy_port, detect_comfy_dirs,
                       load_config, save_config)
from comfy import ComfyClient, ComfyError, explain_failure

DATA_DIR = bootstrap.DATA_DIR          # honours M3_STUDIO_DATA
TRACKS_DIR = DATA_DIR / "tracks"
LIBRARY_PATH = DATA_DIR / "library.json"
WEB_DIR = bootstrap.APP_DIR / "web"
# 7789, beside YuE Studio's 7788, so the two apps can run side by side.
PORT = int(os.environ.get("M3_STUDIO_PORT", "7789"))

app = Flask(__name__, static_folder=None)
# Nothing this app receives is more than a few pages of text.
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024

cfg = load_config()
progress = Progress()

# Set while a search walks the drives for ComfyUI, so the Engine page says
# "searching" instead of "missing" and Recheck does not start a second walk.
locating = threading.Event()
# When the last search ended. The page re-polls while "searching"; a poll
# right after a fruitless search must show "not found" (and Install), not
# start the next walk of the drives — so Recheck searches again only after
# this rest.
_search_done = [float("-inf")]
SEARCH_REST = 30.0
_locate_lock = threading.Lock()


def _say(msg: str) -> None:
    progress.log(msg)               # the setup log, and the console with it


def _search_off() -> bool:
    """M3_STUDIO_NO_SEARCH=1: the tests' configs point at made-up folders on
    purpose, and must neither be repaired nor adopt a real install."""
    return os.environ.get("M3_STUDIO_NO_SEARCH") == "1"


def _heal(search: bool = False) -> None:
    """Verify the saved locations; repair any that moved.

    Without `search` only the quick repair runs (a moved or renamed app
    folder) and never waits on a search in progress. With it, a ComfyUI that
    is still nowhere is searched for across the drives; searches take turns.
    """
    if _search_off():
        locating.clear()            # never leave a caller's "searching" set
        return
    if not _locate_lock.acquire(blocking=search):
        return                      # a search is running; it will report
    try:
        if search:
            locating.set()
        notes = bootstrap.verify_locations(cfg, search=search, log=_say)
        if notes:
            save_config(cfg)
            for n in notes:
                _say(n)
        if search:
            for line in bootstrap.location_report(cfg):
                _say("Verified " + line)
    except Exception as exc:  # noqa: BLE001 — a failed check must not stop boot
        _say(f"Could not verify saved locations: {exc}")
    finally:
        if search:
            _search_done[0] = time.monotonic()
            locating.clear()
        _locate_lock.release()


def _needs_search() -> bool:
    return bootstrap.comfy_lost(cfg)


def _rested() -> bool:
    return time.monotonic() - _search_done[0] > SEARCH_REST


_heal()
comfy_proc = ComfyProcess()
client = ComfyClient(cfg["comfy_url"])

jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()
# Re-entrant: every change to the library is a read, an edit and a write, and
# all three have to happen without another thread slipping in between. A song
# finishing while a track is being deleted otherwise writes back a list that
# never knew about the other one, and whichever wrote first is simply gone.
library_lock = threading.RLock()


# --------------------------------------------------------------------------- #
# request input
#
# Every value below arrives as JSON from outside this process. The front end
# always sends the right shapes, but nothing enforces that, and a value of the
# wrong type used to reach str.strip() or int() and end the request in a 500 —
# or, worse, be written into the live config on its way to failing.
# --------------------------------------------------------------------------- #
def as_text(value, default: str = "") -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return default


def as_int(value, default: int, low: int, high: int) -> int:
    try:
        number = int(float(value))
    except (TypeError, ValueError, OverflowError):
        return default
    return max(low, min(number, high))


def as_float(value, default: float, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if number != number:                      # NaN compares unequal to itself
        return default
    return max(low, min(number, high))


def as_cursor(raw: str) -> int:
    """A log cursor from a query string, however mangled."""
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 0


def json_object() -> dict:
    """Return request JSON when it is an object, otherwise an empty object.

    Valid JSON can also be an array, string, number, or null.  Those values
    used to reach the first ``.get()`` in a write route and turn malformed
    input into a 500 response.  Keeping the shape check at the HTTP boundary
    lets every endpoint handle such input consistently.
    """
    body = request.get_json(silent=True)
    return body if isinstance(body, dict) else {}


def as_bool(value, default: bool = False) -> bool:
    """Read a JSON boolean without treating strings such as "false" as true."""
    return value if isinstance(value, bool) else default


# Bounds are the Music 3 and KSampler node limits, so a silly number is
# clamped here instead of being rejected by ComfyUI after the job has started.
def clean_generate(body: dict) -> dict:
    return {
        "caption": as_text(body.get("caption"))[:8000],
        "lyrics": as_text(body.get("lyrics"))[:20000],
        "title": as_text(body.get("title"))[:120],
        "dit": as_text(body.get("dit"))[:260],
        "te": as_text(body.get("te"))[:260],
        "format": as_text(body.get("format"))[:20],
        "quality": as_text(body.get("quality"))[:20],
        "sampler": as_text(body.get("sampler"))[:60] or None,
        "scheduler": as_text(body.get("scheduler"))[:60] or None,
        "vocal": (as_text(body.get("vocal"))
                  if as_text(body.get("vocal")) in ("male", "female", "duet",
                                                    "instrumental") else ""),
        "instrumental": as_bool(body.get("instrumental")),
        "tiled_decode": as_bool(body.get("tiled_decode"), True),
        "count": as_int(body.get("count"), 1, 1, 4),
        # Past the engine's single pass (360 s) the song is made in parts;
        # parts.MAX_SONG is the most the planner will take on.
        "duration": as_int(body.get("duration"), 120, 1, parts.MAX_SONG),
        "steps": as_int(body.get("steps"), 30, 1, 200),
        "top_k": as_int(body.get("top_k"), 50, 1, 16384),
        "tile_size": as_int(body.get("tile_size"), 1536, 32, 8192),
        "overlap": as_int(body.get("overlap"), 64, 0, 1024),
        "cfg": as_float(body.get("cfg"), 1.7, 0.0, 30.0),
        "cfg_scale": as_float(body.get("cfg_scale"), 1.7, 0.0, 30.0),
        # None means "pick a fresh one", which is not the same as seed 0.
        "seed": (None if body.get("seed") in (None, "")
                 else as_int(body.get("seed"), 0, 0, 2**63 - 1)),
    }


LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", ""}


@app.before_request
def only_this_machine():
    """Answer only to this machine's own names.

    The server listens on loopback, but a browser sends a page's requests to
    whatever a hostname resolves to — so a domain an attacker controls, pointed
    at 127.0.0.1, reaches this server as same-origin and can read every answer.
    Checking the name the request was addressed to closes that off and costs
    nothing for a real local visit.
    """
    host = (request.host or "").rsplit(":", 1)[0].strip("[]").lower()
    if host not in LOCAL_HOSTS:
        return jsonify({"error": "M3 Studio only answers on this machine."}), 403


# --------------------------------------------------------------------------- #
# library
# --------------------------------------------------------------------------- #
def read_library() -> list[dict]:
    """The song list — always a list of records, whatever the file holds.

    library.json is plain text in a folder people can open: it can be
    hand-edited, restored from an older copy, or left half-written by a crash.
    A shape we did not expect must not stop songs being saved or served, so
    anything that is not a usable record is simply left out.
    """
    with library_lock:
        if not LIBRARY_PATH.exists():
            return []
        try:
            items = json.loads(LIBRARY_PATH.read_text(encoding="utf-8"))
        except Exception:
            return []
        if not isinstance(items, list):
            return []
        return [i for i in items if isinstance(i, dict) and i.get("id")]


def model_size(role: str, name: str) -> int:
    """Bytes of one loader's file in the model library, or 0 if not found.

    Read off the disk rather than from the catalogue in MODELS, whose figures
    are rounded and can go stale.
    """
    base = Path(cfg["models_dir"]) if cfg.get("models_dir") else None
    folder = {"dit": "diffusion_models", "te": "text_encoders",
              "vae": "vae"}.get(role)
    if not base or not folder or not name or "/" in name or "\\" in name:
        return 0
    try:
        path = bootstrap.model_path(base, f"{folder}/{name}")
        return path.stat().st_size if path.is_file() else 0
    except OSError:
        return 0


def track_file(item: dict, kind: str = "") -> Path | None:
    """Where a library entry's audio lives, or None if it names nothing sane.

    A song is kept twice: a wav to listen to and keep, and an mp3 to send
    someone. `kind` picks one of those; without it you get the wav, or
    whatever single file an older song was saved as.

    Only ever a plain filename inside data/tracks: the name comes out of a
    file anyone can edit, and it is used to both serve and delete.
    """
    name = Path(str(item.get(kind or "file") or "")).name
    return (TRACKS_DIR / name) if name else None


def track_files(item: dict) -> list[Path]:
    """Everything on disk that belongs to this song."""
    found = []
    for kind in ("file", "mp3"):
        path = track_file(item, kind)
        if path and path not in found:
            found.append(path)
    return found


# --------------------------------------------------------------------------- #
# what a finished song is saved as
#
# ComfyUI writes flac, mp3 and opus itself, so those are rendered straight to
# the format asked for. It has no wav encoder at all, so a wav is rendered
# lossless and converted here — the one format that needs ffmpeg.
# --------------------------------------------------------------------------- #
def convert_audio(src: Path, dest: Path, args: tuple = ()) -> bool:
    """Re-encode one file into another. False if ffmpeg cannot or is absent."""
    ffmpeg = bootstrap.find_tool("ffmpeg")
    if not ffmpeg or src == dest:
        return False
    try:
        done = subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", str(src), *args,
             str(dest)], capture_output=True, text=True, timeout=900)
    except (OSError, subprocess.SubprocessError):
        return False
    return (done.returncode == 0 and dest.is_file()
            and dest.stat().st_size > 0)


def available_formats(rendered: list[str]) -> list[str]:
    """What Save as can offer: what this ComfyUI writes, and wav if we can."""
    formats = [f for f in rendered if f]
    if bootstrap.find_tool("ffmpeg") and "wav" not in formats:
        formats.insert(1 if formats else 0, "wav")
    return formats


def render_format(wanted: str) -> str:
    """What to ask ComfyUI for, to end up with what was asked for."""
    return "flac" if wanted == "wav" else wanted


def save_as(master: Path, track_id: str, wanted: str) -> Path:
    """The file the song is kept as, converting only when we have to.

    Everything but wav comes out of ComfyUI already in the right format. A wav
    is made from the lossless render, and if there is no ffmpeg to make it
    with, the song stays as that render — it still plays and downloads, and
    the Engine page offers to install ffmpeg in one press.
    """
    if wanted != "wav" or master.suffix.lower() == ".wav":
        return master
    dest = TRACKS_DIR / f"{track_id}.wav"
    if not convert_audio(master, dest):
        return master
    try:
        master.unlink(missing_ok=True)      # the wav holds everything it did
    except OSError:
        pass
    return dest


def write_library(items: list[dict]) -> None:
    """Replace the library in one step.

    Written beside the real file and moved over it, so a crash or a full disk
    leaves the previous song list intact instead of a half-written one.
    """
    with library_lock:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        tmp = LIBRARY_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(items, indent=2), encoding="utf-8")
        tmp.replace(LIBRARY_PATH)


def add_track(track: dict) -> None:
    with library_lock:
        items = read_library()
        items.insert(0, track)
        write_library(items)


def audio_duration(path: Path) -> float | None:
    """How long the finished song actually is, in seconds.

    Music 3 stops when the song ends, so a track is usually shorter than the
    max_duration it was asked for — the requested value is a ceiling and must
    never be shown as the length. Read it out of the file instead.

    flac and ogg/opus are parsed here because they need no decoder; anything
    else falls back to ffprobe, and if that is missing the front end backfills
    the real duration the first time the track is played.
    """
    for reader in (_duration_wav, _duration_flac, _duration_opus,
                   _duration_ffprobe):
        try:
            seconds = reader(path)
        except (OSError, ValueError, IndexError, ZeroDivisionError,
                subprocess.SubprocessError, __import__("wave").Error):
            continue
        if seconds and seconds > 0:
            return round(seconds, 2)
    return None


def _duration_wav(path: Path) -> float | None:
    if path.suffix.lower() != ".wav":
        return None
    import wave
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        return handle.getnframes() / rate if rate else None


def _duration_flac(path: Path) -> float | None:
    if path.suffix.lower() != ".flac":
        return None
    with open(path, "rb") as fh:
        if fh.read(4) != b"fLaC":
            return None
        fh.seek(8)  # past the STREAMINFO block header and the block sizes
        head = fh.read(18)
    if len(head) < 18:
        return None
    # 20 bits sample rate, 3 channels, 5 bits per sample, 36 bits total samples
    packed = int.from_bytes(head[10:18], "big")
    rate, total = packed >> 44, packed & ((1 << 36) - 1)
    return total / rate if rate and total else None


def _duration_opus(path: Path) -> float | None:
    """Ogg Opus only.

    Opus fixes its granule clock at 48 kHz whatever the source rate. An .ogg
    holding Vorbis counts granules at the stream's own rate instead, so reading
    one as Opus under-reports every length — bail out unless OpusHead is there
    and let ffprobe handle the rest.
    """
    if path.suffix.lower() not in (".opus", ".ogg", ".oga"):
        return None
    with open(path, "rb") as fh:
        head = fh.read(4096)
        fh.seek(max(0, path.stat().st_size - 65536))
        tail = fh.read()
    start = head.find(b"OpusHead")
    if start < 0:
        return None
    pre_skip = int.from_bytes(head[start + 10:start + 12], "little")
    page = tail.rfind(b"OggS")
    if page < 0 or page + 14 > len(tail):
        return None
    granule = int.from_bytes(tail[page + 6:page + 14], "little")
    if granule in (0, 0xFFFFFFFFFFFFFFFF):
        return None
    return max(0, granule - pre_skip) / 48000.0


def _duration_ffprobe(path: Path) -> float | None:
    ffprobe = bootstrap.find_tool("ffprobe")
    if not ffprobe:
        return None
    out = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, timeout=20,
    ).stdout.strip()
    return float(out) if out else None


def title_from(caption: str, lyrics: str) -> str:
    """A name for a song nobody named: its first sung line, or failing that
    the genre words at the start of its caption."""
    for line in (lyrics or "").splitlines():
        line = line.strip()
        if line and not line.startswith("[") and not line.startswith("("):
            words = line.split()
            return " ".join(words[:6]).strip(" ,.!?-").title() or "Untitled"
    return caption_genre(caption).title() or "Untitled"


def caption_genre(caption: str) -> str:
    """A short line that says what a caption is: the genre from its Basic
    Attributes, or its first meaningful words. Used for titles and for the
    one-line summary under a song."""
    text = caption or ""
    # The genre follows the last attribute: "bpm is 87. key is E, and scale
    # is minor. Alternative Rock / Pop Rock." — or the bpm when no key is set.
    for after in (r"scale is \w+", r"bpm is [\d\-–~ ]+"):
        m = re.search(after + r"[.,]\s*([^.\n]{3,80})", text, re.I)
        if m and "key is" not in m.group(1).lower():
            return m.group(1).strip(" .")
    for line in text.splitlines():
        line = re.sub(r"^[#*\s]+", "", line).strip()
        if not line or line.lower().rstrip(":") in ("global metadata", "vocal details",
                                                   "arrangement"):
            continue
        line = re.sub(r"^[A-Za-z &/]+:\s*", "", line)
        return " ".join(line.split()[:8]).strip(" ,.")
    return ""


def track_title(params: dict) -> str:
    """What to call this song. An instrumental never sings its lyrics box, so
    naming it after that text would describe a song nobody hears."""
    return params.get("title") or title_from(
        params.get("caption", ""),
        "" if params.get("instrumental") else params.get("lyrics", ""))


# --------------------------------------------------------------------------- #
# progress over the ComfyUI websocket (optional dependency)
# --------------------------------------------------------------------------- #
ws_progress: dict[str, dict] = {}


def ws_listener() -> None:
    try:
        import websocket  # websocket-client
    except ImportError:
        return
    while True:
        try:
            url = cfg["comfy_url"].replace("http://", "ws://").replace(
                "https://", "wss://")
            ws = websocket.WebSocket()
            ws.connect(f"{url}/ws?clientId={client.client_id}", timeout=10)
            current = None
            while True:
                raw = ws.recv()
                if not isinstance(raw, str):
                    continue
                msg = json.loads(raw)
                mtype, data = msg.get("type"), msg.get("data") or {}
                pid = data.get("prompt_id") or current
                if mtype == "execution_start":
                    current = data.get("prompt_id")
                elif mtype == "executing" and pid:
                    ws_progress.setdefault(pid, {})["node"] = data.get("node")
                elif mtype == "progress" and pid:
                    # Which node is counting matters: the encoder counts
                    # frames and the sampler counts steps, and one must not be
                    # read as the other.
                    ws_progress.setdefault(pid, {}).update(
                        value=data.get("value", 0), max=data.get("max", 0),
                        progress_node=data.get("node"))
                elif mtype in ("execution_success", "execution_error",
                               "execution_interrupted") and pid:
                    ws_progress.pop(pid, None)
        except Exception:
            time.sleep(4)


# --------------------------------------------------------------------------- #
# generation job
# --------------------------------------------------------------------------- #
TERMINAL = ("done", "error", "cancelled")

# How long a song may show no sign of life at all before it is called stuck.
# Measured from the last change ComfyUI reported — the node it is on, or the
# step it has reached — and only once it is actually executing. A song that is
# merely slow keeps moving and is never cut off: on an 8 GB card the text
# encoder streams through the card a layer at a time, and loading it can sit
# quiet for minutes before the first frame, which is why this is generous.
STALL_LIMIT = 1800.0
# A song ComfyUI has forgotten and never started is not going to begin later.
NEVER_STARTED_LIMIT = 300.0

# Where each phase sits on the job's progress bar. Writing the song (the 8B
# model generating 25 frames a second) is the long part; the diffusion render
# and the decode come after it.
PHASES = {comfy.N_ENC: ("Writing the song", 8, 70),
          comfy.N_SAMPLER: ("Rendering audio", 70, 92),
          comfy.N_DECODE: ("Decoding", 92, 94)}
LOADERS = {comfy.N_TE: "Loading the text encoder",
           comfy.N_DIT: "Loading the diffusion model",
           comfy.N_VAE: "Loading the audio decoder"}


def ran_out(built: dict, seconds: float | None) -> bool:
    """Did this song stop at its length limit rather than at its own end?

    Music 3 writes until the song's end token or max_duration, whichever
    comes first, and says nothing either way. A song as long as the limit it
    was given is one that ran into it: frames are 25 a second, so a cut song
    comes out at the limit to within a frame or two of decoding.
    """
    limit = built.get("duration") or 0
    return bool(seconds and limit and seconds >= limit - 1.0)


class JobStopped(Exception):
    """The person pressed Stop."""


class JobFailed(Exception):
    """Something a person can act on, worded for the job card."""


def build_parts(p: dict) -> list[dict]:
    """Every pass a song needs, built and validated up front.

    One part when the length fits the engine's single pass; otherwise the
    lyrics are cut into parts that each do (parts.plan), each with its own
    caption note and a seed of the same family, so the set can be made again
    exactly from the first seed.
    """
    seed = p.get("seed")
    seed = int(seed) if seed not in (None, "") else random.randint(0, 2**31 - 1)
    planned = parts.plan(p.get("lyrics") or "", bool(p.get("instrumental")),
                         int(p.get("duration") or 120), client.duration_limit())
    out = []
    for part in planned:
        built = client.build_prompt({
            **p,
            "caption": parts.part_caption(p.get("caption") or "", part["index"],
                                          part["count"]),
            "lyrics": part["lyrics"], "duration": part["seconds"],
            "seed": seed + 1000 * (part["index"] - 1)})
        built["index"], built["count"] = part["index"], part["count"]
        out.append(built)
    return out


def wait_prompt(job_id: str, prompt_id: str, set_state, label: str,
                low: float, high: float) -> tuple[list[dict], float]:
    """Follow one prompt to its outputs. Progress is mapped into [low, high]
    of the job's bar. Raises JobStopped or JobFailed."""
    def scale(p: float) -> float:
        return round(low + (max(6.0, min(p, 94.0)) - 6.0) / 88.0 * (high - low), 1)

    # queued_at is when ComfyUI took it; run_since is when ComfyUI began it.
    # They differ by however long the songs ahead of it take, which is time
    # this song must not be charged for.
    queued_at = time.time()
    run_since = 0.0
    last_move = queued_at
    moved = None
    unreachable_since = 0.0
    while True:
        time.sleep(1.5)
        # Read the flag under the lock and act outside it: jobs_lock is a
        # plain Lock, so calling set_state() while holding it would wedge
        # this thread and every /api/jobs request behind it.
        with jobs_lock:
            cancelled = bool(jobs[job_id].get("cancelled"))
        if cancelled:
            client.stop(prompt_id)
            raise JobStopped()

        # One history read answers both questions, and a ComfyUI that blinks
        # — a restart, a moment of load — must not throw away a song that is
        # still sitting in its queue.
        try:
            hist = client.history(prompt_id)
        except requests.RequestException:
            unreachable_since = unreachable_since or time.time()
            if time.time() - unreachable_since > 120:
                raise JobFailed("ComfyUI stopped answering. Check the engine on "
                                "the Engine page, then make the song again.")
            set_state(stage=label + "Waiting for ComfyUI to answer")
            continue
        if unreachable_since:
            unreachable_since = 0.0
            # ComfyUI does not keep its queue over a restart, so a prompt it
            # no longer knows about is never going to finish.
            if not hist and not client.in_queue(prompt_id):
                raise JobFailed("ComfyUI restarted, and the song it was working "
                                "on went with its queue. Press Create to make "
                                "it again.")

        err = client.failed(prompt_id, hist)
        if err:
            raise JobFailed(err)

        outs = client.outputs(prompt_id, hist)
        if outs:
            return outs, run_since

        wp = ws_progress.get(prompt_id) or {}
        node = wp.get("node")
        value, maximum = wp.get("value", 0), wp.get("max", 0)

        # Any of these changing is ComfyUI saying it is still working: the
        # node it has reached, or the step it is on. Timing a song out on
        # total elapsed instead killed slow-but-healthy renders, which is
        # exactly what a big model on a small card looks like.
        fingerprint = (node, value, maximum)
        if fingerprint != moved:
            moved, last_move = fingerprint, time.time()
        if not run_since and (wp or client.is_running(prompt_id)):
            run_since = last_move = time.time()

        if not run_since:
            # Still behind other songs. Nothing is wrong and nothing is owed,
            # so no clock runs — but a prompt ComfyUI has forgotten is never
            # going to start either.
            if (time.time() - queued_at > NEVER_STARTED_LIMIT
                    and not hist and not client.in_queue(prompt_id)):
                raise JobFailed("ComfyUI no longer has this song in its queue "
                                "and never started it. Press Create to make it "
                                "again.")
            set_state(pct=scale(5), stage=label + "Waiting for the songs ahead of it",
                      elapsed=round(time.time() - queued_at))
            continue

        phase = PHASES.get(str(wp.get("progress_node") or node or ""))
        if phase and maximum and wp.get("progress_node") == node:
            stage, lo, hi = phase
            pct = lo + min(value / maximum, 1.0) * (hi - lo)
            if node == comfy.N_ENC:
                # The encoder counts frames against the ceiling it was given;
                # a song that ends earlier jumps ahead at the end.
                stage += f" — {value / 25:.0f}s of up to {maximum / 25:.0f}s"
            else:
                stage += f" — step {value} of {maximum}"
        elif str(node) in LOADERS:
            stage, pct = LOADERS[str(node)], 6
        elif phase:
            stage, pct = phase[0], phase[1]
        else:
            stage, pct = "Working", 6
        set_state(pct=scale(pct), stage=label + stage,
                  elapsed=round(time.time() - run_since))

        quiet = time.time() - last_move
        if quiet > STALL_LIMIT:
            ran = int(time.time() - run_since) // 60
            raise JobFailed(f"ComfyUI reported nothing for {int(quiet) // 60} "
                            f"minutes, after working for {ran}. The engine's "
                            "log on the Engine page says what it was doing. A "
                            "shorter Length is the usual way past a render "
                            "that will not move on a small card.")


def fetch_output(item: dict, dest: Path) -> Path:
    """Copy one finished file out of ComfyUI into data/tracks."""
    TRACKS_DIR.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    try:
        with client.view(item) as resp:
            resp.raise_for_status()
            with open(partial, "wb") as fh:
                for chunk in resp.iter_content(1024 * 256):
                    if chunk:
                        fh.write(chunk)
        if partial.stat().st_size == 0:
            raise ComfyError("ComfyUI returned an empty audio file.")
        partial.replace(dest)
    except Exception:
        # A dropped connection must not leave something that looks like a
        # playable track, or an orphan that accumulates forever.
        partial.unlink(missing_ok=True)
        raise
    return dest


def output_ext(item: dict) -> str:
    ext = Path(as_text(item.get("filename"))).suffix.lower()
    return ext if ext in (".flac", ".mp3", ".opus", ".wav", ".ogg", ".m4a") else ".flac"


# Seconds of crossfade between the parts of a long song. Two independent
# passes never line up bar for bar; a short fade hides the seam where a hard
# cut would click.
CROSSFADE = 2.0


def join_with_ffmpeg(paths: list[Path], dest: Path) -> bool:
    """One file from several, crossfaded in order. False when there is no
    ffmpeg or it could not read the parts."""
    ffmpeg = bootstrap.find_tool("ffmpeg")
    if not ffmpeg or os.environ.get("M3_STUDIO_NO_FFMPEG") == "1" or len(paths) < 2:
        return False
    shortest = min((audio_duration(p) or CROSSFADE * 4) for p in paths)
    fade = max(0.1, min(CROSSFADE, shortest / 4))
    chain, last = [], "[0:a]"
    for i in range(1, len(paths)):
        out = f"[a{i}]"
        chain.append(f"{last}[{i}:a]acrossfade=d={fade:.2f}:c1=tri:c2=tri{out}")
        last = out
    cmd = [ffmpeg, "-y", "-loglevel", "error"]
    for p in paths:
        cmd += ["-i", str(p)]
    cmd += ["-filter_complex", ";".join(chain), "-map", last, str(dest)]
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0 and dest.is_file() and dest.stat().st_size > 0


def join_in_comfy(job_id: str, paths: list[Path], dest: Path, fmt: str,
                  set_state, low: float, high: float) -> Path:
    """The parts handed to ComfyUI and concatenated there (no crossfade)."""
    names = [client.upload_path(p, p.name) for p in paths]
    built = client.build_join_prompt(names, fmt)
    prompt_id = client.queue(built["prompt"])
    outs, _ = wait_prompt(job_id, prompt_id, set_state, "Joining the parts — ",
                          low, high)
    return fetch_output(outs[0], dest.with_suffix(output_ext(outs[0])))


def run_job(job_id: str, params: dict, built=None) -> None:
    def set_state(**kw):
        with jobs_lock:
            # Note when it stopped, so the job list can keep it around long
            # enough to be read rather than measuring from when it started.
            if kw.get("status") in TERMINAL:
                kw.setdefault("ended", time.time())
            jobs[job_id].update(kw)

    wanted = (params.get("format") or "flac").lower()
    params = {**params, "format": render_format(wanted)}
    track_id = uuid.uuid4().hex[:12]
    made: list[dict] = []
    try:
        set_state(stage="Building the graph", pct=2)
        # The API normally builds this before accepting the job, so a missing
        # node or model is reported to the Create click immediately rather
        # than on an asynchronous card. Kept for callers that run a job
        # directly.
        if built is None:
            built = build_parts(params)
        elif isinstance(built, dict):
            built = [built]
        count = len(built)
        set_state(seed=built[0]["seed"], parts=count)

        # The bar: parts share 6-90, the join (if any) 90-94, saving after.
        span = (90.0 if count == 1 else 88.0) - 6.0
        for i, part in enumerate(built):
            label = f"Part {i + 1} of {count} — " if count > 1 else ""
            low, high = 6 + span * i / count, 6 + span * (i + 1) / count
            set_state(stage=label + "Queued in ComfyUI", pct=round(low, 1))
            prompt_id = client.queue(part["prompt"])
            set_state(prompt_id=prompt_id)
            try:
                outs, run_since = wait_prompt(job_id, prompt_id, set_state,
                                              label, low, high)
            except JobFailed as exc:
                raise JobFailed(explain_failure(
                    str(exc), int(part.get("duration") or 0), client.vram_total(),
                    part.get("te", ""), part.get("dit", ""))) from None
            suffix = f".p{i + 1}" if count > 1 else ""
            dest = TRACKS_DIR / f"{track_id}{suffix}{output_ext(outs[0])}"
            path = fetch_output(outs[0], dest)
            seconds = audio_duration(path)
            made.append({"index": i + 1, "seed": part["seed"],
                         "duration": part.get("duration"), "seconds": seconds,
                         "cut": ran_out(part, seconds), "path": path,
                         "run_since": run_since})

        if count > 1:
            set_state(stage="Joining the parts", pct=90)
            dest = TRACKS_DIR / f"{track_id}{made[0]['path'].suffix}"
            if join_with_ffmpeg([m["path"] for m in made], dest):
                master = dest
            else:
                master = join_in_comfy(job_id, [m["path"] for m in made], dest,
                                       params["format"], set_state, 90, 94)
            for m in made:
                m["path"].unlink(missing_ok=True)
        else:
            master = made[0]["path"]

        set_state(stage="Saving the track", pct=95)
        if wanted == "wav":
            set_state(stage="Saving as wav", pct=97)
        kept = save_as(master, track_id, wanted)
        seconds = audio_duration(kept)
        started = min((m["run_since"] for m in made if m["run_since"]),
                      default=0) or jobs[job_id]["created"]
        track = {
            "id": track_id,
            "title": track_title(params),
            "caption": params.get("caption", ""),
            # One line to show under the title, out of a caption that runs to
            # several hundred words.
            "summary": caption_genre(params.get("caption", "")),
            "lyrics": params.get("lyrics", ""),
            "instrumental": bool(params.get("instrumental")),
            "vocal": params.get("vocal", ""),
            "seed": built[0]["seed"],
            # duration is the ceiling that was asked for and is what "reuse
            # settings" loads back; seconds is how long the song actually came
            # out, and is what gets shown.
            "duration": params.get("duration"),
            "seconds": seconds,
            # A part that ran into its ceiling is missing its ending, and so
            # is the song — even when the join hides the seam.
            "cut": any(m["cut"] for m in made),
            "parts": [{k: m[k] for k in ("index", "seed", "duration", "seconds",
                                         "cut")} for m in made] if count > 1 else [],
            # Everything that would make it again (with the same seed).
            "cfg_scale": params.get("cfg_scale"),
            "top_k": params.get("top_k"),
            "steps": params.get("steps"),
            "cfg": params.get("cfg"),
            "sampler": params.get("sampler"),
            "scheduler": params.get("scheduler"),
            "dit": built[0].get("dit"),
            "te": built[0].get("te"),
            "tiled_decode": built[0].get("decode") == "VAEDecodeAudioTiled",
            "file": kept.name,
            "format": kept.suffix.lstrip(".").lower(),
            "render_seconds": round(time.time() - started),
            "created": time.time(),
        }
        add_track(track)
        set_state(status="done", pct=100, stage="Ready", track=track)
    except JobStopped:
        set_state(status="cancelled", stage="Cancelled")
    except (JobFailed, ComfyError) as exc:
        set_state(status="error", error=str(exc), stage="Failed")
    except Exception as exc:  # noqa: BLE001
        set_state(status="error", error=f"{type(exc).__name__}: {exc}",
                  stage="Failed")
    finally:
        with jobs_lock:
            ended = jobs[job_id].get("status") != "done"
        if ended:
            # Parts of a song that was stopped or failed are not a song.
            for m in made:
                m["path"].unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# routes - app shell
# --------------------------------------------------------------------------- #
@app.get("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.get("/web/<path:name>")
def web_asset(name: str):
    return send_from_directory(WEB_DIR, name)


# --------------------------------------------------------------------------- #
# routes - setup
# --------------------------------------------------------------------------- #
@app.get("/api/status")
def api_status():
    online = comfy_online(cfg["comfy_url"])
    models_dir = Path(cfg["models_dir"]) if cfg.get("models_dir") else None
    missing = []
    # Only the files a song needs can stand between someone and the Create
    # button: a loader with no file at all. The optional full-precision
    # diffusion model fills a role that int8 already fills.
    missing_required = []
    if models_dir and models_dir.is_dir():
        absent = bootstrap.missing_models(models_dir, cfg)
        missing = [Path(m[0]).name for m in absent]
        missing_required = [Path(m[0]).name for m in absent if m[3]]
    payload = {
        # Provisional until the live engine has proved that it exposes the
        # nodes and models needed to construct a generation graph.
        "ready": False,
        "comfy_online": online,
        "setup_complete": bool(cfg.get("setup_complete")),
        "missing_models": missing,
        "missing_required_models": missing_required,
        # Whether the Create page's Write button can do anything yet. Read
        # from the config alone — the writer's server is not probed here.
        "lyric_writer": lyricist.configured(cfg),
        "config": {k: cfg.get(k) for k in
                   ("comfy_url", "comfy_dir", "models_dir", "managed",
                    "auto_start_comfy", "download_fp16", "torch_index")},
        "detected": detect_comfy_dirs(),
        "comfy_running_managed": comfy_proc.alive(),
    }
    if online:
        try:
            client.ensure_supported()
            payload["dits"] = client.music3("dit")
            payload["tes"] = client.music3("te")
            payload["vaes"] = client.music3("vae")
            payload["samplers"], payload["schedulers"] = client.samplers()
            payload["formats"] = available_formats(client.save_formats())
            payload["max_duration"] = client.duration_limit()
            # Past one pass a song is made in parts and joined: with ffmpeg
            # (crossfaded) or by the engine itself (a plain concat).
            payload["can_join"] = bool(bootstrap.find_tool("ffmpeg")) or client.can_join()
            payload["max_song"] = parts.MAX_SONG
            # What the card holds, and what each model file weighs, so the
            # pickers can say which ones are heavy for this card. A file
            # larger than the card still works — ComfyUI streams it — but it
            # is slower, and the page should say so before it is picked.
            payload["vram_total"] = client.vram_total()
            payload["model_sizes"] = {
                **{n: model_size("dit", n) for n in payload["dits"]},
                **{n: model_size("te", n) for n in payload["tes"]}}
            # What Auto asks for: the node's own default, not its ceiling.
            payload["recommended_duration"] = client.duration_default()
        except Exception as exc:  # ComfyUI up but too old / still loading
            payload["schema_error"] = str(exc)
        # The two silent "nothing works" states, named rather than left for
        # the person to infer from an empty dropdown.
        #
        # ComfyUI scans its model folders once, at startup. A file that landed
        # afterwards is on disk and absent from its lists until a restart —
        # which reads as a failed download and sends people to re-fetch
        # gigabytes they already have.
        payload["stale_models"] = bool(
            not missing_required and models_dir and models_dir.is_dir()
            and engine_lacks_models())
        # And the address can be answered by a different install than the one
        # set up here, which looks exactly like a broken download.
        stats = bootstrap.comfy_stats(cfg["comfy_url"]) or {}
        payload["engine_argv"] = (stats.get("argv") or [""])[0]
        root = client.engine_root()
        payload["engine_mismatch"] = bool(
            root and cfg.get("comfy_dir")
            and not manager.same_install(cfg["comfy_dir"], root))
        payload["engine_starting"] = engine_starting()
        payload["ready"] = bool(
            cfg.get("setup_complete") and not missing_required
            and not payload["engine_starting"]
            and not payload.get("schema_error")
            and not payload.get("stale_models")
            and not payload.get("engine_mismatch"))
    return jsonify(payload)


def engine_lacks_models() -> bool:
    """True when the engine answering offers no Music 3 file to one of the
    three loaders. Which loader does not matter: any one of them empty is a
    song that cannot be built."""
    return any(not client.music3(role) for role in ("dit", "te", "vae"))


@app.post("/api/setup/start")
def api_setup_start():
    if progress.running:
        return jsonify({"ok": False, "error": "Setup is already running."}), 409
    body = json_object()
    mode = as_text(body.get("mode"), "auto")
    if mode not in ("auto", "existing", "managed", "external"):
        # Anything else used to fall through to the managed route, which
        # clones ComfyUI and installs PyTorch — not something a typo should do.
        return jsonify({"ok": False,
                        "error": f"'{mode}' is not a setup route."}), 400
    chosen = as_text(body.get("comfy_dir"))
    cfg.update(settings_from(body, ("comfy_url", "models_dir", "torch_index")
                             + FLAG_SETTINGS))
    client.url = cfg["comfy_url"]
    save_config(cfg)
    progress.__init__()  # reset log and step states
    threading.Thread(target=bootstrap.run_setup,
                     args=(cfg, progress, comfy_proc, chosen, mode),
                     daemon=True).start()
    return jsonify({"ok": True})


@app.get("/api/setup/state")
def api_setup_state():
    since = as_cursor(request.args.get("since", 0))
    snap = progress.snapshot(since)
    snap["comfy_tail"] = comfy_proc.tail(12)
    return jsonify(snap)


def _note(msg: str) -> None:
    """An engine action, said in both places it is looked for: the engine
    console (next to ComfyUI's own output) and the setup log."""
    comfy_proc.note(msg)
    progress.log(msg)


def _refresh_schema_when_up() -> None:
    """After a (re)start, throw away the cached schema the moment the engine
    answers.

    ComfyClient caches /object_info for two minutes. Without this, the whole
    point of the restart — a fresh model scan — stays hidden behind the old
    cache, and the page goes on saying the models are missing.
    """
    def wait():
        if bootstrap.wait_for_comfy(cfg["comfy_url"], timeout=900):
            try:
                client.schema(force=True)
            except Exception:
                pass
    threading.Thread(target=wait, daemon=True).start()


def take_over_port(url: str, port: int):
    """Close whatever ComfyUI answers on the port.

    Returns ("manager-reboot", None) when ComfyUI-Manager rebooted it in
    place, ("freed", None) when the port is now empty, or (None, advice) when
    it cannot be done — with advice that names the actual obstacle. "Close it
    yourself" is not advice when the thing to close is a python with no window
    and no tray icon: it sends people hunting through Task Manager.

    Nothing that does not look like ComfyUI is ever stopped: the port may be
    configured wrong, and a wrong address is not a licence to kill whatever
    is at it.
    """
    _note("This ComfyUI was not started here — taking it over.")
    try:
        r = requests.post(f"{url}/manager/reboot", json={}, timeout=5)
        accepted = r.status_code in (200, 201, 204)
    except requests.exceptions.RequestException:
        accepted = True          # the connection dropping *is* the reboot
    if accepted:
        deadline = time.time() + 10
        while time.time() < deadline:
            if not comfy_online(url):
                _note("ComfyUI-Manager took the reboot; waiting for the "
                      "engine to come back.")
                return "manager-reboot", None
            time.sleep(0.5)
        _note("ComfyUI-Manager did not take the reboot; stopping the "
              "process instead.")

    def settled_free() -> bool:
        # A supervisor — ComfyUI Desktop, a launcher script — respawns in
        # under a second, so quiet is only free once it *stays* quiet. Report
        # a port free too early and the managed engine starts into a port that
        # is taken again by the time it binds.
        time.sleep(2.0)
        return not comfy_online(url) and not bootstrap.port_pids(port)

    first_pids: list[int] = []
    denied = False
    for attempt in range(3):
        pids = bootstrap.port_pids(port)
        if attempt == 0:
            first_pids = pids
        if not pids:
            if not comfy_online(url) and settled_free():
                return "freed", None
            if not comfy_online(url):
                _note("It came straight back — something restarted it.")
                continue
            return None, (f"Something answers on port {port} but its process "
                          "could not be found — it may belong to another user "
                          "account. Close it yourself, then press Start the "
                          "engine.")
        for pid in pids:
            cmd = bootstrap.pid_cmdline(pid)
            _note(f"Port {port} is held by pid {pid}"
                  + (f": {cmd[:120]}" if cmd else " (command line unreadable)"))
            if cmd and not any(k in cmd.lower()
                               for k in ("python", "main.py", "comfy")):
                return None, (f"Port {port} is held by something that does "
                              f"not look like ComfyUI ({cmd[:90]}). Close it "
                              "yourself, or point Settings at a different "
                              "address.")
        for pid in pids:
            said = bootstrap.kill_pid(pid)
            _note(f"Stopping pid {pid} — {said or 'no reply'}")
            if "denied" in (said or "").lower():
                denied = True
        deadline = time.time() + 8
        while comfy_online(url) and time.time() < deadline:
            time.sleep(0.5)
        if not comfy_online(url):
            if settled_free():
                return "freed", None
            _note("It came straight back — something restarted it.")
            continue
        _note("Still answering — trying again.")

    now = bootstrap.port_pids(port)
    if denied:
        return None, ("The system refused to stop it (access denied) — it was "
                      "started by another user, or as an administrator. Run "
                      "M3 Studio with the same rights once, or close that "
                      "ComfyUI yourself, then press Start the engine.")
    if now and set(now) != set(first_pids):
        return None, ("It keeps coming back under a new process id — "
                      "something is supervising it (ComfyUI Desktop, or a "
                      "launcher script). Close that application, then press "
                      "Start the engine.")
    return None, ("It would not close. The Engine console shows what was "
                  "tried; close that ComfyUI yourself, then press Start the "
                  "engine.")


def _free_local_port(after: int) -> int:
    """The first port past `after` that nothing is bound to, found by binding."""
    import socket
    for port in range(after + 1, after + 21):
        try:
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", port))
            return port
        except OSError:
            continue
    return 0


def foreign_engine_reason() -> str:
    """Why the engine answering the configured address cannot be ours, or "".

    Only judged for a local address, and only when there is a managed install
    to run instead — a remote engine, or an external-mode setup, is the
    user's own and is never second-guessed. A verified-ours engine with an
    empty model list is a models problem, not a port problem: moving would
    bury the real cause, so it never moves.
    """
    if not cfg.get("comfy_dir") or not cfg.get("python"):
        return ""
    host = (urlsplit(cfg["comfy_url"]).hostname or "").lower()
    if host not in ("127.0.0.1", "localhost", "::1"):
        return ""
    root = client.engine_root()
    if root:
        if manager.same_install(cfg["comfy_dir"], root):
            return ""
        return "it is answered by the ComfyUI in " + root
    models_dir = Path(cfg["models_dir"]) if cfg.get("models_dir") else None
    if (models_dir and models_dir.is_dir()
            and not bootstrap.missing_roles(models_dir)
            and not client.music3("te", force=True)
            and not client.music3("dit")):
        return ("it is answered by an engine that does not say where it runs "
                "from and lists none of the Music 3 files, while they are on "
                "disk")
    return ""


def relocate_engine(reason: str) -> str:
    """Move to a free port, start the managed install there, keep the choice.

    This is the by-hand recovery — change the address in Settings, press
    Start the engine — done by the app itself, on every launch and on the
    button, once the port has been lost to something else. The new address is
    saved, so later launches go straight to it.
    """
    old_url = cfg["comfy_url"]
    port = _free_local_port(comfy_port(old_url))
    if not port:
        raise RuntimeError("Every port near " + old_url + " is taken — set an "
                           "address by hand in Settings.")
    cfg["comfy_url"] = f"http://127.0.0.1:{port}"
    client.url = cfg["comfy_url"]
    save_config(cfg)
    _note(f"{old_url} is not usable — {reason}. Moving to port {port} "
          "and starting the managed ComfyUI there.")
    comfy_proc.start(cfg["python"], Path(cfg["comfy_dir"]), port, progress,
                     Path(cfg["models_dir"]) if cfg.get("models_dir") else None)
    _refresh_schema_when_up()
    return cfg["comfy_url"]


def _start_managed(port: int) -> None:
    comfy_proc.start(cfg["python"], Path(cfg["comfy_dir"]), port, progress,
                     Path(cfg["models_dir"]) if cfg.get("models_dir") else None)
    _refresh_schema_when_up()


@app.post("/api/comfy/start")
def api_comfy_start():
    if comfy_online(cfg["comfy_url"]):
        reason = foreign_engine_reason()
        if not reason:
            return jsonify({"ok": True, "already": True})
        try:
            moved_to = relocate_engine(reason)
        except RuntimeError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, "moved": True, "comfy_url": moved_to,
                        "reason": reason})
    if not cfg.get("comfy_dir") or not cfg.get("python"):
        if cfg.get("comfy_dir") and not cfg.get("managed", True):
            return jsonify({"ok": False,
                            "error": "M3 Studio does not know which Python "
                                     "that ComfyUI runs on, so it will not "
                                     "start it. Start ComfyUI yourself, then "
                                     "press Recheck."}), 400
        return jsonify({"ok": False,
                        "error": "Run setup first."}), 400
    port = comfy_port(cfg["comfy_url"])
    try:
        _note("Starting ComfyUI…")
        _start_managed(port)
    except RuntimeError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    return jsonify({"ok": True})


@app.post("/api/comfy/restart")
def api_comfy_restart():
    """Stop and start ComfyUI, so it rescans its model folders — the one
    thing only a restart does.

    An engine M3 Studio did not start (an orphan from an earlier run, one
    launched by hand, a ComfyUI Desktop) is taken over rather than declared
    someone else's problem. Each route says which one it took, because "it
    restarted" means something different in each:

        managed         ours: stopped and started again
        started         nothing was there: started
        takeover        someone else's: closed, and ours put in its place
        manager-reboot  ComfyUI-Manager rebooted it in place
        stopped         closed, but this app has no ComfyUI of its own

    A refusal is a 409 whose error names the obstacle, not a shrug.
    """
    url = cfg["comfy_url"]
    port = comfy_port(url)
    can_start = bool(cfg.get("comfy_dir") and cfg.get("python"))

    if comfy_proc.alive():
        if not can_start:
            return jsonify({"ok": False, "error": "Run setup first."}), 400
        _note("Restarting the managed engine…")
        comfy_proc.stop()
        try:
            _start_managed(port)
        except RuntimeError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, "how": "managed"})

    if not comfy_online(url):
        if not can_start:
            return jsonify({"ok": False, "error": "Run setup first."}), 400
        _note("Starting ComfyUI…")
        try:
            _start_managed(port)
        except RuntimeError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, "how": "started"})

    how, advice = take_over_port(url, port)
    if advice:
        _note(advice)
        return jsonify({"ok": False, "error": advice}), 409
    if how == "manager-reboot":
        _refresh_schema_when_up()
        return jsonify({"ok": True, "how": "manager-reboot"})
    if not can_start:
        return jsonify({"ok": True, "how": "stopped",
                        "note": "Stopped it. M3 Studio has no ComfyUI of its "
                                "own to start — run setup, or start yours "
                                "again yourself."})
    _note("Starting a managed engine in its place…")
    try:
        _start_managed(port)
    except RuntimeError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    return jsonify({"ok": True, "how": "takeover"})


@app.get("/api/comfy/log")
def api_comfy_log():
    """The engine's own console: the visible cue that it is starting, up, or
    saying exactly what failed to load. Clamped, because an unbounded n is a
    way to ask the server for two thousand lines on a 2.5-second poll."""
    n = as_int(request.args.get("n"), 80, 1, 400)
    return jsonify({"lines": comfy_proc.tail(n),
                    "running": comfy_proc.alive(),
                    "online": comfy_online(cfg["comfy_url"])})


TEXT_SETTINGS = ("comfy_url", "comfy_dir", "models_dir", "torch_index")
FLAG_SETTINGS = ("auto_start_comfy", "download_fp16")


def settings_from(body: dict, keys: tuple[str, ...]) -> dict:
    """Read the settings a request is allowed to change, as their real types.

    Built in full before anything is applied: a half-applied change that then
    fails leaves the running app pointing at a value it already rejected.
    """
    change: dict = {}
    for key in keys:
        if key not in body:
            continue
        value = body[key]
        if key in FLAG_SETTINGS:
            if isinstance(value, bool):
                change[key] = value
        elif isinstance(value, str):
            change[key] = value
        # Anything else is not something the page sends. Leaving the key alone
        # keeps a setting that was working from being replaced by nonsense.
    if "comfy_url" in change:
        usable = clean_url(change["comfy_url"])
        if usable:
            change["comfy_url"] = usable
        else:
            change.pop("comfy_url")
    return change


@app.get("/api/config")
def api_config_get():
    """The saved settings, and nothing else — no probes, answers in
    milliseconds. The Settings dialog fills itself from this on open; it used
    to be filled by a status call that can take seconds, and a Save pressed
    before that returned wrote the checkboxes' blank defaults over real
    settings. That is how "start ComfyUI with M3 Studio" got switched off by
    someone changing the address."""
    return jsonify({"config": {k: cfg.get(k)
                               for k in (TEXT_SETTINGS + FLAG_SETTINGS)}})


@app.post("/api/config")
def api_config():
    body = json_object()
    cfg.update(settings_from(body, TEXT_SETTINGS + FLAG_SETTINGS))
    client.url = cfg["comfy_url"]
    save_config(cfg)
    # The same view GET gives. Answering with the whole config handed the
    # saved HuggingFace token and the lyric writer's API key to the page.
    return jsonify({"ok": True, "config": {k: cfg.get(k) for k in
                                           (TEXT_SETTINGS + FLAG_SETTINGS)}})


# --------------------------------------------------------------------------- #
# routes - generation
# --------------------------------------------------------------------------- #
@app.post("/api/generate")
def api_generate():
    params = clean_generate(json_object())
    if not params["caption"].strip():
        return jsonify({"error": "Describe the music in the caption before "
                                 "generating."}), 400
    if not comfy_online(cfg["comfy_url"]):
        return jsonify({"error": "ComfyUI is not running. Start it from "
                                 "Settings."}), 503

    count = params["count"]
    # Validate and construct every graph before creating job records. This is
    # both a readiness probe and an all-or-nothing batch: an outdated ComfyUI,
    # stale model list, or invalid selected option is returned directly and
    # cannot leave several doomed jobs behind in the queue UI.
    render_params = {**params,
                     "format": render_format(
                         (params.get("format") or "flac").lower())}
    seed = params["seed"]
    try:
        # A fixed seed with ×3 makes three takes on consecutive seeds, so the
        # set can be made again exactly — not three copies of one song.
        built_prompts = [build_parts(
            {**render_params, "seed": None if seed is None else seed + i})
            for i in range(count)]
    except ComfyError as exc:
        return jsonify({"error": str(exc)}), 409
    except requests.RequestException:
        return jsonify({"error": "ComfyUI stopped answering while M3 Studio "
                                 "checked whether it was ready. Try again after "
                                 "the Engine page says ready."}), 503

    created = []
    with jobs_lock:
        stale = [k for k, j in jobs.items()
                 if j["status"] != "running"
                 and time.time() - j.get("ended", j["created"]) > 3600]
        for k in stale:
            jobs.pop(k, None)
    for built in built_prompts:
        job_id = uuid.uuid4().hex[:12]
        with jobs_lock:
            jobs[job_id] = {"id": job_id, "status": "running", "pct": 0,
                            "stage": "Starting", "created": time.time(),
                            "title": track_title(params),
                            "summary": caption_genre(params.get("caption", "")),
                            "parts": len(built)}
        threading.Thread(target=run_job, args=(job_id, dict(params), built),
                         daemon=True).start()
        created.append(job_id)
        time.sleep(0.2)  # keep queue order stable
    return jsonify({"jobs": created})


@app.get("/api/jobs")
def api_jobs():
    """Everything still going, plus whatever stopped in the last two minutes.

    Measured from when a job ended, not from when it began. A song takes
    minutes, so keying this on its start time dropped every real failure out of
    the list the moment it failed — the card vanished and the reason with it.
    """
    cutoff = time.time() - 120
    with jobs_lock:
        active = [j for j in jobs.values()
                  if j["status"] == "running"
                  or j.get("ended", j["created"]) > cutoff]
        return jsonify(sorted(active, key=lambda j: j["created"], reverse=True))


@app.post("/api/jobs/<job_id>/cancel")
def api_cancel(job_id: str):
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id]["cancelled"] = True
    return jsonify({"ok": True})


# --------------------------------------------------------------------------- #
# routes - library
# --------------------------------------------------------------------------- #
@app.get("/api/library")
def api_library():
    return jsonify(read_library())


@app.get("/api/track/<track_id>")
def api_track(track_id: str):
    for item in read_library():
        if item["id"] == track_id:
            path = track_file(item)
            if path is None or not path.is_file():
                return jsonify({"error": "Audio file is missing."}), 404
            mime = mimetypes.guess_type(path.name)[0] or "audio/flac"
            return send_file(path, mimetype=mime, conditional=True,
                             download_name=f"{item.get('title') or 'track'}{path.suffix}")
    return jsonify({"error": "Track not found."}), 404


@app.patch("/api/track/<track_id>")
def api_rename(track_id: str):
    body = json_object()
    with library_lock:
        items = read_library()
        for item in items:
            if item["id"] == track_id:
                title = as_text(body.get("title")).strip()
                if title:
                    item["title"] = title[:120]
                if body.get("seconds") is not None:
                    # Sent by the player once the browser has decoded the file,
                    # for formats audio_duration() could not read without
                    # ffprobe.
                    try:
                        seconds = round(float(body["seconds"]), 2)
                    except (TypeError, ValueError):
                        seconds = 0.0
                    if 0 < seconds < 7200:
                        item["seconds"] = seconds
                write_library(items)
                return jsonify({"ok": True, "track": item})
    return jsonify({"error": "Track not found."}), 404


@app.delete("/api/track/<track_id>")
def api_delete(track_id: str):
    with library_lock:
        items = read_library()
        keep = [i for i in items if i["id"] != track_id]
        gone = [i for i in items if i["id"] == track_id]
        write_library(keep)
    # The file goes after the list is settled — deleting it first would leave a
    # library entry pointing at nothing if the write failed.
    for item in gone:
        for path in track_files(item):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
    return jsonify({"ok": True})


# --------------------------------------------------------------------------- #
# routes - dependencies
# --------------------------------------------------------------------------- #
@app.get("/api/deps")
def api_deps():
    # What ComfyUI itself offers, so the model row can tell a file that has not
    # been downloaded apart from one the engine cannot see. None means it could
    # not be asked, which is not the same as an empty list.
    online = comfy_online(cfg["comfy_url"])
    listed = (client.music3("dit") + client.music3("te")) if online else None
    engine_root = client.engine_root() if online else ""
    fresh = request.args.get("fresh") == "1"
    if not locating.is_set() and not _search_off():
        _heal()
        if _needs_search() and _rested():
            # Recheck with ComfyUI still nowhere: search the drives in the
            # background — the row says "searching" and the page polls.
            locating.set()
            threading.Thread(target=_heal, args=(True,), daemon=True).start()
    searching = locating.is_set()
    return jsonify({"items": manager.dependencies(cfg, listed, engine_root,
                                                  fresh, engine_note(),
                                                  searching=searching),
                    "searching": searching,
                    "os": __import__("platform").system(),
                    "torch_index": cfg.get("torch_index", "")})


@app.post("/api/deps/<dep_id>/install")
def api_dep_install(dep_id: str):
    body = json_object()
    if dep_id not in manager.INSTALLABLE:
        return jsonify({"error": f"There is nothing called '{dep_id}' to "
                                 "install."}), 400
    if body.get("torch_index") is not None:
        cfg["torch_index"] = as_text(body["torch_index"])
        save_config(cfg)
    try:
        task = manager.install_dependency(dep_id, cfg, body)
        return jsonify({"ok": True, "task": task.view()})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400


@app.get("/api/tasks")
def api_tasks():
    since = as_cursor(request.args.get("since", 0))
    task_id = request.args.get("id", "")
    if task_id:
        task = manager.TASKS.get(task_id)
        if not task:
            return jsonify({"error": "No such task."}), 404
        return jsonify(task.view(since))
    return jsonify([t.view(t.view()["cursor"]) for t in manager.TASKS.list()[:25]])


@app.post("/api/tasks/<task_id>/cancel")
def api_task_cancel(task_id: str):
    task = manager.TASKS.get(task_id)
    if task:
        task.cancel = True
    return jsonify({"ok": True})


# --------------------------------------------------------------------------- #
# routes - song writer
#
# Optional. A chat model the person already has writes a structured caption,
# lyrics and a title into the Create page. The key is kept like the HuggingFace token:
# saved in the config, returned masked, and bound to the address it was
# entered for.
# --------------------------------------------------------------------------- #
@app.get("/api/llm/settings")
def api_llm_settings():
    return jsonify(lyricist.settings_view(cfg))


@app.post("/api/llm/settings")
def api_llm_settings_save():
    try:
        lyricist.apply_settings(cfg, json_object())
    except lyricist.LyricistError as exc:
        return jsonify({"error": str(exc)}), 400
    save_config(cfg)
    return jsonify({"ok": True, **lyricist.settings_view(cfg)})


@app.get("/api/llm/models")
def api_llm_models():
    try:
        return jsonify({"models": lyricist.list_models(cfg)})
    except lyricist.LyricistError as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/api/lyrics/write")
def api_lyrics_write():
    """Start a write. The answer arrives on the task: poll /api/tasks?id=."""
    ask = lyricist.clean_request(json_object())
    try:
        conn = lyricist.connection(cfg)
    except lyricist.LyricistError as exc:
        return jsonify({"error": str(exc)}), 409
    if ask["want"] == "polish" and not ask["lyrics"]:
        return jsonify({"error": "There are no lyrics to polish yet."}), 400
    if ask["want"] == "caption" and not (ask["brief"] or ask["lyrics"]
                                         or ask["caption"]):
        return jsonify({"error": "Say in a line how the song should sound, "
                                 "or write some lyrics first."}), 400
    # One at a time: a second press while the first is still streaming would
    # pay for two answers and keep only one of them.
    if manager.TASKS.running("lyrics"):
        return jsonify({"error": "Already writing. Stop that one first."}), 409
    task = manager.spawn("lyrics", "Writing the song",
                         lambda t: lyricist.released(lyricist.run)(t, conn, ask),
                         meta={"want": ask["want"]})
    return jsonify({"ok": True, "task": task.view()})


# --------------------------------------------------------------------------- #
# routes - huggingface
# --------------------------------------------------------------------------- #
@app.get("/api/hf/settings")
def api_hf_settings():
    token = cfg.get("hf_token") or ""
    return jsonify({
        "endpoint": cfg.get("hf_endpoint") or manager.DEFAULT_ENDPOINT,
        "token_set": bool(token),
        "token_hint": ("…" + token[-4:]) if len(token) > 4 else "",
        "repo": cfg.get("hf_repo") or manager.DEFAULT_REPO,
        "folders": manager.MODEL_FOLDERS,
        "curated": manager.CURATED,
        "models_dir": cfg.get("models_dir", ""),
    })


@app.post("/api/hf/settings")
def api_hf_settings_save():
    body = json_object()
    if "token" in body:
        cfg["hf_token"] = as_text(body["token"]).strip()
    if body.get("endpoint") is not None:
        cfg["hf_endpoint"] = (as_text(body["endpoint"]).strip()
                              or manager.DEFAULT_ENDPOINT)
    if body.get("repo"):
        cfg["hf_repo"] = as_text(body["repo"]).strip() or manager.DEFAULT_REPO
    if body.get("models_dir"):
        cfg["models_dir"] = as_text(body["models_dir"]).strip()
    save_config(cfg)
    return jsonify({"ok": True})


@app.get("/api/hf/browse")
def api_hf_browse():
    repo = (request.args.get("repo") or cfg.get("hf_repo")
            or manager.DEFAULT_REPO).strip()
    revision = request.args.get("revision") or "main"
    try:
        data = manager.hf_list(cfg, repo, revision)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400
    have = {(m["folder"], m["name"]) for m in manager.local_models(cfg)}
    for f in data["files"]:
        folder = manager.guess_folder(f["path"])
        f["folder"] = folder
        f["installed"] = (folder, Path(f["path"]).name) in have
    cfg["hf_repo"] = repo
    save_config(cfg)
    return jsonify(data)


@app.post("/api/hf/download")
def api_hf_download():
    body = json_object()
    repo = (as_text(body.get("repo")) or cfg.get("hf_repo")
            or manager.DEFAULT_REPO).strip()
    path = as_text(body.get("path")).strip()
    if not path:
        return jsonify({"error": "Pick a file to download."}), 400
    try:
        task = manager.hf_download(cfg, repo, path,
                                   as_text(body.get("folder")),
                                   as_text(body.get("revision")) or "main")
        return jsonify({"ok": True, "task": task.view()})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400


@app.get("/api/hf/local")
def api_hf_local():
    return jsonify({"models": manager.local_models(cfg),
                    "models_dir": cfg.get("models_dir", "")})


@app.delete("/api/hf/local")
def api_hf_delete():
    body = json_object()
    try:
        manager.delete_model(cfg, body.get("folder", ""), body.get("name", ""))
        return jsonify({"ok": True})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400


# --------------------------------------------------------------------------- #
# How long to keep withholding "ready" from an engine that has not announced
# itself. A build that prints a banner we do not recognise must not be held
# back for ever — past this, answering its port is taken as good enough.
STARTING_GRACE = 180.0


def engine_starting() -> bool:
    """True while the engine M3 Studio started is still putting itself up.

    Answering /system_stats is not the same as being able to render. The page
    called an engine ready the moment the port replied, so a first launch
    showed "Engine ready" while ComfyUI was still loading — and a song made in
    that window fails inside the nodes for reasons that have nothing to do
    with the song. ComfyUI says when it is really serving; until it does, and
    only for a process of ours, this is the honest answer.
    """
    return bool(comfy_proc.alive() and not comfy_proc.serving
                and comfy_proc.started_at
                and time.time() - comfy_proc.started_at < STARTING_GRACE)


def engine_note() -> str:
    """Why the address is silent, when M3 Studio started the engine itself.

    A managed engine that exited is a different problem from one that was
    never started, and the Engine row cannot tell them apart on its own: both
    are simply nothing answering the port. Saying which one it is turns a
    dead end into the one place worth looking — the console above it, which
    now ends with what the engine said on its way out.
    """
    if engine_starting():
        return ("ComfyUI is starting — it has not finished loading yet. "
                "The Engine console below shows how far it has got.")
    if comfy_proc.alive():
        return "ComfyUI was started and is still loading — give it a moment."
    code = comfy_proc.exit_code
    if code is None:
        return ""
    return (f"ComfyUI started, then stopped on its own (exit code {code}). "
            "The Engine console below ends with the reason.")


def engine_trouble() -> list[str]:
    """Why the engine already answering is no use as it stands, or [].

    Only the problems a restart actually cures. A ComfyUI too old to have the
    Music 3 nodes is deliberately not one of them: a restart cannot conjure
    nodes, and treating their absence as a reason to replace the engine
    would kill a working one once per launch and fix nothing. That one is
    reported on the Engine page, with Update beside it.
    """
    reasons = []
    try:
        client.schema(force=True)
    except Exception as exc:  # noqa: BLE001
        _note(f"The engine already running would not describe itself "
              f"({exc}) — leaving it alone.")
        return []
    if "MiniMaxMusic3TextEncode" not in client.schema():
        _note("The engine already running has no MiniMax Music 3 nodes — "
              "update ComfyUI from the Engine page.")
        return []
    models_dir = Path(cfg["models_dir"]) if cfg.get("models_dir") else None
    weights_here = bool(models_dir and models_dir.is_dir()
                        and not bootstrap.missing_roles(models_dir))
    if weights_here and engine_lacks_models():
        reasons.append("it started before the model files landed, so it has "
                       "not scanned them")
    return reasons


def ensure_engine_at_boot() -> None:
    """A launch ends with a working engine, without a button being pressed.

    Offline: start the managed one. Online and provably a *different*
    install: step aside onto a free port — M3 Studio's own gentler cure,
    which leaves the other engine running (see relocate_engine). Online, at
    our own address, and useless — a startup scan that ran before the weights
    landed — restart it in place, through the same looks-like-ComfyUI guard
    the Restart button uses; stepping aside there would leave the stale engine
    holding the card and fix nothing. Online and healthy: adopt it, and say
    so, because silence at that point reads as a failure to start.

    An external-mode setup (`managed` false) is the person's own ComfyUI: it
    is told what is wrong and left alone.
    """
    if not (cfg.get("setup_complete") and cfg.get("auto_start_comfy", True)):
        return
    if not (cfg.get("comfy_dir") and cfg.get("python")):
        return
    url = cfg["comfy_url"]
    port = comfy_port(url)
    try:
        if not comfy_online(url):
            _note("Starting ComfyUI from the last setup…")
            _start_managed(port)
            return

        # Something answers — but a launch must not take that on faith.
        reason = foreign_engine_reason()
        if reason:
            relocate_engine(reason)
            return

        trouble = engine_trouble()
        if not trouble:
            _note(f"Adopting the ComfyUI already running at {url}.")
            return
        if not cfg.get("managed", True):
            _note("The engine already running has problems ("
                  + "; ".join(trouble) + ") but it is yours, not M3 Studio's "
                  "— restart it yourself, or press Restart the engine.")
            return
        _note("The engine already running is no use as it stands — "
              + "; ".join(trouble) + ". Replacing it.")
        how, advice = take_over_port(url, port)
        if advice:
            _note(advice)
            return
        if how == "manager-reboot":
            _refresh_schema_when_up()
            return
        _note("Starting a managed engine in its place…")
        _start_managed(port)
    except RuntimeError as exc:
        # The app still comes up; the Engine page explains the rest.
        _note(str(exc))


def boot() -> None:
    """Verify every saved location — searching the drives if ComfyUI is
    lost — then bring the engine up."""
    _heal(search=True)
    ensure_engine_at_boot()


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TRACKS_DIR.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=ws_listener, daemon=True).start()
    # The engine comes up on its own; the page can open while it does.
    threading.Thread(target=boot, daemon=True).start()

    url = f"http://127.0.0.1:{PORT}"
    print(f"\n  M3 Studio  →  {url}\n")
    if os.environ.get("M3_STUDIO_NO_BROWSER") != "1":
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    try:
        app.run(host="127.0.0.1", port=PORT, threaded=True, debug=False)
    finally:
        comfy_proc.stop()


if __name__ == "__main__":
    main()
