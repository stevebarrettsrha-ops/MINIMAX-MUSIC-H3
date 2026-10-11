"""
comfy.py - talks to a running ComfyUI instance.

The generation graph is built from ComfyUI's own /object_info schema rather than
from a hard-coded API workflow. Node authors rename and add inputs between
releases; reading the schema means a rename shows up as "input not found"
instead of a silent wrong-value bug, and new required inputs get sensible
defaults automatically.

Graph (same shape as ComfyUI's "Text to Music (MiniMax Music 3)" blueprint):

    CLIPLoader (type minimax) ─ CLIP ─► MiniMaxMusic3TextEncode ─┬─ CONDITIONING ─► KSampler (+)
                                         caption, lyrics, seed   ├─ ConditioningZeroOut ─► KSampler (−)
                                                                 └─ seconds ─► EmptyMiniMaxMusic3LatentAudio
    UNETLoader ─ MODEL ─────────────────────────────────────────────────────► KSampler
    EmptyMiniMaxMusic3LatentAudio ─ LATENT ──────────────────────────────────► KSampler
    KSampler ─ LATENT ─► VAEDecodeAudioTiled ◄─ VAE ─ VAELoader
                         └─ AUDIO ─► SaveAudioAdvanced

The text encoder is where the song is written: an 8B language model generates
the whole performance frame by frame (25 a second) and stops when the song
ends, so `max_duration` is a ceiling, not a length. Its `seconds` output sizes
the latent the diffusion model then renders into audio.
"""

from __future__ import annotations

import json
import random
import threading
import time
import uuid

import requests

# Every MiniMax Music 3 file carries this in its name (bootstrap.MARKER).
MARKER = "music3"

# The official blueprint's values. The node's own cfg_scale default is 1.5;
# the published workflow uses 1.7 for both passes, and so do we.
DEFAULTS = {"steps": 30, "cfg": 1.7, "cfg_scale": 1.7, "top_k": 50,
            "sampler": "euler", "scheduler": "simple",
            "tile_size": 1536, "overlap": 64, "duration": 120}

# Node ids, kept equal to the blueprint's so a prompt can be read side by side
# with it, and so the progress listener can tell the two long phases apart.
N_DIT, N_TE, N_VAE, N_ENC, N_ZERO, N_LATENT, N_SAMPLER, N_DECODE, N_SAVE = (
    "6", "3", "7", "13", "10", "15", "9", "12", "16")


class ComfyError(RuntimeError):
    pass


class ComfyClient:
    def __init__(self, url: str = "http://127.0.0.1:8188") -> None:
        self.url = url.rstrip("/")
        self.client_id = str(uuid.uuid4())
        self._schema: dict | None = None
        self._schema_at = 0.0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # schema
    # ------------------------------------------------------------------ #
    def schema(self, force: bool = False) -> dict:
        with self._lock:
            if force or self._schema is None or time.time() - self._schema_at > 120:
                r = requests.get(f"{self.url}/object_info", timeout=30)
                r.raise_for_status()
                self._schema = r.json()
                self._schema_at = time.time()
            return self._schema

    def node_inputs(self, class_type: str, force: bool = False) -> dict:
        info = self.schema(force=force).get(class_type)
        if not info:
            raise ComfyError(
                f"This ComfyUI has no '{class_type}' node. MiniMax Music 3 needs "
                "a recent ComfyUI (one with the MiniMaxMusic3TextEncode node) — "
                "update ComfyUI from the Engine page and restart it.")
        spec = info.get("input", {})
        merged = {}
        merged.update(spec.get("required", {}) or {})
        merged.update(spec.get("optional", {}) or {})
        return merged

    # Everything a default Create click uses. The status endpoint must not
    # advertise "Engine ready" when the ordinary graph cannot be constructed.
    REQUIRED_NODES = ("UNETLoader", "CLIPLoader", "VAELoader", "KSampler",
                      "MiniMaxMusic3TextEncode", "EmptyMiniMaxMusic3LatentAudio",
                      "ConditioningZeroOut")
    # The graph needs one of each of these, in order of preference — not a
    # particular one. Requiring the tiled decoder would call a ComfyUI that
    # renders songs perfectly well unable to run at all.
    DECODE_NODES = ("VAEDecodeAudioTiled", "VAEDecodeAudio")
    SAVE_NODES = ("SaveAudioAdvanced", "SaveAudio")

    def ensure_supported(self) -> None:
        """Fail with the real cause before anything else can mask it."""
        schema = self.schema()
        missing = [n for n in self.REQUIRED_NODES if n not in schema]
        for group in (self.DECODE_NODES, self.SAVE_NODES):
            if not any(n in schema for n in group):
                missing.append(" or ".join(group))
        if missing:
            raise ComfyError(
                "This ComfyUI cannot run MiniMax Music 3 — it is missing "
                + ", ".join(missing)
                + ". Update ComfyUI (Engine page → ComfyUI → Update), then "
                  "restart the engine.")

    @staticmethod
    def combo_options(spec) -> list[str]:
        """Read a dropdown's choices out of an /object_info input spec.

        ComfyUI writes combos two ways depending on how the node was declared:

            V1 (nodes.py)     [["a.safetensors", "b.safetensors"], {...}]
            V3 (comfy_api)    ["COMBO", {"options": ["a.safetensors", ...]}]

        The loaders are V1 and the Music 3 and audio nodes are V3, so both
        shapes turn up in the same graph. Reading only the first would
        silently report "no models installed" for half of it.
        """
        if not isinstance(spec, (list, tuple)) or not spec:
            return []
        if isinstance(spec[0], (list, tuple)):
            options = spec[0]
        else:
            opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
            options = opts.get("options")
        if not isinstance(options, (list, tuple)):
            return []
        # A dynamic combo also keys off "options", but its entries are dicts;
        # those are not plain choices and belong to _dynamic_options().
        return [str(o) for o in options if isinstance(o, (str, int, float))]

    @staticmethod
    def _opts(spec) -> dict:
        return (spec[1] if isinstance(spec, (list, tuple)) and len(spec) > 1
                and isinstance(spec[1], dict) else {})

    # ------------------------------------------------------------------ #
    # model lists
    # ------------------------------------------------------------------ #
    LOADERS = {"dit": ("UNETLoader", "unet_name"),
               "te": ("CLIPLoader", "clip_name"),
               "vae": ("VAELoader", "vae_name")}

    def listed(self, role: str, force: bool = False) -> list[str]:
        """Every file the engine offers to the loader for `role`."""
        node, field = self.LOADERS[role]
        try:
            return self.combo_options(self.node_inputs(node, force=force).get(field))
        except Exception:
            return []

    def music3(self, role: str, force: bool = False) -> list[str]:
        """The Music 3 files among them."""
        return [n for n in self.listed(role, force) if MARKER in n.lower()]

    def all_music3(self) -> list[str]:
        out: list[str] = []
        for role in self.LOADERS:
            out += self.music3(role)
        return out

    @staticmethod
    def _best(names: list[str], preferred: str = "") -> str | None:
        """The chosen file when it is there; otherwise int8, which fits on far
        more cards; otherwise whichever Music 3 file there is."""
        if preferred and preferred in names:
            return preferred
        int8 = [n for n in names if "int8" in n.lower()]
        return (int8 or names or [None])[0]

    def pick(self, role: str, preferred: str = "") -> str:
        chosen = self._best(self.music3(role), preferred)
        if chosen:
            return chosen
        # An empty list is not proof the file is missing. ComfyUI scans its
        # model folders once, at startup, and this client caches /object_info
        # on top of that — so ask again with both caches bypassed before
        # blaming the download.
        chosen = self._best(self.music3(role, force=True), preferred)
        if chosen:
            return chosen
        node, _ = self.LOADERS[role]
        what = {"dit": "diffusion model (models/diffusion_models)",
                "te": "text encoder (models/text_encoders)",
                "vae": "audio VAE (models/vae)"}[role]
        raise ComfyError(
            f"ComfyUI lists no MiniMax Music 3 {what} in {node}. If the "
            "Models page says the file is downloaded, ComfyUI started before "
            "it landed: press Restart the engine on the Engine page, which "
            "makes it rescan its model folders.")

    # ------------------------------------------------------------------ #
    # what the engine says about itself
    # ------------------------------------------------------------------ #
    def _system_stats(self) -> dict:
        try:
            r = requests.get(f"{self.url}/system_stats", timeout=10)
            r.raise_for_status()
            return r.json() or {}
        except Exception:
            return {}

    def engine_root(self) -> str:
        """The folder of the ComfyUI actually answering this address.

        /system_stats reports the process's own argv, which starts with the
        main.py it was launched from. Comparing that against the install being
        managed is the only way to notice that the port is held by a *different*
        ComfyUI — which looks exactly like a broken model download.

        Empty when it cannot be told: older builds do not report argv.
        """
        argv = (self._system_stats().get("system") or {}).get("argv") or []
        for arg in argv:
            if isinstance(arg, str) and arg.lower().endswith("main.py"):
                # Both separators appear: a Windows path read on any platform.
                cut = max(arg.rfind("/"), arg.rfind("\\"))
                return arg[:cut] if cut > 0 else ""
        return ""

    def vram_total(self) -> int:
        """Bytes of video memory on the card ComfyUI is using, or 0 if unknown."""
        best = 0
        devices = self._system_stats().get("devices") or []
        for dev in devices:
            if not isinstance(dev, dict):
                continue
            if dev.get("type") in ("cpu", "mps") and len(devices) > 1:
                continue            # a real accelerator is listed too
            try:
                best = max(best, int(dev.get("vram_total") or 0))
            except (TypeError, ValueError):
                continue
        return best

    def samplers(self) -> tuple[list[str], list[str]]:
        spec = self.node_inputs("KSampler")
        return (self.combo_options(spec.get("sampler_name")),
                self.combo_options(spec.get("scheduler")))

    def _duration_spec(self) -> dict:
        try:
            return self._opts(self.node_inputs("MiniMaxMusic3TextEncode")
                              .get("max_duration"))
        except ComfyError:
            return {}

    def duration_limit(self) -> int:
        """The longest song this engine's Music 3 will write (9000 frames at
        25 a second is 360 s), read from the schema rather than assumed."""
        try:
            return int(float(self._duration_spec().get("max") or 0)) or 360
        except (TypeError, ValueError):
            return 360

    def duration_default(self) -> int:
        """The node's own default, distinct from its hard limit."""
        try:
            value = int(float(self._duration_spec().get("default") or 0))
        except (TypeError, ValueError):
            value = 0
        return max(1, min(value, self.duration_limit())) if value else DEFAULTS["duration"]

    def pick_option(self, spec, *preferred: str) -> str | None:
        """The first preferred choice this schema actually offers.

        Sampler and scheduler lists differ between ComfyUI builds. Sending a
        name the node has never heard of is rejected at queue time, so fall
        back to the input's own default rather than insisting on ours.
        """
        options = self.combo_options(spec)
        for want in preferred:
            if want and want in options:
                return want
        if not options:
            return next((w for w in preferred if w), None)
        default = self._opts(spec).get("default")
        return default if default in options else options[0]

    # ------------------------------------------------------------------ #
    # SaveAudioAdvanced.format is a dynamic combo: choosing "mp3" or "opus"
    # makes ComfyUI expect a sibling "quality" input that does not exist for
    # "flac". Both the option list and that extra input live inside the spec.
    # ------------------------------------------------------------------ #
    @staticmethod
    def _dynamic_options(spec) -> list[dict]:
        if not isinstance(spec, (list, tuple)) or len(spec) < 2:
            return []
        options = (spec[1] or {}).get("options")
        return [o for o in options if isinstance(o, dict)] if isinstance(options, list) else []

    def save_node(self) -> str:
        schema = self.schema()
        return next((n for n in self.SAVE_NODES if n in schema), self.SAVE_NODES[0])

    def save_formats(self) -> list[str]:
        if self.save_node() != "SaveAudioAdvanced":
            return ["flac"]          # SaveAudio writes flac and nothing else
        try:
            spec = self.node_inputs("SaveAudioAdvanced").get("format")
        except ComfyError:
            return ["flac"]
        keys = [str(o["key"]) for o in self._dynamic_options(spec) if o.get("key")]
        return keys or self.combo_options(spec) or ["flac"]

    def _format_extras(self, spec, chosen: str) -> dict:
        """Defaults for the inputs that the chosen format pulls in (quality)."""
        extras: dict = {}
        for option in self._dynamic_options(spec):
            if str(option.get("key")) != chosen:
                continue
            nested = option.get("inputs") or {}
            for group in ("required", "optional"):
                for name, definition in (nested.get(group) or {}).items():
                    opts = self._opts(definition)
                    choices = self.combo_options(definition)
                    if "default" in opts:
                        extras[name] = opts["default"]
                    elif choices:
                        extras[name] = choices[0]
        return extras

    # ------------------------------------------------------------------ #
    # graph construction
    # ------------------------------------------------------------------ #
    @staticmethod
    def _match(available: dict, candidates: list[str]) -> str | None:
        for c in candidates:
            if c in available:
                return c
        low = {k.lower(): k for k in available}
        for c in candidates:
            if c.lower() in low:
                return low[c.lower()]
        return None

    def _node(self, class_type: str, wanted: dict) -> dict:
        """Build one API node: requested values by name + schema defaults."""
        spec = self.node_inputs(class_type)
        inputs: dict = {}

        for key, candidates in wanted.items():
            value = candidates["value"]
            name = self._match(spec, candidates["names"])
            if name is None:
                if candidates.get("required"):
                    raise ComfyError(
                        f"{class_type} has no input for '{key}'. This ComfyUI "
                        f"version is not compatible with the M3 Studio graph — "
                        f"update ComfyUI.")
                continue
            inputs[name] = value

        # Fill remaining scalar inputs with their schema defaults so the prompt
        # validates even if a node gained a new required field.
        for name, definition in spec.items():
            if name in inputs or name == "control_after_generate":
                continue
            if not isinstance(definition, (list, tuple)) or not definition:
                continue
            kind, opts = definition[0], self._opts(definition)
            choices = self.combo_options(definition)
            if choices:                          # enum, either schema version
                inputs[name] = opts.get("default", choices[0])
            elif kind in ("INT", "FLOAT", "STRING", "BOOLEAN"):
                if "default" in opts:
                    inputs[name] = opts["default"]
                elif kind == "STRING":
                    inputs[name] = ""
            # link types (MODEL/CLIP/VAE/LATENT/AUDIO/...) are left alone; if one
            # is required and unlinked ComfyUI reports it clearly.
        return {"class_type": class_type, "inputs": inputs}

    def build_prompt(self, p: dict) -> dict:
        """p: caption, lyrics, instrumental, duration, seed, cfg_scale, top_k,
        steps, cfg, sampler, scheduler, dit, te, tiled_decode, tile_size,
        overlap, format, quality."""
        self.ensure_supported()
        seed = p.get("seed")
        seed = int(seed) if seed not in (None, "") else random.randint(0, 2**31 - 1)
        caption = (p.get("caption") or "").strip()
        lyrics = song_lyrics(p.get("lyrics") or "", bool(p.get("instrumental")))
        # A ceiling: the model ends the song when the song ends, so asking for
        # longer never pads a short song — it only stops a long one being cut.
        duration = max(1, min(int(p.get("duration") or DEFAULTS["duration"]),
                              self.duration_limit()))
        dit = self.pick("dit", p.get("dit") or "")
        te = self.pick("te", p.get("te") or "")
        vae = self.pick("vae", p.get("vae") or "")

        g: dict = {}
        g[N_DIT] = self._node("UNETLoader", {
            "unet": {"names": ["unet_name"], "value": dit, "required": True},
            "dtype": {"names": ["weight_dtype"], "value": "default"},
        })
        clip_spec = self.node_inputs("CLIPLoader")
        g[N_TE] = self._node("CLIPLoader", {
            "clip": {"names": ["clip_name"], "value": te, "required": True},
            # "minimax" is the CLIP type that loads the Music 3 encoder (and,
            # for a different file, MiniMax H3's). Read from the schema so a
            # build that renames it fails here, by name.
            "type": {"names": ["type"], "required": True,
                     "value": self._clip_type(clip_spec.get("type"))},
        })
        g[N_VAE] = self._node("VAELoader", {
            "vae": {"names": ["vae_name"], "value": vae, "required": True},
        })
        g[N_ENC] = self._node("MiniMaxMusic3TextEncode", {
            "clip": {"names": ["clip"], "value": [N_TE, 0], "required": True},
            "caption": {"names": ["caption", "tags", "style"], "value": caption,
                        "required": True},
            "lyrics": {"names": ["lyrics", "lyric"], "value": lyrics,
                       "required": True},
            "seed": {"names": ["seed"], "value": seed},
            "duration": {"names": ["max_duration", "duration", "seconds"],
                         "value": float(duration)},
            "cfg_scale": {"names": ["cfg_scale"],
                          "value": float(p.get("cfg_scale") or DEFAULTS["cfg_scale"])},
            "top_k": {"names": ["top_k"],
                      "value": int(p.get("top_k") or DEFAULTS["top_k"])},
        })
        g[N_ZERO] = self._node("ConditioningZeroOut", {
            "cond": {"names": ["conditioning"], "value": [N_ENC, 0],
                     "required": True},
        })
        g[N_LATENT] = self._node("EmptyMiniMaxMusic3LatentAudio", {
            "seconds": {"names": ["seconds"], "value": [N_ENC, 1],
                        "required": True},
            "batch": {"names": ["batch_size"], "value": 1},
        })
        ks_spec = self.node_inputs("KSampler")
        g[N_SAMPLER] = self._node("KSampler", {
            "model": {"names": ["model"], "value": [N_DIT, 0], "required": True},
            "positive": {"names": ["positive"], "value": [N_ENC, 0],
                         "required": True},
            "negative": {"names": ["negative"], "value": [N_ZERO, 0],
                         "required": True},
            "latent": {"names": ["latent_image"], "value": [N_LATENT, 0],
                       "required": True},
            "seed": {"names": ["seed", "noise_seed"], "value": seed},
            "steps": {"names": ["steps"],
                      "value": int(p.get("steps") or DEFAULTS["steps"])},
            "cfg": {"names": ["cfg"],
                    "value": float(p.get("cfg") or DEFAULTS["cfg"])},
            "sampler": {"names": ["sampler_name"],
                        "value": self.pick_option(ks_spec.get("sampler_name"),
                                                  p.get("sampler") or "",
                                                  DEFAULTS["sampler"])},
            "scheduler": {"names": ["scheduler"],
                          "value": self.pick_option(ks_spec.get("scheduler"),
                                                    p.get("scheduler") or "",
                                                    DEFAULTS["scheduler"])},
            "denoise": {"names": ["denoise"], "value": 1.0},
        })

        # Preferred first, then whatever this ComfyUI actually has.
        schema = self.schema()
        wanted_decode = ("VAEDecodeAudioTiled" if p.get("tiled_decode", True)
                         else "VAEDecodeAudio")
        decode_class = next((n for n in (wanted_decode, *self.DECODE_NODES)
                             if n in schema), wanted_decode)
        decode_wanted = {
            "samples": {"names": ["samples"], "value": [N_SAMPLER, 0],
                        "required": True},
            "vae": {"names": ["vae"], "value": [N_VAE, 0], "required": True},
        }
        if decode_class == "VAEDecodeAudioTiled":
            decode_wanted["tile"] = {"names": ["tile_size", "tile"],
                                     "value": int(p.get("tile_size")
                                                  or DEFAULTS["tile_size"])}
            decode_wanted["overlap"] = {"names": ["overlap"],
                                        "value": int(p.get("overlap")
                                                     or DEFAULTS["overlap"])}
        g[N_DECODE] = self._node(decode_class, decode_wanted)

        # Always lossless unless asked otherwise. ComfyUI writes flac, mp3 or
        # opus and no wav at all, so a wav is made from the flac afterwards.
        save_class = self.save_node()
        fmt = p.get("format") or "flac"
        if fmt not in self.save_formats():
            fmt = "flac"
        save_wanted = {
            "audio": {"names": ["audio"], "value": [N_DECODE, 0], "required": True},
            "prefix": {"names": ["filename_prefix"], "value": "audio/M3Studio"},
        }
        if save_class == "SaveAudioAdvanced":
            save_wanted["format"] = {"names": ["format"], "value": fmt}
        g[N_SAVE] = self._node(save_class, save_wanted)
        if save_class == "SaveAudioAdvanced":
            # mp3 and opus bring a "quality" input along with them; flac
            # brings none. It lives inside the format option rather than at
            # the top level of the schema, so _node() cannot see it. Current
            # ComfyUI names a nested input after its parent ("format.quality");
            # earlier builds took it bare. Both are sent — an input a node
            # does not have is ignored, a missing one fails validation.
            save_spec = self.node_inputs("SaveAudioAdvanced")
            extras = self._format_extras(save_spec.get("format"), fmt)
            if "quality" in extras and p.get("quality"):
                extras["quality"] = p["quality"]
            for name, value in extras.items():
                g[N_SAVE]["inputs"][name] = value
                g[N_SAVE]["inputs"]["format." + name] = value

        return {"prompt": g, "seed": seed, "dit": dit, "te": te, "vae": vae,
                "decode": decode_class, "duration": duration,
                "lyrics_sent": lyrics, "format": fmt}

    def _clip_type(self, spec) -> str:
        options = self.combo_options(spec)
        if not options or "minimax" in options:
            return "minimax"
        hit = next((o for o in options if "minimax" in o.lower()), None)
        if hit:
            return hit
        raise ComfyError("This ComfyUI's CLIPLoader has no 'minimax' type, so "
                         "it cannot load the Music 3 text encoder. Update "
                         "ComfyUI.")

    # ------------------------------------------------------------------ #
    # queue / results
    # ------------------------------------------------------------------ #
    def queue(self, prompt: dict) -> str:
        body = {"prompt": prompt, "client_id": self.client_id}
        r = requests.post(f"{self.url}/prompt", json=body, timeout=60)
        if r.status_code >= 400:
            try:
                err = r.json()
            except Exception:
                raise ComfyError(r.text[:500])
            raise ComfyError(_readable_error(err))
        return r.json()["prompt_id"]

    def interrupt(self) -> None:
        try:
            requests.post(f"{self.url}/interrupt", timeout=10)
        except Exception:
            pass

    @staticmethod
    def _in(state: dict, keys: tuple[str, ...], prompt_id: str) -> bool:
        for key in keys:
            for entry in state.get(key) or []:
                if len(entry) > 1 and entry[1] == prompt_id:
                    return True
        return False

    def is_running(self, prompt_id: str) -> bool:
        return self._in(self.queue_state(), ("queue_running",), prompt_id)

    def in_queue(self, prompt_id: str) -> bool:
        """Still running or still waiting its turn."""
        return self._in(self.queue_state(),
                        ("queue_running", "queue_pending"), prompt_id)

    def stop(self, prompt_id: str) -> None:
        """Take one prompt out of the queue, whatever state it is in.

        /interrupt stops whatever is executing right now and nothing else, so
        sending it for a prompt that is still waiting would kill somebody
        else's song and leave this one queued. Drop it from the pending queue
        by id first, and interrupt only if it really is the one running.
        """
        try:
            requests.post(f"{self.url}/queue", json={"delete": [prompt_id]},
                          timeout=10)
        except Exception:
            pass
        if self.is_running(prompt_id):
            self.interrupt()

    def history(self, prompt_id: str) -> dict:
        r = requests.get(f"{self.url}/history/{prompt_id}", timeout=20)
        r.raise_for_status()
        return r.json().get(prompt_id) or {}

    def queue_state(self) -> dict:
        try:
            return requests.get(f"{self.url}/queue", timeout=10).json()
        except Exception:
            return {}

    def outputs(self, prompt_id: str, hist: dict | None = None) -> list[dict]:
        hist = self.history(prompt_id) if hist is None else hist
        found = []
        for node_out in (hist.get("outputs") or {}).values():
            for key in ("audio", "audios", "result"):
                for item in node_out.get(key, []) or []:
                    if isinstance(item, dict) and item.get("filename"):
                        found.append(item)
        return found

    def failed(self, prompt_id: str, hist: dict | None = None) -> str | None:
        hist = self.history(prompt_id) if hist is None else hist
        status = hist.get("status") or {}
        if status.get("status_str") == "error":
            for kind, data in status.get("messages", []):
                if kind == "execution_error":
                    return (f"{data.get('node_type')}: "
                            f"{data.get('exception_message')}")
                if kind == "execution_interrupted":
                    return "Stopped in ComfyUI."
            return "Generation failed inside ComfyUI."
        return None

    def view(self, item: dict):
        params = {"filename": item.get("filename", ""),
                  "subfolder": item.get("subfolder", ""),
                  "type": item.get("type", "output")}
        return requests.get(f"{self.url}/view", params=params, stream=True,
                            timeout=120)


# --------------------------------------------------------------------------- #
# lyrics as the model takes them
# --------------------------------------------------------------------------- #
INSTRUMENTAL_LYRICS = "[Intro]\n\n[Instrumental]\n\n[Outro]"


def is_tag(line: str) -> bool:
    line = line.strip()
    return line.startswith("[") and line.endswith("]") and 2 < len(line) <= 42


def song_lyrics(lyrics: str, instrumental: bool) -> str:
    """What the encoder is handed.

    Music 3 reads section tags as structure and everything else as words to
    sing. An instrumental keeps the structure the person wrote and none of the
    words — and when there is no structure at all, a plain intro, break and
    outro, because an empty lyrics box gives the model nothing to pace by.
    """
    text = (lyrics or "").strip()
    if not instrumental:
        return text
    tags = [ln.strip() for ln in text.splitlines() if is_tag(ln)]
    return "\n\n".join(tags) if tags else INSTRUMENTAL_LYRICS


# --------------------------------------------------------------------------- #
# errors a person can act on
# --------------------------------------------------------------------------- #
def _gb(n: int) -> str:
    """Binary gigabytes: an 8 GB card's 8188 MiB reads as 8.0, not 8.6."""
    return f"{n / 2**30:.1f} GB"


def explain_failure(err: str, duration: int = 0, vram_bytes: int = 0,
                    te: str = "", dit: str = "") -> str:
    """Add the cause to an error that names only a symptom.

    The original text is kept — someone may need the exact wording — and the
    reading goes after it.
    """
    low = err.lower()
    small = bool(vram_bytes and vram_bytes < 12e9)
    if ("out of memory" in low or "outofmemory" in low or "cuda error" in low
            or ("allocat" in low and "memory" in low)):
        advice = [f"{err} — the card ran out of memory"]
        if small:
            advice.append(f"(it has {_gb(vram_bytes)})")
        tips = ["turn Low-VRAM decode on"]
        if duration > 150:
            tips.insert(0, f"try a shorter Length than {duration // 60}:"
                           f"{duration % 60:02d} — the render grows with it")
        if "bf16" in te.lower() or "fp16" in dit.lower():
            tips.append("pick the int8 models at the top of Create")
        tips.append("close games, browsers with video, or a local chat model "
                    "that holds the card")
        return " ".join(advice) + ". Most likely to help: " + "; ".join(tips) + "."
    if "killed" in low or "memoryerror" in low or "paging file" in low:
        return (f"{err} — the machine ran out of system memory. The text "
                "encoder is larger than an 8 GB card, so the part that does not "
                "fit waits in RAM: close other programs, or give Windows a "
                "larger page file, and try again.")
    if "tokenizer" in low or "audio_decoder" in low or "state_dict" in low:
        return (f"{err} — the text encoder file did not load as a Music 3 "
                "encoder. Check it downloaded completely on the Models page, "
                "and that ComfyUI is up to date.")
    return err


def _readable_error(err: dict) -> str:
    """Turn ComfyUI's validation payload into one sentence a person can act on."""
    node_errors = err.get("node_errors") or {}
    for node_id, info in node_errors.items():
        for e in info.get("errors", []):
            return (f"{info.get('class_type', 'node ' + str(node_id))}: "
                    f"{e.get('message')} {e.get('details', '')}".strip())
    top = err.get("error") or {}
    if top:
        return f"{top.get('message', 'Rejected by ComfyUI')} " \
               f"{top.get('details', '')}".strip()
    return json.dumps(err)[:400]
