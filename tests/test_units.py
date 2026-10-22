"""Pieces that need no server: the graph, the writer's parser, model roles."""
import json
import tempfile
from pathlib import Path

from harness import Suite, tiny_safetensors, MODEL_FILES

import bootstrap
import comfy
import lyricist
import manager
import parts

OBJECT_INFO = json.loads((Path(__file__).with_name("object_info.json")).read_text())


class FakeClient(comfy.ComfyClient):
    """A client whose schema is the captured one, with no network."""

    def __init__(self, info=None):
        super().__init__("http://127.0.0.1:1")
        self._info = info or OBJECT_INFO

    def schema(self, force=False):
        return self._info


def run() -> Suite:
    s = Suite("units")
    c = FakeClient()

    # ---- graph ----------------------------------------------------------
    built = c.build_prompt({"caption": "Indie pop, female vocal",
                            "lyrics": "[Verse]\nhello\n[Chorus]\nworld",
                            "duration": 90, "seed": 7})
    g = built["prompt"]
    types = {n["class_type"] for n in g.values()}
    s.check("graph has the blueprint's nodes",
            {"UNETLoader", "CLIPLoader", "VAELoader", "MiniMaxMusic3TextEncode",
             "ConditioningZeroOut", "EmptyMiniMaxMusic3LatentAudio", "KSampler",
             "VAEDecodeAudioTiled", "SaveAudioAdvanced"} <= types, str(types))
    enc = g[comfy.N_ENC]["inputs"]
    s.check("caption and lyrics reach the encoder",
            enc["caption"].startswith("Indie pop") and "[Verse]" in enc["lyrics"])
    s.check("max_duration is a float of the asked length",
            enc["max_duration"] == 90.0 and isinstance(enc["max_duration"], float))
    s.check("the official cfg_scale and top_k are the defaults",
            enc["cfg_scale"] == 1.7 and enc["top_k"] == 50, str(enc))
    s.check("one seed drives both passes",
            enc["seed"] == 7 and g[comfy.N_SAMPLER]["inputs"]["seed"] == 7)
    ks = g[comfy.N_SAMPLER]["inputs"]
    s.check("sampler defaults match the blueprint",
            (ks["steps"], ks["cfg"], ks["sampler_name"], ks["scheduler"])
            == (30, 1.7, "euler", "simple"), str(ks))
    s.check("negative is the zeroed conditioning",
            ks["negative"] == [comfy.N_ZERO, 0] and ks["positive"] == [comfy.N_ENC, 0])
    s.check("latent length comes from the encoder's seconds",
            g[comfy.N_LATENT]["inputs"]["seconds"] == [comfy.N_ENC, 1])
    s.check("text encoder loads as the minimax CLIP type",
            g[comfy.N_TE]["inputs"]["type"] == "minimax")
    s.check("int8 files are preferred on auto",
            "int8" in built["dit"] and "int8" in built["te"], built["dit"])
    s.check("unrelated files in a loader's list are never picked",
            "music3" in built["vae"], built["vae"])
    picked = c.build_prompt({"caption": "x", "dit":
                             "minimax_music3_dit_fp16.safetensors"})
    s.check("a named model is used", picked["dit"].endswith("fp16.safetensors"))
    ghost = c.build_prompt({"caption": "x", "dit": "gone.safetensors"})
    s.check("a model no longer listed falls back to auto",
            "int8" in ghost["dit"])
    long = c.build_prompt({"caption": "x", "duration": 5000})
    s.check("length is clamped to the node's ceiling (360 s)",
            long["duration"] == 360 and c.duration_limit() == 360,
            str(long["duration"]))
    s.check("the node's own default is what Auto uses",
            c.duration_default() == 120, str(c.duration_default()))
    plain = c.build_prompt({"caption": "x", "tiled_decode": False})
    s.check("Low VRAM decode off uses the plain decoder",
            plain["decode"] == "VAEDecodeAudio")
    tiled = g[comfy.N_DECODE]["inputs"]
    s.check("tiled decode uses the blueprint's tile and overlap",
            (tiled.get("tile_size"), tiled.get("overlap")) == (1536, 64), str(tiled))
    mp3 = c.build_prompt({"caption": "x", "format": "mp3"})
    s.check("mp3 brings its quality input along",
            "quality" in mp3["prompt"][comfy.N_SAVE]["inputs"]
            and mp3["format"] == "mp3", str(mp3["prompt"][comfy.N_SAVE]))
    s.check("an unknown format falls back to flac",
            c.build_prompt({"caption": "x", "format": "wav"})["format"] == "flac")
    batch = [c.build_prompt({"caption": "x"})["seed"] for _ in range(3)]
    s.check("a missing seed is drawn fresh each time", len(set(batch)) > 1)

    # instrumental lyrics
    s.check("instrumental keeps only the section tags",
            comfy.song_lyrics("[Verse]\nla la\n\n[Chorus]\nooh", True)
            == "[Verse]\n\n[Chorus]")
    s.check("instrumental with no tags gets a plain structure",
            comfy.song_lyrics("", True) == comfy.INSTRUMENTAL_LYRICS)
    s.check("sung lyrics pass through untouched",
            comfy.song_lyrics("  [Verse]\nhi  ", False) == "[Verse]\nhi")

    # an old ComfyUI
    old = FakeClient({k: v for k, v in OBJECT_INFO.items()
                      if k != "MiniMaxMusic3TextEncode"})
    try:
        old.build_prompt({"caption": "x"})
        s.check("an engine without Music 3 nodes is refused", False)
    except comfy.ComfyError as exc:
        s.check("an engine without Music 3 nodes is refused with the cure",
                "Update ComfyUI" in str(exc) and "MiniMaxMusic3TextEncode" in str(exc),
                str(exc))
    empty = json.loads(json.dumps(OBJECT_INFO))
    empty["CLIPLoader"]["input"]["required"]["clip_name"][0] = ["t5.safetensors"]
    try:
        FakeClient(empty).build_prompt({"caption": "x"})
        s.check("no text encoder listed is refused", False)
    except comfy.ComfyError as exc:
        s.check("no text encoder listed names the loader and the restart",
                "text encoder" in str(exc) and "Restart" in str(exc), str(exc))

    # failures explained
    oom = comfy.explain_failure("MiniMaxMusic3TextEncode: CUDA out of memory",
                                240, 8 * 1024 ** 3, "x_bf16", "")
    s.check("out of memory gets the 8 GB advice",
            "shorter Length" in oom and "int8" in oom and "8.0 GB" in oom, oom)
    s.check("other errors pass through untouched",
            comfy.explain_failure("KSampler: boom") == "KSampler: boom")

    # ---- model roles ----------------------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        s.check("an empty folder misses all three roles",
                bootstrap.missing_roles(root) == ["dit", "te", "vae"])
        for rel in MODEL_FILES:
            tiny_safetensors(root / rel)
        s.check("the three default files fill every role",
                bootstrap.missing_roles(root) == [])
        (root / MODEL_FILES[0]).unlink()
        tiny_safetensors(root / "unet" / "minimax_music3_dit_fp16.safetensors")
        s.check("the fp16 model in the old unet folder fills the dit role",
                bootstrap.missing_roles(root) == [])
        part = root / MODEL_FILES[1]
        raw = part.read_bytes()
        part.write_bytes(raw[:-4])
        s.check("a truncated text encoder is caught by its own header",
                bootstrap.missing_roles(root) == ["te"])
        todo = bootstrap.missing_models(root, {"download_fp16": True})
        s.check("setup downloads the default for a missing role, required",
                [(Path(m[0]).name, m[3]) for m in todo][0]
                == ("minimax_music3_text_encoder_pruned_int8_convrot.safetensors", True),
                str(todo))
        s.check("a role filled by an alternative is not downloaded again",
                not any("dit_int8" in m[0] for m in todo))

    s.check("folders are guessed for the Comfy-Org layout",
            manager.guess_folder("text_encoders/a.safetensors") == "text_encoders"
            and manager.guess_folder("minimax_music3_dav.safetensors") == "vae"
            and manager.guess_folder("x_dit_fp16.safetensors") == "diffusion_models")

    # ---- set up on Windows, opened on Linux (a drive shared by both) -----
    # The Windows venv and ffmpeg are still on disk, so they "exist"; every
    # run of them is [Errno 8] Exec format error and the engine never starts.
    import os
    import platform as _platform
    import subprocess as _subprocess
    from unittest.mock import patch as _patch
    with tempfile.TemporaryDirectory() as tmp, \
            _patch.object(_platform, "system", return_value="Linux"):
        app = Path(tmp) / "minimax-music-h3-main"
        comfy_dir = app / "ComfyUI"
        (comfy_dir / "models").mkdir(parents=True)
        (comfy_dir / "main.py").write_text("")
        (comfy_dir / "requirements.txt").write_text("")
        win_py = app / "comfy-venv" / "Scripts" / "python.exe"
        win_py.parent.mkdir(parents=True)
        win_py.write_bytes(b"MZ\x90\x00" + b"\x00" * 60)
        elf = app / "elf"
        elf.write_bytes(b"\x7fELF\x02\x01")
        s.check("a Windows .exe is foreign on Linux",
                bootstrap.foreign_binary(win_py))
        s.check("an ELF binary is native on Linux",
                not bootstrap.foreign_binary(elf))
        s.check("a script or unreadable file is never called foreign",
                not bootstrap.foreign_binary(comfy_dir / "main.py")
                and not bootstrap.foreign_binary(app / "nope"))
        with _patch.object(_platform, "system", return_value="Windows"):
            s.check("an ELF binary is foreign on Windows",
                    bootstrap.foreign_binary(elf))

        cfg_w = dict(bootstrap.DEFAULT_CONFIG, comfy_dir=str(comfy_dir),
                     models_dir=str(comfy_dir / "models"), python=str(win_py),
                     setup_complete=True, managed=True)
        s.check("comfy_python does not hand out the foreign interpreter",
                manager.comfy_python(cfg_w) == "",
                manager.comfy_python(cfg_w))
        notes = bootstrap.heal_paths(cfg_w)
        s.check("a saved interpreter built for Windows is cleared on Linux",
                cfg_w["python"] == "", cfg_w["python"])
        s.check("and the reason is reported",
                any("another operating system" in n for n in notes),
                "; ".join(notes))
        moved = dict(cfg_w, python="D:\\AI\\minimax-music-h3-main\\"
                     "comfy-venv\\Scripts\\python.exe")
        real_app = bootstrap.APP_DIR
        bootstrap.APP_DIR = app
        try:
            bootstrap.heal_paths(moved)
        finally:
            bootstrap.APP_DIR = real_app
        s.check("a rebased path that lands on the Windows python is cleared",
                moved["python"] == "", moved["python"])
        s.check("the Windows venv is left alone; Linux gets its own beside it",
                bootstrap.venv_python(comfy_dir)
                == app / "comfy-venv-linux" / "bin" / "python",
                str(bootstrap.venv_python(comfy_dir)))
        s.check("existing_python never runs the foreign interpreter",
                bootstrap.existing_python(comfy_dir) == "")
        rows = {d["id"]: d for d in manager.dependencies(cfg_w)}
        s.check("PyTorch says the old venv is for another system, not Errno 8",
                rows["torch"]["state"] == "missing"
                and "another operating system" in rows["torch"]["detail"]
                and rows["torch"]["action"] == "install",
                rows["torch"]["detail"][:90])
        s.check("ComfyUI packages are missing and point at PyTorch first",
                rows["comfy_reqs"]["state"] == "missing"
                and "PyTorch first" in rows["comfy_reqs"]["detail"],
                rows["comfy_reqs"]["detail"])
        (app / "comfy-venv" / "bin").mkdir()
        (app / "comfy-venv" / "bin" / "python").write_bytes(b"\x7fELF")
        s.check("a comfy-venv that also has a native interpreter is used",
                bootstrap.venv_python(comfy_dir)
                == app / "comfy-venv" / "bin" / "python")

        real_tools = bootstrap.TOOLS_DIR
        bootstrap.TOOLS_DIR = app / "data" / "tools"
        try:
            ffdir = bootstrap.TOOLS_DIR / "ffmpeg" / "bin"
            ffdir.mkdir(parents=True)
            (ffdir / "ffmpeg.exe").write_bytes(b"MZ\x90\x00")
            s.check("a bundled Windows ffmpeg is passed over on Linux",
                    bootstrap.find_tool("ffmpeg") != str(ffdir / "ffmpeg.exe"))
            with _patch.object(bootstrap.shutil, "which", return_value=None):
                rows = {d["id"]: d for d in manager.dependencies(cfg_w)}
            s.check("and the ffmpeg row says why instead of reading ok",
                    rows["ffmpeg"]["state"] != "ok"
                    and "another operating system" in rows["ffmpeg"]["detail"]
                    and rows["ffmpeg"]["action"] == "install",
                    rows["ffmpeg"]["detail"][:90])
        finally:
            bootstrap.TOOLS_DIR = real_tools

        # apt-get from a task with no terminal: sudo -n fails where sudo
        # needs a password, which on a desktop Ubuntu is nearly everyone
        task = manager.Task("t", "t")
        no = _subprocess.CompletedProcess([], 1)
        with _patch.object(_subprocess, "run", return_value=no), \
                _patch.object(manager.shutil, "which",
                              side_effect=lambda n: "/usr/bin/" + n), \
                _patch.dict(os.environ, {"DISPLAY": ":0"}):
            s.check("sudo that needs a password becomes pkexec on a desktop",
                    manager._elevate(["sudo", "-n", "apt-get", "install", "-y",
                                      "git"], task)
                    == ["pkexec", "apt-get", "install", "-y", "git"])
        with _patch.object(_subprocess, "run", return_value=no), \
                _patch.dict(os.environ, {"DISPLAY": "", "WAYLAND_DISPLAY": ""}):
            s.check("and stays sudo -n, failing fast, without one",
                    manager._elevate(["sudo", "-n", "apt-get", "install", "-y",
                                      "git"], task)
                    == ["sudo", "-n", "apt-get", "install", "-y", "git"])

    # ---- the writer's answer -------------------------------------------
    answer = """<think>plan the song</think>
**TITLE:** Neon Rain
CAPTION:
Global Metadata
Basic Attributes: bpm is 100. key is A, and scale is minor. Synthwave.
Lyrics are about the city (a caption line, not a heading)

Vocal Details
Vocal Gender & Timbre: Singer A (Male).

Arrangement
Primary: synth bass.
LYRICS:
[Verse]
Walking through the rain
[Chorus]
Neon on my skin
"""
    r = lyricist.parse(answer)
    s.check("title, caption and lyrics are split", r["title"] == "Neon Rain"
            and r["caption"].startswith("Global Metadata")
            and r["lyrics"].startswith("[Verse]"), json.dumps(r)[:300])
    s.check("a caption line starting with 'Lyrics' stays in the caption",
            "Lyrics are about the city" in r["caption"])
    s.check("tags are spaced into sections",
            "Walking through the rain\n\n[Chorus]" in r["lyrics"], r["lyrics"])
    s.check("reasoning is never shown", "plan the song" not in json.dumps(r))
    s.check("a complete caption raises no warning", not r["warnings"], str(r["warnings"]))
    j = lyricist.parse('{"title":"T","style":"Pop.","lyrics":"[Verse]\\nhi"}')
    s.check("a JSON answer with 'style' is read as the caption",
            j["caption"] == "Pop." and j["title"] == "T")
    bare = lyricist.parse("just some words\nwith no labels")
    s.check("an unlabelled answer lands in the lyrics with a warning",
            bare["lyrics"].startswith("just") and bare["warnings"])
    req = lyricist.clean_request({"want": "caption", "lyrics": "[Verse]\nx",
                                  "vocal": "duet", "duration": 9999})
    s.check("requests are bounded", req["duration"] == 1800 and req["vocal"] == "duet")
    prompt = lyricist.user_prompt(req)
    s.check("a caption-only ask keeps the lyrics exactly",
            "keep exactly" in prompt and "[Verse]" in prompt, prompt)
    auto = lyricist.clean_request({"auto": True, "duration": 360})
    s.check("Auto length means a typical song, not the ceiling",
            auto["duration"] == lyricist.TYPICAL_SECONDS)
    # ---- a long song, in parts -----------------------------------------
    verse = lambda n: "[Verse]\n" + "\n".join(f"line {n} {i}" for i in range(8))
    chorus = "[Chorus]\n" + "\n".join("hook" for _ in range(4))
    song = "\n\n".join(["[Intro]", verse(1), chorus, verse(2), chorus,
                         "[Bridge]\nturn\nturn\nturn\nturn", "[Solo]", chorus,
                         verse(3), chorus, "[Outro]"])
    est = parts.estimate_seconds(song)
    s.check("an eleven-section song estimates a full-length song", 300 <= est <= 400, str(est))
    double = parts.estimate_seconds(song + "\n\n" + song)
    s.check("twice the sections estimates past one pass", double > 360, str(double))
    s.check("no lyrics estimate the node's default", parts.estimate_seconds("") == 120)
    one = parts.plan(song, False, 300)
    s.check("a song that fits is one part with all its sections",
            len(one) == 1 and one[0]["lyrics"].count("[") == 11
            and one[0]["seconds"] == 300, str(one)[:200])
    two = parts.plan(song, False, 540)
    s.check("nine minutes is two parts", len(two) == 2 and two[0]["count"] == 2)
    s.check("every section lands in exactly one part, in order",
            "\n\n".join(x["lyrics"] for x in two) == song,
            repr("\n\n".join(x["lyrics"] for x in two))[:200])
    s.check("parts are cut at section boundaries",
            all(x["lyrics"].startswith("[") for x in two))
    s.check("each part's ceiling is within one pass and near its share",
            all(60 <= x["seconds"] <= 360 for x in two)
            and 480 <= sum(x["seconds"] for x in two) <= 660,
            str([x["seconds"] for x in two]))
    four = parts.plan(song, False, 1200)
    s.check("twenty minutes is four parts, balanced by content",
            len(four) == 4 and max(len(x["lyrics"]) for x in four)
            < 3 * min(len(x["lyrics"]) for x in four), str([len(x["lyrics"]) for x in four]))
    short = parts.plan("[Verse]\nonly one section", False, 900)
    s.check("fewer sections than parts means fewer parts", len(short) == 1)
    inst = parts.plan("", True, 800)
    s.check("an instrumental with no lyrics repeats a plain structure",
            len(inst) == 3 and all(x["lyrics"] == parts.INSTRUMENTAL_LYRICS for x in inst)
            and sum(x["seconds"] for x in inst) >= 780, str(inst))
    cap = parts.part_caption("Roots reggae.", 2, 3)
    s.check("a middle part is told it is one song, no intro, no ending",
            cap.startswith("Roots reggae.") and "part 2 of 3" in cap
            and "no intro" in cap and "without an outro" in cap, cap)
    s.check("a single part's caption is untouched",
            parts.part_caption("x", 1, 1) == "x")
    g = c.build_join_prompt(["a.flac", "b.flac", "c.flac"], "mp3")["prompt"]
    kinds = [n["class_type"] for n in g.values()]
    s.check("the join graph loads each part and chains the concats",
            kinds.count("LoadAudio") == 3 and kinds.count("AudioConcat") == 2
            and g["c2"]["inputs"]["audio1"] == ["c1", 0]
            and g[comfy.N_SAVE]["inputs"]["audio"] == ["c2", 0], str(g)[:300])
    return s


if __name__ == "__main__":
    suite = run()
    raise SystemExit(1 if suite.failures else 0)
