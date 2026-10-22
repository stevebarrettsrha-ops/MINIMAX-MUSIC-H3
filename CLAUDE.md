# M3 Studio — invariants

A local front end for MiniMax Music 3 that drives ComfyUI. It is ported from
YuE Studio (stevebarrettsrha-ops/Yue-2---Suno-Like-Music); the rules below that
carry over were learned there the hard way, so keep them.

## Hard rules

1. **`web/index.html` stays one file with no build step.** CSS and JS inline.
2. **Never hard-code the ComfyUI API workflow.** `comfy.py` builds the graph
   from `/object_info` and matches input names through candidate lists, so a
   renamed input fails by name instead of silently sending a wrong value.
3. **Never read a dropdown's choices as `spec[0]`.** `/object_info` writes
   combos as `[["a","b"], {…}]` for V1 nodes (the loaders, KSampler) and
   `["COMBO", {"options": […]}]` for V3 ones (the Music 3 and audio nodes).
   Go through `ComfyClient.combo_options()`. `SaveAudioAdvanced.format` is a
   dynamic combo whose chosen option brings a sibling `quality` input;
   `_format_extras()` fills it.
4. **Python detection is by execution, never PATH lookup** (Windows Store
   stubs). `run.bat` and `bootstrap.find_python` both test with `-c`.
5. **Model downloads are resumable and checked.** Stream to `<name>.part`,
   `Range` on retry, `replace()` only when the bytes match the declared
   `Content-Length`. A model is complete when its own safetensors header says
   so (`safetensors_size()`), never when it clears a size in `MODELS`.
6. **Never touch an existing ComfyUI's Python environment.** Dependency
   installs run only in managed mode, only inside `comfy-venv`. Sharing YuE
   Studio's ComfyUI is the "existing" route and leaves its environment alone.
7. **HuggingFace work goes through the front end** (`/api/hf/*`). Tokens and
   writer keys are returned masked, never in full. No response carries the
   config's secrets.
8. **Model deletes are path-checked** (folder in `MODEL_FOLDERS`, bare name,
   resolved parent equals the folder).
9. **Long work is a Task, never a blocking request.** The page polls
   `/api/tasks`; cancel sets `task.cancel`.
10. **Finished audio is copied into `data/tracks/`.** ComfyUI's output folder
    is not storage.
11. **Nothing that does not look like ComfyUI is ever killed**
    (`take_over_port()`), `settled_free()` waits 2 s before calling a port
    free, every (re)start calls `_refresh_schema_when_up()`, `kill_pid()`
    returns what the system said, and child processes are read as UTF-8 with
    `errors="replace"`.
12. **"Ready" is not "the port answered".** `engine_starting()` withholds
    ready until ComfyUI prints its "Starting server" banner (for an engine we
    started), and `ready` also needs the Music 3 nodes, a file for each of the
    three loaders, no stale scan and no foreign engine at the address.
13. **A song is timed out for stalling, never for being slow or queued.**
    `STALL_LIMIT` runs from the last node/step ComfyUI reported, and only once
    the prompt is running. On an 8 GB card the text encoder streams through
    the card and is slow by design.
14. **The song writer is optional and keeps keys where they belong**
    (bound to the address they were entered for, https or loopback only).
    Auto length reaches it as `TYPICAL_SECONDS`, never the 360 s ceiling,
    and a local writer is told to release the GPU after every write.
15. **A saved path can be there and still wrong.** An app folder on a drive
    shared by Windows and Linux carries the Windows
    `comfy-venv/Scripts/python.exe` and `data/tools/ffmpeg/bin/ffmpeg.exe`;
    they exist, and every run of them is `[Errno 8] Exec format error`.
    `foreign_binary()` reads the first bytes (MZ, ELF, Mach-O); `heal_paths()`
    clears a foreign interpreter, `comfy_python()` / `existing_python()` and
    `find_tool()` pass over one. The foreign venv is never rebuilt in place —
    that would break it for the other system — so `venv_dir()` gives this one
    `comfy-venv-linux` (or `-windows`, `-darwin`) beside it, and `run.sh` does
    the same for a Windows `.venv` (`.venv-linux`). Setup never stores the
    interpreter that *builds* the venv (often the app's own, torch-less) as
    `cfg["python"]`, and `verify_locations()` never adopts it for a managed
    engine. apt-get runs as `sudo -n`, or `pkexec` on a desktop where sudo
    needs a password; a failure names the command to run by hand.

## Music 3 specifics

- **Three roles, not one checkpoint.** `bootstrap.MODELS` lists files by
  role: `dit` (UNETLoader, `diffusion_models/`), `te` (CLIPLoader type
  `minimax`, `text_encoders/`), `vae` (VAELoader, `vae/`). A role is filled by
  any complete file listed under it, including in the old `unet/` and `clip/`
  folders (`model_path()`). Only an *empty role* gates `ready`
  (`missing_roles()`). The fp16 diffusion model is an alternative, never
  required.
- **Every Music 3 file has `music3` in its name** (`MARKER`). Loader lists are
  filtered by it: a loader's list also holds Flux, T5 and other files, and
  "the first file in the list" must never be picked.
- **Auto prefers int8** for both the diffusion model and the text encoder.
  A model the person picked sticks (it rides in the draft and is restored
  once the engine's list arrives), and the pickers are rewritten only when
  the engine's list actually changes.
- **Defaults are the official blueprint's**: KSampler 30 steps, cfg 1.7,
  euler, simple; encoder cfg_scale 1.7 (the node's own default is 1.5),
  top_k 50; tiled decode 1536/64; one seed drives both passes.
- **Length is a ceiling.** `max_duration` (≤ 360 s, read from the schema)
  bounds the encoder; the encoder's `seconds` output sizes the latent. A song
  as long as its limit is marked `cut`. Auto is sized to the lyrics
  (`parts.estimate_seconds`), never the ceiling: the encoder reserves its
  KV cache for the whole ceiling (~1 GB per two minutes on the card).
- **Past one pass, a song is parts.** `parts.plan()` cuts the lyrics at
  section boundaries into as few parts as the engine's limit allows, each
  with its own ceiling; `build_parts()` builds every pass up front (so a
  bad option fails at the Create click), on seeds `base + 1000·(i−1)`;
  `run_job` renders them in order and joins them — ffmpeg `acrossfade`
  when there is one (`M3_STUDIO_NO_FFMPEG=1` forces the other route for
  tests), else ComfyUI's LoadAudio → AudioConcat → save, with the parts
  uploaded to its input folder. A part that hits its ceiling marks the
  whole song `cut`. The page's `lyricsSeconds()` is the same arithmetic as
  `parts.estimate_seconds()`: change both or neither.
- **Voice is words.** The Vocal buttons add a line to the caption unless the
  caption already says it. An instrumental sends only the lyrics' section
  tags (or `[Intro] [Instrumental] [Outro]`), never the words.
- **Progress has two phases.** The ws listener records which node a
  `progress` message belongs to: the encoder counts frames (25/s) against the
  ceiling, the sampler counts steps. They must not be read as each other.
- **The song writer returns TITLE / CAPTION / LYRICS.** `parse()` accepts a
  heading only with its colon or alone on a line, so a caption sentence that
  starts with "Lyrics" stays in the caption. What the person asked to keep
  (caption on a lyrics-only write, lyrics on a caption-only write) is put
  back whatever the model did.

## Validation gate — run after any edit

```bash
python tests/run.py gate      # py_compile + node --check of the inline JS + element ids
python tests/run.py           # everything: gate, units, api, ui
```

A missing function in the inline script kills all interactivity silently;
`node --check` is not optional. The UI group needs Playwright
(`pip install -r requirements-dev.txt && python -m playwright install chromium`)
and steps aside without it. `tests/object_info.json` is a capture from a real
ComfyUI. Refresh it with `tests/capture_object_info.py`, never by hand.
