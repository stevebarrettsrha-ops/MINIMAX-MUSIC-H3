# M3 Studio

A local, Suno-style front end for **MiniMax Music 3**. Describe the music,
write (or generate) lyrics, press Create, and get a full song with vocals,
up to six minutes long, in 32 kHz stereo. It all runs on your own machine,
with no account and no credits.

It is the sister app of [YuE Studio](https://github.com/stevebarrettsrha-ops/Yue-2---Suno-Like-Music),
with the same layout, engine handling and library, built around a different
model. It drives ComfyUI in the background. On first launch it installs
ComfyUI (or uses one you already have, including YuE Studio's), downloads the
model files and starts the engine for you.

---

## Running it

**Windows**: double-click `run.bat`.
**macOS / Linux**: open a terminal in this folder and run `./run.sh`.

On Linux, double-clicking `run.sh` in the file manager opens it in a text
editor; that is the file manager's choice, not a fault. Right-click it and
pick **Run as a Program** instead; it opens a terminal and runs there. If a
terminal answers `Permission denied` (an unzip can drop the executable bit),
run `bash run.sh`, or `chmod +x run.sh` once. On Debian and Ubuntu the first
run also needs `sudo apt-get install -y python3-venv`; the script says so when
it is missing, and picks up where it stopped once it is installed.

Either way the browser opens at <http://127.0.0.1:7789>. YuE Studio uses
7788, so both apps can be open at once.

The setup panel appears on the first run. Pick a route:

| Route | What happens |
|---|---|
| Use the ComfyUI already here | Found installs are listed, including a YuE Studio folder next to this one. Only the Music 3 models are downloaded. **Recommended if you have YuE Studio**: one engine and one PyTorch install for both apps. |
| Install a fresh ComfyUI | Clones ComfyUI into `./ComfyUI`, builds its own venv, installs PyTorch and downloads the models. |
| Connect to one I start myself | Set the address and models folder in Settings first. Setup only fetches missing models. |

MiniMax Music 3 needs a **recent ComfyUI**, one with the
`MiniMaxMusic3TextEncode` node. If the Engine page says the nodes are
missing, press **Update** on the ComfyUI row, then **Restart the engine**.

---

## Running on an 8 GB card (RTX 4060 and similar)

MiniMax Music 3 has three parts:

| Part | File (default) | Size | Where it runs |
|---|---|---|---|
| Text encoder: the 8B language model that *writes* the song, frame by frame | `minimax_music3_text_encoder_pruned_int8_convrot.safetensors` | ~9.2 GB | **Larger than an 8 GB card**, so ComfyUI streams it through the card a layer at a time |
| Diffusion model: renders the written song into audio | `minimax_music3_dit_int8_convrot.safetensors` | ~2.5 GB | Fits comfortably |
| Audio VAE: decodes to 32 kHz stereo | `minimax_music3_dav.safetensors` | small | Fits; tiled decode keeps it small |

The defaults are already the 8 GB choice: int8 everywhere, and **Low VRAM
decode** on. What to expect:

- **It works, but slowly.** Measured on an RTX 4060 (8 GB): a 2:00 song took
  **34½ minutes**, 32 of them writing the song and under 2 rendering the
  audio, so about **16 minutes per minute of music**. The job card shows
  the frames written so far and the elapsed time; the library records how
  long each song took (**Caption and lyrics → Rendered in**).
- **Songs longer than 6:00 are made in parts.** The model writes at most
  6:00 (9,000 frames) in one pass; that limit is the model's, not the app's.
  Past it, M3 Studio cuts the lyrics into parts at section boundaries, each
  within one pass, renders them one after another with the same caption and
  a seed family, and joins them into one track (crossfaded with ffmpeg, or
  concatenated by ComfyUI's own AudioConcat node when there is no ffmpeg).
  Each part's caption says it is part N of M of one continuous song, with no
  intro after the first and no ending before the last. That steers the model
  rather than binds it: the parts are separate performances, and the seams
  are where to listen. Lengths up to 30:00 are offered; a 10:00 song on an
  RTX 4060 is about 2¾ hours.
- **Length is a ceiling, and Auto fits it to the lyrics.** The model ends the
  song when the song ends; a ceiling shorter than the lyrics cuts the last
  sections off (badged **CUT**), which looks like the `[Section]` tags being
  ignored. Auto reads the lyrics and adds headroom (60 s to 30:00, in parts
  past 6:00). Time is
  only spent on what is written, but the encoder reserves memory for the
  whole ceiling (about 1 GB per two minutes), so don't set 6:00 for a short
  song.
- **System RAM matters.** The part of the text encoder that is not on the
  card waits in RAM. 32 GB is comfortable; 16 GB is the floor. The Engine
  page's *Graphics card and memory* row checks this.
- **Give it the whole card.** Close games, video and browser tabs playing
  video. Don't run YuE Studio's engine and this one as separate ComfyUIs at
  the same time. Share one ComfyUI between the two apps instead (see setup).
- **A local song writer shares the card too.** Use a small model (3–4B) in
  Ollama; M3 Studio asks Ollama to unload it after every write.
- **PyTorch must be built for CUDA 13.0.** Current ComfyUI switches off its
  CUDA kernels, including the int8 ones the 8 GB files use, on RTX 20-series
  and newer cards when PyTorch is older than cu130. A fresh install picks
  cu130 automatically; update the NVIDIA driver first if it refuses. The
  Engine page's PyTorch row warns when an existing install, YuE Studio's
  included, is on an older build.
- **Don't download the bf16 text encoder** (16.7 GB). It is offered on the
  Models page for 24 GB cards only.
- If a song fails with *out of memory*, the job card says what to change. It
  is almost always a shorter Length, or something else holding the card.

---

## Using it

MiniMax Music 3 takes two texts.

**Caption**: how the music sounds. A short line works:

> Upbeat synth-pop, female vocal, 118 BPM, glossy 80s drums, neon night-drive feel

For finer control, write it in three sections, the format the model was
trained on. The ≡ button inserts the template:

```
Global Metadata
Basic Attributes: bpm is 112. key is D, and scale is major. Indie Pop.
Global Emotional Progression: …
Application Scenarios & Imagery: …
Sonics & Production Profile: …

Vocal Details
Vocal Gender & Timbre: Singer A (Female). …
Vocal Style: …
Harmony/Backing Vocals: …
Vocal FX: …

Arrangement
Instrument Lifecycle Description (Primary/Secondary Layering): …
Groove & Foundation Progression: …
Embellishments, Textures & Spatial FX: …
```

**Start from a preset…** loads one of ten ready-made captions (indie pop,
synthwave, lo-fi, folk, rock, R&B, house, orchestral, country, K-pop) to
edit. Tag chips and **Explore genres** add words to a `Tags:` line at the end
of the caption.

**Lyrics**: the words to sing, under section tags: `[Intro]`, `[Verse]`,
`[Pre-Chorus]`, `[Chorus]`, `[Post-Chorus]`, `[Bridge]`, `[Instrumental]`,
`[Solo]`, `[Outro]`. The tags are the song's structure. Write repeats out in
full. The ≡ button on the Lyrics card inserts a structure sized to the
Length. Choose **Vocal: None** for an instrumental. Only the tags are sent,
as structure.

### The song writer (optional)

The panel at the top of Create turns one line ("a dark K-pop song about
walking through neon rain at midnight") into a full structured caption,
lyrics and a title. **Caption only** writes a caption for lyrics you already
have and leaves the lyrics as they are. **Lyrics only** writes lyrics for your
caption. **Polish mine** tidies your lyrics. **Put back** undoes a write.

It uses a chat model you choose under **Settings → Song writer**:

| Service | Key | Notes |
|---|---|---|
| Ollama, LM Studio, llama.cpp server | none | Runs on this computer. Press **List** to pick a model. |
| Claude (Anthropic) | yes | Defaults to `claude-opus-5-5`. |
| OpenAI, OpenRouter, any OpenAI-compatible server | usually | Give the API base, including `/v1`. |

The key is stored in `data/config.json`. It is shown back only as its last
four characters, and it is tied to the address it was entered for. A key is
only ever sent over `https://`, or to this computer. Making songs never needs
the writer.

### More Options

Every control maps onto a real node input:

| Control | Node input | Default |
|---|---|---|
| Vocal | words added to the caption (the model takes voice as text) | Any |
| Length | `MiniMaxMusic3TextEncode.max_duration` (a ceiling); past 6:00, one pass per part | Auto — sized to the lyrics |
| Style strength | `MiniMaxMusic3TextEncode.cfg_scale` | 1.7 (the official workflow's value) |
| Variety | `MiniMaxMusic3TextEncode.top_k` | 50 |
| Detail | `KSampler.steps` | 30 |
| Low VRAM decode | `VAEDecodeAudioTiled` (1536 / 64) vs `VAEDecodeAudio` | On |
| Seed | both passes' seed | random |

Songs that run into their Length are badged **CUT** in the list.
**Load these settings** on a song brings back everything that made it,
including the models. From the song's details it also brings back the seed,
so Create makes the same song again. **Make another take** re-renders it on a
fresh seed.

---

## The Models page

- **Recommended files**: the Comfy-Org repack, with one button per file.
- **Browse a repo**: any HuggingFace repo (`Comfy-Org/MiniMax-Music-3` by
  default), each file's target folder worked out for you.
- **Token and mirror**: for gated repos, or a mirror if huggingface.co is
  slow where you are.
- **Downloads** resume: partial files are kept as `.part` and continued.
- **On this machine**: every model file found, with sizes and Delete.

Files land in your ComfyUI models folder:

```
models/diffusion_models/minimax_music3_dit_int8_convrot.safetensors              ~2.5 GB  required (or the fp16 one)
models/text_encoders/minimax_music3_text_encoder_pruned_int8_convrot.safetensors ~9.2 GB  required
models/vae/minimax_music3_dav.safetensors                                                  required
models/diffusion_models/minimax_music3_dit_fp16.safetensors                       ~4.9 GB  optional
models/text_encoders/minimax_music3_text_encoder_pruned_bf16.safetensors          ~16.7 GB 24 GB cards only
```

## The Engine page

The same as YuE Studio's. It lists Python, Git, ComfyUI, PyTorch, ComfyUI's
packages, ffmpeg, the model files, your graphics card and memory, and the
engine. Each missing item has an Install button. The live engine console and
**Restart the engine** handle the two silent failures: a ComfyUI that started
before the models landed, and a different ComfyUI answering the address.

---

## Requirements

- Python 3.10 or newer (`run.bat` / `run.sh` offer to install it).
- Git, for the fresh-ComfyUI route.
- A recent ComfyUI with the MiniMax Music 3 nodes.
- An NVIDIA GPU. 8 GB works with the default int8 files; 32 GB of system RAM
  is recommended alongside it.
- About 13 GB of disk for the models, plus about 3 GB for PyTorch on the
  fresh-install route.
- ffmpeg, only to save as wav. ComfyUI writes flac, mp3 and opus itself.

## Licence of the model

MiniMax Music 3's weights are under the
[MiniMax-Music3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-Music3/blob/main/LICENSE).
Commercial use is allowed for companies under 20 million USD of yearly
revenue; above that it needs a commercial licence. Read it before publishing
what you make. M3 Studio itself does not download or bundle the weights until
you ask it to.

## Development

```bash
pip install -r requirements.txt -r requirements-dev.txt
python -m playwright install chromium     # browser tests only
python tests/run.py                       # gate, units, api, ui
python tests/run.py gate                  # just the syntax gate
```

The tests run against `tests/mock_comfy.py`, a stand-in ComfyUI whose
`/object_info` (`tests/object_info.json`) was captured from a real ComfyUI.
`tests/capture_object_info.py` refreshes it. See `CLAUDE.md` for the rules
the code keeps.
