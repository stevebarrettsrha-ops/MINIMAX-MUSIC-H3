"""
lyricist.py - the song writer: a structured caption, lyrics and a title,
written by a language model.

MiniMax Music 3 takes two texts. The caption describes the music in three
sections — Global Metadata, Vocal Details, Arrangement — and the lyrics carry
the words under [Section] tags, which the model reads as the song's structure.
A detailed caption is what makes the difference between a generic track and
the one that was imagined, and writing one by hand is the slow part. This asks
a chat model the person already has — Ollama or LM Studio on this machine, any
OpenAI-compatible server, or Claude — to write them in exactly that shape.

It runs as a Task (manager.spawn), streams, and stops between chunks when the
task is cancelled. Nothing here is required: with no writer set up, the page
simply keeps its Write button pointed at Settings.

The two-stage idea (expand an idea into a structured caption, then write
lyrics that fit it, with structure scaled to the length) follows the
ComfyUI-MiniMax-M3-SongPlanner node by benjiyaya (MIT); the caption's shape is
the one documented on the MiniMax Music 3 model card. The prompts here are
written for this app.
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
from urllib.parse import urlsplit

import requests

# id: label, wire protocol, default address, whether a key is expected, the
# environment variable a key may come from, and the model to suggest.
#
# The environment variable is only read for the provider's own default
# address: a key meant for OpenAI must never follow a changed address to
# somebody else's server.
PROVIDERS = {
    "ollama": {"label": "Ollama (this computer)", "wire": "openai",
               "base": "http://127.0.0.1:11434/v1", "key": False, "env": "",
               "model": ""},
    "lmstudio": {"label": "LM Studio (this computer)", "wire": "openai",
                 "base": "http://127.0.0.1:1234/v1", "key": False, "env": "",
                 "model": ""},
    "llamacpp": {"label": "llama.cpp server (this computer)", "wire": "openai",
                 "base": "http://127.0.0.1:8080/v1", "key": False, "env": "",
                 "model": ""},
    "anthropic": {"label": "Claude (Anthropic)", "wire": "anthropic",
                  "base": "https://api.anthropic.com", "key": True,
                  "env": "ANTHROPIC_API_KEY", "model": "claude-opus-5-5"},
    "openai": {"label": "OpenAI", "wire": "openai",
               "base": "https://api.openai.com/v1", "key": True,
               "env": "OPENAI_API_KEY", "model": ""},
    "openrouter": {"label": "OpenRouter", "wire": "openai",
                   "base": "https://openrouter.ai/api/v1", "key": True,
                   "env": "OPENROUTER_API_KEY", "model": ""},
    "custom": {"label": "Other OpenAI-compatible server", "wire": "openai",
               "base": "", "key": False, "env": "", "model": ""},
}
DEFAULT_PROVIDER = "ollama"

# Claude models that take the server-side refusal fallback. A policy decline
# on one of them is re-run on Anthropic's recommended model inside the same
# call instead of ending the write with nothing.
FALLBACK_MODELS = ("claude-opus-5-5", "claude-fable-5-1", "claude-opus-5",
                   "claude-sonnet-5-5")
FALLBACK_BETA = "server-side-fallback-2026-07-01"

# A caption and a song are a few hundred words each. The ceiling bounds what a runaway costs; it
# is far above what a finished answer needs.
MAX_OUTPUT_TOKENS = 16000
MAX_CHARS = 200_000
CONNECT_TIMEOUT = 10
# A local model can sit silent while it loads into memory before the first
# word arrives. This is the gap allowed between chunks, not the whole write.
READ_TIMEOUT = 300

WANTS = ("song", "caption", "lyrics", "polish")
# What Auto length means to the writer: a typical song.
TYPICAL_SECONDS = 180
# How long LM Studio may keep a model it loaded on demand once a write is
# done. Only models it loaded itself (JIT) honour it; one loaded by hand stays.
LMSTUDIO_TTL = 60


class LyricistError(RuntimeError):
    """Something the person can act on, worded for the page."""


# --------------------------------------------------------------------------- #
# connection
# --------------------------------------------------------------------------- #
def _loopback(host: str) -> bool:
    if host in ("localhost", ""):
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def clean_base(base: str) -> str:
    """An API address, or "" when it is not one.

    No credentials, query or fragment: the key travels in a header, and an
    address that carries one would write it into the config in plain view.
    """
    base = (base or "").strip().rstrip("/")
    if not base:
        return ""
    try:
        parts = urlsplit(base)
        _ = parts.port
    except ValueError:
        return ""
    if (parts.scheme not in ("http", "https") or not parts.hostname
            or parts.username or parts.password or parts.query
            or parts.fragment or any(c.isspace() for c in base)):
        return ""
    return base


def provider_of(cfg: dict) -> dict:
    pid = cfg.get("llm_provider") or DEFAULT_PROVIDER
    return {"id": pid, **PROVIDERS.get(pid, PROVIDERS[DEFAULT_PROVIDER])}


def base_of(cfg: dict) -> str:
    return clean_base(cfg.get("llm_base") or "") or provider_of(cfg)["base"]


def key_of(cfg: dict) -> tuple[str, str]:
    """The key to send, and where it came from ("saved", "environment", "").

    A saved key is bound to the address it was entered for. Pointing the
    writer somewhere else does not carry the secret along.
    """
    base = base_of(cfg)
    saved = cfg.get("llm_key") or ""
    if saved and cfg.get("llm_key_base") == base:
        return saved, "saved"
    prov = provider_of(cfg)
    if prov["env"] and base == prov["base"] and os.environ.get(prov["env"]):
        return os.environ[prov["env"]], "environment"
    return "", ""


def connection(cfg: dict) -> dict:
    """Everything a write needs, or a LyricistError saying what is missing."""
    prov = provider_of(cfg)
    base = base_of(cfg)
    if not base:
        raise LyricistError("Set the song writer's address in Settings.")
    model = (cfg.get("llm_model") or prov["model"] or "").strip()
    if not model:
        raise LyricistError("Choose a model for the song writer in Settings "
                            "— List models shows what the server has.")
    key, _ = key_of(cfg)
    if prov["key"] and not key:
        raise LyricistError(f"{prov['label']} needs an API key. Add it in "
                            "Settings.")
    host = urlsplit(base).hostname or ""
    if key and urlsplit(base).scheme != "https" and not _loopback(host):
        raise LyricistError("An API key is only sent over https, or to this "
                            "computer. Use an https:// address.")
    return {"provider": prov["id"], "wire": prov["wire"], "label": prov["label"],
            "base": base, "model": model, "key": key}


def settings_view(cfg: dict) -> dict:
    """What the page may see. The key is never returned, only its last four."""
    prov = provider_of(cfg)
    key, source = key_of(cfg)
    return {
        "provider": prov["id"],
        "base": base_of(cfg),
        "model": cfg.get("llm_model") or prov["model"],
        "key_set": bool(key),
        "key_source": source,
        "key_hint": ("…" + key[-4:]) if source == "saved" and len(key) > 4 else "",
        "configured": configured(cfg),
        "providers": [{"id": pid, "label": p["label"], "base": p["base"],
                       "needs_key": p["key"], "model": p["model"],
                       "env": p["env"]}
                      for pid, p in PROVIDERS.items()],
    }


def configured(cfg: dict) -> bool:
    try:
        connection(cfg)
        return True
    except LyricistError:
        return False


def apply_settings(cfg: dict, body: dict) -> dict:
    """Merge a settings form into cfg. Returns the keys that changed.

    A key typed in is bound to the address being saved with it. An address
    that changes without a new key drops the old one rather than sending it
    somewhere it was never meant for.
    """
    change: dict = {}
    pid = body.get("provider")
    if isinstance(pid, str) and pid in PROVIDERS:
        change["llm_provider"] = pid
        if pid != (cfg.get("llm_provider") or DEFAULT_PROVIDER):
            # A new service starts from its own address and model, not the
            # last one's: Claude asked for "qwen3:8b" at Ollama's port is two
            # mistakes the person never made.
            change["llm_base"] = ""
            change["llm_model"] = ""
    if isinstance(body.get("base"), str):
        base = clean_base(body["base"])
        if body["base"].strip() and not base:
            raise LyricistError("That address is not usable. Give the API "
                                "base, e.g. http://127.0.0.1:11434/v1, with "
                                "no key or query in it.")
        change["llm_base"] = base
    if isinstance(body.get("model"), str):
        change["llm_model"] = body["model"].strip()[:200]
    after = {**cfg, **change}
    new_base = base_of(after)
    if isinstance(body.get("key"), str) and body["key"].strip():
        key = body["key"].strip()
        if len(key) > 8192 or any(not 33 <= ord(c) <= 126 for c in key):
            raise LyricistError("An API key is printable characters with no "
                                "spaces or line breaks.")
        change["llm_key"] = key
        change["llm_key_base"] = new_base
    elif body.get("clear_key") is True:
        change["llm_key"] = ""
        change["llm_key_base"] = ""
    elif cfg.get("llm_key") and cfg.get("llm_key_base") != new_base:
        change["llm_key"] = ""
        change["llm_key_base"] = ""
    cfg.update(change)
    return change


# --------------------------------------------------------------------------- #
# the ask
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = """\
You are a music producer and songwriter writing for MiniMax Music 3, a model \
that generates a complete song — vocals and arrangement — from a structured \
music caption and tagged lyrics. Answer with exactly three parts, in this \
order and format, and nothing else:

TITLE: <a short song title>
CAPTION:
<the structured caption>
LYRICS:
<the lyrics>

CAPTION
Write these three sections, each heading alone on its own line:

Global Metadata
Basic Attributes: tempo as "bpm is N" (or a narrow range), and key and scale \
only if they serve the song (e.g. "key is E, and scale is minor"). Then the \
genre and subgenres.
Global Emotional Progression: how the feeling moves from the opening to the \
peak to the ending.
Application Scenarios & Imagery: where the song belongs and what it evokes.
Sonics & Production Profile: the soundstage, low end, dynamics, era and mix \
character.

Vocal Details
Vocal Gender & Timbre: e.g. "Singer A (Female)", then register and timbre.
Vocal Style: delivery in the verses, how it builds, what the chorus does.
Harmony/Backing Vocals: where harmonies and backing voices appear.
Vocal FX: reverb, delay, saturation — restrained and specific.
For an instrumental, write "Instrumental — no vocals" and describe the lead \
melodic instrument here instead.

Arrangement
Instrument Lifecycle Description (Primary/Secondary Layering): the primary \
instruments and when they enter, change and leave; then the secondary layers.
Groove & Foundation Progression: drums and bass section by section.
Embellishments, Textures & Spatial FX: transitions, risers, solos, ear candy.

Caption rules: 250 to 450 words. Concrete musical events, not decorative \
adjectives. One consistent genre across all three sections. Follow the \
lyrics' sections in order (intro, verse, chorus, bridge, outro) so the \
arrangement has a readable energy arc. No lyrics, no song title and no \
commentary inside the caption. Keep what the person already chose — genre, \
tempo, voice, instruments — unless the brief asks for something else.

LYRICS
- Every section starts with a tag alone on its own line: [Intro], [Verse], \
[Pre-Chorus], [Chorus], [Post-Chorus], [Bridge], [Instrumental], [Solo], \
[Outro]. A blank line between sections.
- Under a sung tag, only the words to be sung: no stage directions, chords, \
timings, speaker names or translations. Short ad-libs in parentheses are fine.
- [Instrumental] and [Solo] are the tag alone. An [Intro] or [Outro] without \
words is the tag alone too.
- Write every repeat out in full — never "(repeat chorus)" or "x2".
- 4 to 8 lines per verse, 4 to 6 per chorus; singable lines with a steady \
syllable count, concrete images, and a chorus hook that returns.
- Match the requested length and never exceed it: roughly one section per 20 \
to 30 seconds. A short song is [Intro] [Verse] [Chorus] [Outro]; a standard \
one adds a second verse, a pre-chorus and a bridge; only a long one gets solos \
and extra instrumental breaks.
- For an instrumental, LYRICS holds only section tags.

TITLE: two to five words, from the song's own images. No quotation marks.

Write original words. Do not quote or imitate an existing song's lyrics, and \
do not name real artists.\
"""


def clean_request(body: dict) -> dict:
    """The page's ask, bounded to what the Create page can hold."""
    def text(key: str, limit: int) -> str:
        value = body.get(key)
        return value.strip()[:limit] if isinstance(value, str) else ""

    want = body.get("want")
    vocal = body.get("vocal")
    # Auto is the engine's allocation ceiling, not a length anyone asked for.
    # Written to, it makes every song as long as it can be — and render time
    # grows with length — so Auto means a typical song here.
    auto = body.get("auto") is True
    try:
        duration = int(float(body.get("duration") or TYPICAL_SECONDS))
    except (TypeError, ValueError, OverflowError):
        duration = TYPICAL_SECONDS
    return {
        "want": want if want in WANTS else "song",
        "brief": text("brief", 2000),
        "caption": text("caption", 8000),
        "lyrics": text("lyrics", 20000),
        "title": text("title", 120),
        "language": text("language", 60) or "English",
        "vocal": vocal if vocal in ("male", "female", "duet", "instrumental") else "",
        "duration": TYPICAL_SECONDS if auto else max(10, min(duration, 360)),
        "auto": auto,
    }


def user_prompt(req: dict) -> str:
    """The person's side of the conversation, from the Create page's fields."""
    instrumental = req["vocal"] == "instrumental"
    task = {
        "song": "Write a complete song: title, caption and lyrics.",
        "caption": ("Write the structured caption for the lyrics below, which "
                    "stay exactly as they are. Repeat them unchanged under "
                    "LYRICS, and give a title."),
        "lyrics": ("Write the lyrics and a title for the caption below. Repeat "
                   "the caption unchanged under CAPTION."),
        "polish": ("Polish the draft lyrics below. Keep their words, meaning "
                   "and language wherever they work; fix the section tags, "
                   "scansion, rhymes that strain, and anything the model "
                   "would sing that is not a lyric. Keep the caption unless it "
                   "contradicts the lyrics."),
    }[req["want"]]
    lines = [task, ""]
    if req["brief"]:
        lines += ["What the song is about / how it should sound:", req["brief"], ""]
    if req["caption"]:
        lines += ["Caption so far:", req["caption"], ""]
    if req["title"]:
        lines += ["Working title: " + req["title"]]
    if instrumental:
        lines += ["Voice: none — this is an instrumental. LYRICS holds only "
                  "section tags."]
    else:
        lines += ["Language of the lyrics: " + req["language"]]
        if req["vocal"] == "duet":
            lines += ["Voice: a male and female duet — say who sings which "
                      "section in Vocal Details."]
        elif req["vocal"]:
            lines += [f"Voice: {req['vocal']} lead vocal"]
    minutes, seconds = divmod(req["duration"], 60)
    length = (f"{minutes} min {seconds:02d} s" if minutes else f"{seconds} s")
    if req.get("auto"):
        lines += [f"Length: a typical song, about {length}. Do not write more "
                  "than that needs."]
    else:
        lines += [f"Length: up to {length} — a limit, not a quota."]
    if req["lyrics"] and not instrumental:
        label = {"polish": "Draft lyrics:",
                 "caption": "Lyrics (keep exactly):"}.get(
            req["want"], "Lyrics written so far (use them as a starting point):")
        lines += ["", label, req["lyrics"]]
    return "\n".join(lines).strip()


# --------------------------------------------------------------------------- #
# reading the answer
# --------------------------------------------------------------------------- #
_THINK = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.S | re.I)
_OPEN_THINK = re.compile(r"<think(?:ing)?>.*\Z", re.S | re.I)
_FENCE = re.compile(r"^\s*```[a-z]*\s*$", re.I | re.M)
_LABEL = re.compile(r"^\s*(?:#+\s*)?(?:\*\*|__)?\[?(title|caption|style|lyrics)\]?"
                    r"(?:\*\*|__)?\s*:\s*(?:\*\*|__)?\s*(.*)$"
                    r"|^\s*(?:#+\s*)?(?:\*\*|__)?(title|caption|style|lyrics)"
                    r"(?:\*\*|__)?\s*$", re.I)
_TAG = re.compile(r"^\s*\[[^\]\n]{1,40}\]\s*$")
SECTIONS = ("Global Metadata", "Vocal Details", "Arrangement")


def visible(text: str) -> str:
    """What the model said, without reasoning it thought out loud."""
    text = _THINK.sub("", text)
    return _OPEN_THINK.sub("", text)


def parse(text: str, want: str = "song") -> dict:
    """Split an answer into title, caption and lyrics.

    Tolerant on purpose: local models decorate headings with markdown, wrap
    everything in a code fence, or answer in JSON when they were not asked to.
    A heading needs its colon ("CAPTION:") or a line to itself, so a caption
    sentence that happens to begin with "Lyrics" is not mistaken for one.
    """
    text = _FENCE.sub("", visible(text)).strip()
    out = {"title": "", "caption": "", "lyrics": "", "warnings": []}
    if text.startswith("{"):
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                for k in ("title", "caption", "lyrics"):
                    if isinstance(data.get(k), str):
                        out[k] = data[k].strip()
                if not out["caption"] and isinstance(data.get("style"), str):
                    out["caption"] = data["style"].strip()
                return _tidy(out, want)
        except ValueError:
            pass
    current, bucket = None, {"title": [], "caption": [], "lyrics": []}
    for line in text.splitlines():
        m = _LABEL.match(line)
        if m:
            name = (m.group(1) or m.group(3)).lower()
            name = "caption" if name == "style" else name
            # Inside the lyrics only a new TITLE/CAPTION heading ends them.
            if current != "lyrics" or name != "lyrics":
                current = name
                if (m.group(2) or "").strip():
                    bucket[current].append(m.group(2))
                continue
        if current:
            bucket[current].append(line)
    if not any(bucket.values()):
        bucket["lyrics"] = text.splitlines()
        out["warnings"].append("The writer did not label its answer; "
                               "everything it wrote went into the lyrics.")
    out["title"] = " ".join(x.strip() for x in bucket["title"] if x.strip())
    out["caption"] = "\n".join(bucket["caption"])
    out["lyrics"] = "\n".join(bucket["lyrics"])
    return _tidy(out, want)


def _tidy(out: dict, want: str) -> dict:
    out["title"] = out["title"].strip().strip('"“”\'*').strip()[:120]
    cap = [ln.rstrip() for ln in out["caption"].strip().splitlines()]
    # Collapse runs of blank lines; the model reads the caption as text and
    # the encoder squeezes blank lines anyway.
    tidy_cap: list[str] = []
    for ln in cap:
        if ln.strip() == "" and (not tidy_cap or tidy_cap[-1] == ""):
            continue
        tidy_cap.append(ln)
    out["caption"] = "\n".join(tidy_cap).strip()[:8000]
    lines = [ln.rstrip() for ln in out["lyrics"].strip().splitlines()]
    tidy: list[str] = []
    for ln in lines:
        # A tag gets a blank line before it, so sections stay apart however
        # the model spaced them.
        if _TAG.match(ln) and tidy and tidy[-1] != "":
            tidy.append("")
        if ln == "" and tidy and tidy[-1] == "":
            continue
        tidy.append(ln.strip() if _TAG.match(ln) else ln)
    out["lyrics"] = "\n".join(tidy).strip()[:20000]
    if out["lyrics"] and not any(_TAG.match(ln) for ln in tidy):
        out["warnings"].append("The lyrics came back without [Section] tags. "
                               "Music 3 sings them anyway, but tags are how it "
                               "knows the song's structure.")
    if want in ("song", "lyrics", "polish") and not out["lyrics"]:
        out["warnings"].append("The writer returned no lyrics.")
    if want in ("song", "caption") and not out["caption"]:
        out["warnings"].append("The writer returned no caption; yours is kept.")
    elif out["caption"]:
        absent = [h for h in SECTIONS if h.lower() not in out["caption"].lower()]
        if absent:
            out["warnings"].append("The caption is missing "
                                   + ", ".join(absent) + ". It still works; "
                                   "the three sections give finer control.")
    return out


# --------------------------------------------------------------------------- #
# the wire
# --------------------------------------------------------------------------- #
def _error_text(resp: requests.Response) -> str:
    try:
        body = resp.json()
    except ValueError:
        return resp.text.strip()[:300]
    err = body.get("error") if isinstance(body, dict) else None
    if isinstance(err, dict):
        return str(err.get("message") or err)[:300]
    if isinstance(err, str):
        return err[:300]
    return str(body)[:300]


def _http_failure(conn: dict, status: int, said: str) -> LyricistError:
    where = conn["label"]
    if status in (401, 403):
        return LyricistError(f"{where} refused the API key ({status}): {said}")
    if status == 404:
        return LyricistError(f"{where} answered 404 — the address or the "
                             f"model name is wrong: {said}")
    if status == 429:
        return LyricistError(f"{where} is rate-limiting or out of credit: {said}")
    return LyricistError(f"{where} answered {status}: {said}")


def _openai_headers(conn: dict) -> dict:
    headers = {"Content-Type": "application/json"}
    if conn["key"]:
        headers["Authorization"] = "Bearer " + conn["key"]
    return headers


def _stream_openai(conn: dict, system: str, prompt: str, task, on_text) -> None:
    """Chat completions, streamed. Plain JSON back is handled too — some
    servers ignore "stream"."""
    body = {"model": conn["model"], "stream": True,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": prompt}]}
    if conn.get("provider") == "lmstudio":
        body["ttl"] = LMSTUDIO_TTL
    try:
        resp = requests.post(conn["base"] + "/chat/completions", json=body,
                             headers=_openai_headers(conn), stream=True,
                             timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
    except requests.ConnectionError:
        raise LyricistError(f"Nothing answered at {conn['base']}. Is "
                            f"{conn['label']} running?") from None
    except requests.Timeout:
        raise LyricistError(f"{conn['label']} did not answer in time.") from None
    with resp:
        if resp.status_code >= 400:
            raise _http_failure(conn, resp.status_code, _error_text(resp))
        kind = resp.headers.get("Content-Type", "")
        if "text/event-stream" not in kind:
            try:
                data = resp.json()
                on_text(data["choices"][0]["message"]["content"] or "")
            except (ValueError, KeyError, IndexError, TypeError):
                raise LyricistError(f"{conn['label']} sent an answer M3 "
                                    "Studio could not read.") from None
            return
        for raw in resp.iter_lines():
            if task.cancel:
                return
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                return
            try:
                data = json.loads(payload)
            except ValueError:
                continue
            if isinstance(data.get("error"), (dict, str)):
                raise LyricistError(f"{conn['label']} stopped: "
                                    f"{_error_text_from(data)}")
            for choice in data.get("choices") or []:
                piece = (choice.get("delta") or {}).get("content")
                if piece:
                    on_text(piece)


def _error_text_from(data: dict) -> str:
    err = data.get("error")
    return str(err.get("message") if isinstance(err, dict) else err)[:300]


def _stream_anthropic(conn: dict, system: str, prompt: str, task, on_text) -> None:
    """Claude, through the official SDK, streamed."""
    try:
        import anthropic
    except ImportError:
        raise LyricistError("Claude needs the anthropic package. Restart M3 "
                            "Studio with run.sh / run.bat so it installs "
                            "requirements.txt.") from None
    client = anthropic.Anthropic(api_key=conn["key"] or None,
                                 base_url=conn["base"],
                                 timeout=anthropic.Timeout(READ_TIMEOUT,
                                                           connect=CONNECT_TIMEOUT))
    ask = {"model": conn["model"], "max_tokens": MAX_OUTPUT_TOKENS,
           "system": system,
           "messages": [{"role": "user", "content": prompt}]}
    try:
        if conn["model"] in FALLBACK_MODELS:
            stream = client.beta.messages.stream(**ask, betas=[FALLBACK_BETA],
                                                 fallbacks="default")
        else:
            stream = client.messages.stream(**ask)
        with stream as events:
            for piece in events.text_stream:
                if task.cancel:
                    return
                on_text(piece)
            final = events.get_final_message()
    except anthropic.AuthenticationError as exc:
        raise _http_failure(conn, 401, exc.message) from None
    except anthropic.PermissionDeniedError as exc:
        raise _http_failure(conn, 403, exc.message) from None
    except anthropic.NotFoundError as exc:
        raise _http_failure(conn, 404, exc.message) from None
    except anthropic.RateLimitError as exc:
        raise _http_failure(conn, 429, exc.message) from None
    except anthropic.APIStatusError as exc:
        raise _http_failure(conn, exc.status_code, exc.message) from None
    except anthropic.APIConnectionError:
        raise LyricistError(f"Could not reach {conn['base']}.") from None
    if final.stop_reason == "refusal":
        raise LyricistError("Claude declined to write this one. Try "
                            "rewording the brief.")


def list_models(cfg: dict) -> list[str]:
    """The model names the configured server offers, for the Settings list."""
    prov = provider_of(cfg)
    base = base_of(cfg)
    if not base:
        raise LyricistError("Set the song writer's address first.")
    key, _ = key_of(cfg)
    conn = {"label": prov["label"], "base": base, "key": key}
    if prov["wire"] == "anthropic":
        try:
            import anthropic
        except ImportError:
            raise LyricistError("Claude needs the anthropic package.") from None
        if not key:
            raise LyricistError("Add the API key first.")
        try:
            client = anthropic.Anthropic(api_key=key, base_url=base,
                                         timeout=CONNECT_TIMEOUT * 2)
            return sorted(m.id for m in client.models.list(limit=100))
        except anthropic.APIStatusError as exc:
            raise _http_failure(conn, exc.status_code, exc.message) from None
        except anthropic.APIConnectionError:
            raise LyricistError(f"Could not reach {base}.") from None
    try:
        resp = requests.get(base + "/models", headers=_openai_headers(conn),
                            timeout=CONNECT_TIMEOUT * 2)
    except requests.RequestException:
        raise LyricistError(f"Nothing answered at {base}. Is {prov['label']} "
                            "running?") from None
    if resp.status_code >= 400:
        raise _http_failure(conn, resp.status_code, _error_text(resp))
    try:
        data = resp.json()
        items = data.get("data") if isinstance(data, dict) else data
        return sorted({str(m["id"]) for m in items or [] if isinstance(m, dict)
                       and m.get("id")})
    except (ValueError, TypeError, KeyError):
        raise LyricistError(f"{prov['label']} sent a model list M3 Studio "
                            "could not read.") from None


# --------------------------------------------------------------------------- #
# giving the GPU back
#
# A local writer shares the graphics card with Music 3. Ollama keeps a model in
# memory for five minutes after it answers; a 7-8B model is several GB, and
# ComfyUI, finding the card full, has even less room to stream the
# Music 3 text encoder through — the
# same song, much slower. So once a write is over, Ollama is told to let go.
# --------------------------------------------------------------------------- #
def release(conn: dict, task=None) -> None:
    if conn.get("provider") != "ollama":
        return
    root = re.sub(r"/v1/?$", "", conn["base"])
    try:
        requests.post(root + "/api/generate",
                      json={"model": conn["model"], "keep_alive": 0},
                      timeout=CONNECT_TIMEOUT)
        if task is not None:
            task.log("Asked Ollama to unload the model, so songs get the "
                     "whole GPU.")
    except requests.RequestException:
        pass


def released(fn):
    """A task body that hands the GPU back however it ends."""
    def body(task, conn: dict, req: dict) -> None:
        try:
            fn(task, conn, req)
        finally:
            release(conn, task)
    return body


# --------------------------------------------------------------------------- #
# the task
# --------------------------------------------------------------------------- #
def run(task, conn: dict, req: dict) -> None:
    """Body of the "lyrics" task. The answer lands in task.meta["result"];
    what has arrived so far is in task.meta["draft"] while it streams."""
    prompt = user_prompt(req)
    what = {"song": "a song", "caption": "a caption", "lyrics": "lyrics",
            "polish": "a polish"}[req["want"]]
    task.log(f"Asking {conn['label']} ({conn['model']}) for {what}.")
    got: list[str] = []
    size = [0]

    def on_text(piece: str) -> None:
        got.append(piece)
        size[0] += len(piece)
        if size[0] > MAX_CHARS:
            raise LyricistError("The writer kept going far past a song's "
                                "length, so it was stopped.")
        shown = visible("".join(got))
        words = len(shown.split())
        # meta is replaced, never edited in place: /api/tasks serialises it
        # from another thread.
        task.set(meta={"want": req["want"], "draft": shown[-6000:]},
                 detail=(f"Writing… {words} words" if words
                         else "Thinking…"))

    task.set(detail="Waiting for the model…")
    if conn["wire"] == "anthropic":
        _stream_anthropic(conn, SYSTEM_PROMPT, prompt, task, on_text)
    else:
        _stream_openai(conn, SYSTEM_PROMPT, prompt, task, on_text)
    if task.cancel:
        task.log("Stopped.")
        task.set(state="cancelled", detail="Stopped")
        return
    result = parse("".join(got), req["want"])
    # What the person asked to keep is kept, whatever the model did with it.
    if req["want"] in ("lyrics", "polish") and req["caption"]:
        result["caption"] = req["caption"]
    if req["want"] == "caption" and req["lyrics"]:
        result["lyrics"] = req["lyrics"]
    if req["vocal"] == "instrumental":
        result["lyrics"] = "\n\n".join(ln.strip() for ln in result["lyrics"].splitlines()
                                       if _TAG.match(ln))
    if not (result["lyrics"] or result["caption"]):
        raise LyricistError("The writer answered, but nothing in it could be "
                            "used. Try again, or pick a larger model.")
    for w in result["warnings"]:
        task.log(w)
    task.set(meta={"want": req["want"], "result": result},
             detail="Done", pct=100)
    task.log("Done.")
