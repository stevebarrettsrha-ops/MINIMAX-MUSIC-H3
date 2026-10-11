"""
parts.py - a song longer than the model writes in one pass, made in parts.

MiniMax Music 3 writes at most 9,000 frames (6:00) in one go: ComfyUI's node
stops there and the model's position limit sits just behind it. A longer song
is therefore several passes — each a whole number of lyric sections, each
within the limit — rendered one after another with the same caption, seed
family and settings, and joined into one track afterwards.

Everything here is arithmetic on text, so the page and the tests can rely on
it without an engine: the length a set of lyrics wants, and how to cut them
into parts that each fit.
"""

from __future__ import annotations

import math
import re

# The engine's own ceiling for one pass; the real figure is read from the
# node's schema at run time and passed in.
PASS_LIMIT = 360
# A part shorter than this is not worth a separate pass.
MIN_PART = 60
# The longest song the app will plan. Thirty minutes is five passes at the
# limit, which on an 8 GB card is most of a day; anything past it is a
# mistake, not a wish.
MAX_SONG = 1800

TAG = re.compile(r"^\s*\[[^\]\n]{1,40}\]\s*$")
INSTRUMENTAL_TAG = re.compile(
    r"^\s*\[(intro|outro|instrumental|solo|interlude|break)[^\]]*\]\s*$", re.I)
INSTRUMENTAL_LYRICS = "[Intro]\n\n[Instrumental]\n\n[Outro]"

# Seconds a line of lyrics is worth, before headroom. These are averages the
# first real songs were measured against; a slow ballad runs longer and a
# rap shorter, which the headroom is for.
PER_SUNG_LINE = 4.0
PER_INSTRUMENTAL_TAG = 12.0
PER_OTHER_TAG = 3.0
HEADROOM = 1.35


def sections(lyrics: str) -> list[list[str]]:
    """The lyrics as sections: each starts at a tag line (text before the
    first tag is a section of its own). Blank lines are dropped."""
    out: list[list[str]] = []
    for raw in (lyrics or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if TAG.match(line) or not out:
            out.append([line])
        else:
            out[-1].append(line)
    return out


def section_seconds(section: list[str], instrumental: bool) -> float:
    secs = 0.0
    for line in section:
        if TAG.match(line):
            secs += PER_INSTRUMENTAL_TAG if INSTRUMENTAL_TAG.match(line) else PER_OTHER_TAG
        elif not instrumental:
            secs += PER_SUNG_LINE
    return secs


def estimate_seconds(lyrics: str, instrumental: bool = False) -> int:
    """What Auto asks for: the lyrics' worth plus headroom, rounded up to
    half a minute, between one minute and MAX_SONG. Nothing to go on means
    the node's own default, two minutes."""
    secs = sections(lyrics)
    if not secs:
        return 120
    est = sum(section_seconds(x, instrumental) for x in secs) * HEADROOM
    return int(max(MIN_PART, min(MAX_SONG, math.ceil(est / 30) * 30)))


def plan(lyrics: str, instrumental: bool, duration: int,
         limit: int = PASS_LIMIT) -> list[dict]:
    """Cut a song of `duration` seconds into parts the engine can write.

    One part when it fits. Otherwise as few parts as the limit allows, each
    a run of whole sections with as equal a share of the song as the
    sections permit, and each given a ceiling of its share of the duration
    (never above the limit). An instrumental with no sections is repeated
    structure. Lyrics with fewer sections than parts are cut as far as the
    sections go, and the last part carries what is left.

    Returns [{"lyrics", "seconds", "index", "count"}].
    """
    duration = max(1, min(int(duration), MAX_SONG))
    limit = max(MIN_PART, min(int(limit), PASS_LIMIT))
    count = max(1, math.ceil(duration / limit))
    secs = sections(lyrics)
    if count == 1:
        return [{"lyrics": "\n\n".join("\n".join(x) for x in secs),
                 "seconds": duration, "index": 1, "count": 1}]
    if not secs:
        each = min(limit, math.ceil(duration / count / 30) * 30)
        return [{"lyrics": INSTRUMENTAL_LYRICS, "seconds": each,
                 "index": i + 1, "count": count} for i in range(count)]
    count = min(count, len(secs))
    weights = [section_seconds(x, instrumental) or 1.0 for x in secs]
    total = sum(weights)
    groups: list[list[list[str]]] = []
    current: list[list[str]] = []
    acc = 0.0
    for i, (section, w) in enumerate(zip(secs, weights)):
        left_parts = count - len(groups)
        left_secs = len(secs) - i
        # Close the group once it holds its share, as long as every part
        # still to come can have at least one section.
        if (current and acc >= total / count * (len(groups) + 1) - total / count / 2
                and left_parts > 1 and left_secs >= left_parts):
            groups.append(current)
            current = []
        current.append(section)
        acc += w
    if current:
        groups.append(current)
    while len(groups) > count:           # never more parts than planned
        groups[-2].extend(groups.pop())
    out = []
    for i, group in enumerate(groups):
        share = sum(section_seconds(x, instrumental) for x in group) / total
        ceiling = math.ceil(duration * share * 1.1 / 30) * 30
        out.append({"lyrics": "\n\n".join("\n".join(x) for x in group),
                    "seconds": int(max(MIN_PART, min(limit, ceiling))),
                    "index": i + 1, "count": len(groups)})
    return out


def part_caption(caption: str, index: int, count: int) -> str:
    """The caption for one pass of a song in parts.

    The model has no notion of continuing an earlier pass, so it is told in
    words what this pass is: the same song, the same key, tempo and voice,
    and where it sits — no intro after the first part, no ending before the
    last. Words steer the model rather than bind it; the join afterwards
    is a crossfade, not a bar-exact splice.
    """
    if count <= 1:
        return caption
    note = (f"This is part {index} of {count} of one continuous song: keep the "
            "same key, tempo, singer, instruments and production throughout.")
    if index > 1:
        note += " It starts straight in on its first section, with no intro."
    if index < count:
        note += " It ends ready to continue, without an outro or a final ending."
    return (caption.rstrip() + "\n\n" + note) if caption.strip() else note
