# Third-party notices and credits

## YuE Studio

Most of M3 Studio's plumbing comes from YuE Studio
(<https://github.com/stevebarrettsrha-ops/Yue-2---Suno-Like-Music>), by the same
owner: the installer (`bootstrap.py`), dependency and HuggingFace handling
(`manager.py`), the chat-model connection in `lyricist.py`, the engine
supervision in `server.py`, and the page's layout and player. The parts
specific to YuE2 were removed (ABC scores, melody plans, covers, SheetSage2),
and the graph, model catalogue and song writer were rewritten for MiniMax
Music 3.

## ComfyUI-MiniMax-M3-SongPlanner

The song writer's two-part idea comes from benjiyaya's ComfyUI node
(<https://github.com/benjiyaya/ComfyUI-MiniMax-M3-SongPlanner>, MIT): expand an
idea into a structured caption, then write lyrics that fit it, with the
song's structure scaled to its length. No code from it is used, and the
prompts in `lyricist.py` were written for this app.

## MiniMax Music 3 and ComfyUI

The caption's three-section shape (Global Metadata, Vocal Details,
Arrangement) and the lyric section tags are the ones documented in the
MiniMax Music 3 model card and repository
(<https://github.com/MiniMax-AI/MiniMax-Music3>). The generation graph mirrors
the node layout of ComfyUI's "Text to Music (MiniMax Music 3)" blueprint
(<https://github.com/Comfy-Org/ComfyUI>), including its default steps, cfg,
sampler, scheduler and tiled-decode sizes. M3 Studio builds that graph from
the running engine's own schema and does not copy ComfyUI code.

The caption presets in `web/index.html` were written for this app. They are
not taken from MiniMax's demo page or its caption-rewriter templates.

The MiniMax Music 3 weights are not part of this repository. They are
downloaded on request from `Comfy-Org/MiniMax-Music-3` and are covered by the
MiniMax-Music3 Community License
(<https://huggingface.co/MiniMaxAI/MiniMax-Music3/blob/main/LICENSE>).
