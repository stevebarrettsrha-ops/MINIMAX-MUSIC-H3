"""Refresh tests/object_info.json from a real, running ComfyUI.

    python tests/capture_object_info.py http://127.0.0.1:8188

Keeps only the nodes the Music 3 graph uses, and swaps each loader's file list
for the Music 3 files a set-up install has (plus one unrelated file, so a
filter that forgets to filter is caught). Everything else — input names,
types, defaults, limits, combo shapes — stays exactly as the server sent it.
"""
import json
import pathlib
import sys

import requests

NODES = ["UNETLoader", "CLIPLoader", "VAELoader", "KSampler",
         "MiniMaxMusic3TextEncode", "EmptyMiniMaxMusic3LatentAudio",
         "ConditioningZeroOut", "VAEDecodeAudio", "VAEDecodeAudioTiled",
         "SaveAudioAdvanced", "SaveAudio", "LoadAudio", "AudioConcat"]
FILES = {
    ("UNETLoader", "unet_name"): [
        "flux1-dev.safetensors",
        "minimax_music3_dit_fp16.safetensors",
        "minimax_music3_dit_int8_convrot.safetensors"],
    ("CLIPLoader", "clip_name"): [
        "minimax_music3_text_encoder_pruned_int8_convrot.safetensors",
        "t5xxl_fp16.safetensors"],
    ("VAELoader", "vae_name"): [
        "ae.safetensors", "minimax_music3_dav.safetensors"],
    # LoadAudio lists ComfyUI's input folder, which is empty on a fresh
    # install: the parts of a long song are uploaded there to be joined.
    ("LoadAudio", "audio"): [],
}


def main(url: str) -> None:
    info = requests.get(url.rstrip("/") + "/object_info", timeout=60).json()
    missing = [n for n in NODES if n not in info]
    if missing:
        sys.exit("That ComfyUI lacks: " + ", ".join(missing))
    out = {n: info[n] for n in NODES}
    for (node, field), names in FILES.items():
        spec = out[node]["input"]["required"][field]
        if isinstance(spec[0], list):
            spec[0] = names
        else:
            spec[1]["options"] = names
    path = pathlib.Path(__file__).with_name("object_info.json")
    path.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"wrote {path} ({len(out)} nodes)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8188")
