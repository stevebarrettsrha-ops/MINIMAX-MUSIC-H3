"""The server end to end, against the stand-in ComfyUI."""
import json
import time

import requests

from harness import Suite, Stack

import comfy


CAPTION = ("Global Metadata\nBasic Attributes: bpm is 112. key is D, and scale "
           "is major. Indie Pop / Bedroom Pop.\n\nVocal Details\nSinger A (Female).")


def run() -> Suite:
    s = Suite("api")
    with Stack() as st:
        status = st.wait_ready()
        s.check("ready with the three files listed", status.get("ready"),
                json.dumps(status)[:400])
        s.check("status lists only the Music 3 files",
                all("music3" in n for n in status.get("dits", []) + status.get("tes", []))
                and len(status.get("dits", [])) == 2, str(status.get("dits")))
        s.check("status carries the card and the length limit",
                status.get("vram_total") == 8 * 1024 ** 3
                and status.get("max_duration") == 360
                and status.get("recommended_duration") == 120)
        s.check("mp3 and wav are offered to save as",
                "mp3" in status.get("formats", []), str(status.get("formats")))

        r = st.post("/api/generate", {"caption": "", "lyrics": "x"})
        s.check("an empty caption is refused before anything is queued",
                r.status_code == 400 and not st.prompts(), r.text)

        r = st.post("/api/generate", {"caption": CAPTION, "duration": 60,
                                      "lyrics": "[Verse]\nSun on the water\n[Chorus]\nLa la",
                                      "seed": 11, "steps": 20, "cfg_scale": 2.0,
                                      "top_k": 80, "format": "flac"})
        s.check("generate accepts the song", r.ok, r.text)
        job = st.wait_job(r.json()["jobs"][0])
        s.check("the song finishes", job["status"] == "done", json.dumps(job)[:500])
        track = job.get("track") or {}
        s.check("the track records what made it",
                track.get("seed") == 11 and track.get("steps") == 20
                and track.get("cfg_scale") == 2.0 and track.get("top_k") == 80
                and "int8" in (track.get("te") or ""), json.dumps(track)[:500])
        s.check("its length is read from the file, not the ceiling",
                abs((track.get("seconds") or 0) - 42.0) < 0.5 and not track.get("cut"),
                str(track.get("seconds")))
        s.check("the title comes from the first sung line",
                track.get("title") == "Sun On The Water", track.get("title"))
        s.check("the summary is the caption's genre",
                track.get("summary") == "Indie Pop / Bedroom Pop", track.get("summary"))
        prompt = st.prompts()[-1]["prompt"]
        enc = prompt[comfy.N_ENC]["inputs"]
        s.check("the prompt carries the chosen values",
                enc["cfg_scale"] == 2.0 and enc["top_k"] == 80
                and enc["max_duration"] == 60.0 and enc["seed"] == 11)
        lib = st.get("/api/library").json()
        s.check("the song is in the library", lib and lib[0]["id"] == track.get("id"))
        audio = st.get("/api/track/" + track["id"])
        s.check("the audio is served", audio.ok and audio.content[:4] == b"fLaC")

        r = st.post("/api/generate", {"caption": CAPTION, "seed": 100, "count": 2,
                                      "duration": 30})
        ids = r.json().get("jobs", [])
        done = [st.wait_job(i) for i in ids]
        seeds = sorted((j.get("track") or {}).get("seed") for j in done)
        s.check("two takes on a fixed seed use consecutive seeds",
                seeds == [100, 101], str(seeds))

        r = st.post("/api/generate", {"caption": CAPTION, "vocal": "instrumental",
                                      "instrumental": True,
                                      "lyrics": "[Verse]\nwords\n[Chorus]\nmore",
                                      "duration": 30})
        st.wait_job(r.json()["jobs"][0])
        enc = st.prompts()[-1]["prompt"][comfy.N_ENC]["inputs"]
        s.check("an instrumental sends the tags and none of the words",
                enc["lyrics"] == "[Verse]\n\n[Chorus]", repr(enc["lyrics"]))

        tid = track["id"]
        requests.patch(st.url + "/api/track/" + tid, json={"title": "Renamed"}, timeout=10)
        s.check("a song can be renamed",
                any(t["id"] == tid and t["title"] == "Renamed"
                    for t in st.get("/api/library").json()))
        requests.delete(st.url + "/api/track/" + tid, timeout=10)
        s.check("a song can be deleted",
                not any(t["id"] == tid for t in st.get("/api/library").json()))

        deps = st.get("/api/deps").json()["items"]
        models = next(i for i in deps if i["id"] == "models")
        s.check("the Engine page says the models are all there",
                models["state"] == "ok", json.dumps(models))
        s.check("the Engine page has a hardware row",
                any(i["id"] == "hardware" for i in deps))

        cfg = st.post("/api/config", {"comfy_url": "localhost:9", "hf_token": "x"}).json()
        s.check("config answers never carry secrets",
                "hf_token" not in json.dumps(cfg) and "llm_key" not in json.dumps(cfg))
        s.check("a bare host:port gains its scheme",
                cfg["config"]["comfy_url"] == "http://localhost:9")
        bad = st.get("/api/status", headers={"Host": "evil.example:7789"})
        s.check("only this machine is answered", bad.status_code == 403)

    with Stack(mock_env={"MOCK_DELAY": "4"}) as st:
        st.wait_ready()
        r = st.post("/api/generate", {"caption": CAPTION, "duration": 120})
        jid = r.json()["jobs"][0]
        stages = set()
        end = time.time() + 2.5
        while time.time() < end:
            for j in st.get("/api/jobs").json():
                if j["id"] == jid:
                    stages.add(j.get("stage", ""))
            time.sleep(0.15)
        s.check("the job card shows the song being written, frame by frame",
                any(x.startswith("Writing the song — ") and "of up to 120s" in x
                    for x in stages), str(stages))
        st.post(f"/api/jobs/{jid}/cancel")
        job = st.wait_job(jid)
        s.check("a song can be stopped", job["status"] == "cancelled", job["status"])

    with Stack(mock_env={"MOCK_CUT": "1"}) as st:
        st.wait_ready()
        r = st.post("/api/generate", {"caption": CAPTION, "duration": 40})
        job = st.wait_job(r.json()["jobs"][0])
        s.check("a song that runs to its limit is marked cut",
                (job.get("track") or {}).get("cut") is True, json.dumps(job)[:400])

    with Stack(mock_env={"MOCK_FAIL_NODE": "MiniMaxMusic3TextEncode"}) as st:
        st.wait_ready()
        r = st.post("/api/generate", {"caption": CAPTION, "duration": 240})
        job = st.wait_job(r.json()["jobs"][0])
        s.check("an out-of-memory failure is explained for the card",
                job["status"] == "error" and "shorter Length" in job.get("error", ""),
                job.get("error"))

    with Stack(models=False) as st:
        status = st.get("/api/status").json()
        s.check("missing files keep Create from claiming ready",
                not status["ready"] and len(status["missing_required_models"]) == 3,
                json.dumps(status)[:300])
        deps = st.get("/api/deps").json()["items"]
        models = next(i for i in deps if i["id"] == "models")
        s.check("the Engine page names the missing roles",
                models["state"] == "missing" and "text encoder" in models["detail"],
                models["detail"])

    with Stack(mock_env={"MOCK_BLANK_MODELS": "1000"}) as st:
        status = st.get("/api/status").json()
        s.check("files on disk but not listed read as a stale scan",
                status.get("stale_models") is True and not status["ready"],
                json.dumps(status)[:300])

    with Stack() as st:
        st.wait_ready()
        r = st.post("/api/lyrics/write", {"want": "song", "brief": "a song"})
        s.check("writing with no writer set up points at Settings",
                r.status_code == 409 and "Settings" in r.json().get("error", ""), r.text)
        r = st.post("/api/llm/settings", {"provider": "custom",
                                          "base": "http://10.0.0.5:8000/v1",
                                          "model": "m", "key": "sk-secretkey123"})
        s.check("a key is not sent over plain http to another machine",
                not st.get("/api/llm/settings").json()["configured"])
        view = st.get("/api/llm/settings").json()
        s.check("the key is shown only as its last four",
                "secretkey" not in json.dumps(view) and view["key_hint"] == "…y123")
    return s


if __name__ == "__main__":
    suite = run()
    raise SystemExit(1 if suite.failures else 0)
