"""The page in a real browser (Playwright + Chromium), against the stand-in.

Skipped, not failed, when Playwright is not installed:
    pip install -r requirements-dev.txt && python -m playwright install chromium
"""
import json
import os

from harness import Suite, Stack


def run() -> Suite:
    s = Suite("ui")
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("  skip — Playwright is not installed")
        return s
    with Stack() as st, sync_playwright() as pw:
        st.wait_ready()
        exe = "/opt/pw-browsers/chromium" if os.path.exists("/opt/pw-browsers/chromium") else None
        try:
            browser = pw.chromium.launch(**({"executable_path": exe} if exe else {}))
        except Exception:
            browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.goto(st.url)
        page.wait_for_function("document.querySelector('#enginePill span')"
                               ".textContent === 'Engine ready'", timeout=20000)
        s.check("the page loads with no script errors", not errors, "; ".join(errors))
        s.check("the setup sheet stays closed on a set-up install",
                page.is_hidden("#setupVeil"))

        page.click('.nav[data-view="create"]')
        s.check("the 8 GB card hint is shown", page.is_visible("#cardHint")
                and "8 GB card" in page.inner_text("#cardHint"))
        dits = page.eval_on_selector_all("#dit option", "o => o.map(x => x.value)")
        s.check("the diffusion picker lists the Music 3 files only",
                dits == ["", "minimax_music3_dit_fp16.safetensors",
                         "minimax_music3_dit_int8_convrot.safetensors"], str(dits))
        page.select_option("#preset", index=1)
        cap = page.input_value("#caption")
        s.check("a preset fills the caption", cap.startswith("Global Metadata"))
        s.check("the caption counter sees three sections",
                "structured" in page.inner_text("#capCount"))
        page.click("#styleChips .chip >> nth=0")
        tag = page.eval_on_selector("#styleChips .chip", "b => b.textContent")
        s.check("a tag chip writes a Tags line and lights up",
                ("Tags: " + tag) in page.input_value("#caption")
                and "on" in page.get_attribute("#styleChips .chip >> nth=0", "class"))
        page.click("#styleChips .chip >> nth=0")
        s.check("pressing it again takes it out",
                "Tags:" not in page.input_value("#caption"))
        page.fill("#lyrics", "Sun on the water")
        page.wait_for_selector("#lyricsNote:not([hidden])")
        s.check("untagged lyrics get a note about structure",
                "Section" in page.inner_text("#lyricsNote"))
        page.fill("#lyrics", "")
        page.click("#btnStructure")
        s.check("the structure button inserts tags",
                "[Chorus]" in page.input_value("#lyrics"))
        page.fill("#lyrics", "[Verse]\nSun on the water\n\n[Chorus]\nLa la la")
        page.click('[role="tab"][data-tab="advanced"]')
        s.check("Advanced opens More Options", page.is_visible("#segVocal"))
        page.click('#segVocal button[data-v="male"]')
        page.select_option("#duration", "30")
        page.click("#btnCreate")
        page.wait_for_selector("#trackList .trow", timeout=40000)
        s.check("Create makes a song that lands in the workspace",
                "Sun On The Water" in page.inner_text("#trackList"))
        prompt = st.prompts()[-1]["prompt"]
        enc = prompt["13"]["inputs"]
        s.check("the page sends what is on screen",
                enc["max_duration"] == 30.0 and "[Chorus]" in enc["lyrics"]
                and "Global Metadata" in enc["caption"], json.dumps(enc)[:300])

        page.click("#trackList .trow [data-act='view']")
        s.check("a song's caption and lyrics can be read back",
                page.is_visible("#viewVeil")
                and "Global Metadata" in page.inner_text("#viewCaption"))
        page.click("#btnViewReuse")
        s.check("loading a song's settings keeps its seed",
                page.input_value("#seed") == str(st.get("/api/library").json()[0]["seed"]))

        page.click("#navSettings")
        page.wait_for_function("!document.querySelector('#btnSaveCfg').disabled")
        s.check("settings list the song writer's services",
                "Ollama" in page.inner_text("#llmProvider"))
        page.keyboard.press("Escape")

        page.click('.nav[data-view="engine"]')
        page.wait_for_selector("#depList .fitem")
        s.check("the Engine page lists the hardware",
                "Graphics card" in page.inner_text("#depList"))
        page.click('.nav[data-view="models"]')
        page.wait_for_selector("#curatedList .fitem")
        s.check("the Models page offers the Comfy-Org files",
                "minimax_music3_dav.safetensors" in page.inner_text("#curatedList"))
        page.set_viewport_size({"width": 420, "height": 860})
        page.click('.nav[data-view="create"]')
        width = page.evaluate("document.documentElement.scrollWidth")
        s.check("no sideways scroll on a phone", width <= 420, str(width))
        s.check("no script errors along the way", not errors, "; ".join(errors))
        browser.close()
    return s


if __name__ == "__main__":
    suite = run()
    raise SystemExit(1 if suite.failures else 0)
