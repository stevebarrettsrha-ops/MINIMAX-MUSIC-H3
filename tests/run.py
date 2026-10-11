"""Run the suite.

    python tests/run.py          everything
    python tests/run.py gate     only the syntax gate (compile + inline JS check)
    python tests/run.py units    one group: gate, units, api, ui
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from harness import Suite  # noqa: E402

PY_FILES = ["server.py", "comfy.py", "bootstrap.py", "manager.py", "lyricist.py"]


def gate() -> Suite:
    """A missing function in the inline script kills every button silently,
    so the page's JavaScript is checked as well as the Python."""
    s = Suite("gate")
    for f in PY_FILES:
        done = subprocess.run([sys.executable, "-m", "py_compile", str(ROOT / f)],
                              capture_output=True, text=True)
        s.check(f"{f} compiles", done.returncode == 0, done.stderr[-400:])
    src = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    js = "\n".join(re.findall(r"<script>(.*?)</script>", src, re.S))
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
        fh.write(js)
    try:
        done = subprocess.run(["node", "--check", fh.name], capture_output=True,
                              text=True)
        s.check("inline JavaScript parses", done.returncode == 0, done.stderr[-600:])
    except FileNotFoundError:
        print("  skip — node is not installed, the inline script is unchecked")
    # CSS that escapes its <style> block renders as text across the top of
    # the page and leaves everything below it unstyled.
    css = re.findall(r"<style>(.*?)</style>", src, re.S)
    s.check("one stylesheet, its braces balanced",
            src.count("<style>") == 1 and len(css) == 1
            and css[0].count("{") == css[0].count("}") and "<" not in css[0],
            f"{src.count('<style>')} style tags")
    ids = set(re.findall(r'id="([^"]+)"', src))
    used = set(re.findall(r'\$\("([^"]+)"\)', src))
    s.check("every element the script asks for exists", used <= ids,
            ", ".join(sorted(used - ids)))
    return s


def main() -> int:
    def group(module):
        return lambda: __import__(module).run()
    groups = {"gate": gate, "units": group("test_units"),
              "api": group("test_api"), "ui": group("test_ui")}
    wanted = sys.argv[1:] or list(groups)
    suites = [groups[w]() for w in wanted]
    passed = sum(x.passed for x in suites)
    failed = [f"{x.name}: {f}" for x in suites for f in x.failures]
    print(f"\n{passed} passed, {len(failed)} failed")
    for f in failed:
        print("  FAIL " + f)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
