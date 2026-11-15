"""Map the watcher's output of a record into a PDF deck: one slide per movie, two fixed templates, no text written by hand.

    PYTHONPATH=src python tools/watcher_deck.py experiments/exp38_gap_function --out presentation/plexus_exp38.pdf
    python tools/watcher_deck.py <record> --pane plexus,math          # two slides per movie: the plexus pane, then the math pane
    python tools/watcher_deck.py <record> --pane-for 0014=math --steps 3,4,14

TEMPLATES (the only two; the geometry is a constant of this file, so every slide of every deck has the same columns)
    two columns   the step's movie on the left, centred in its column both ways; on the right ONE watcher pane, top-aligned at
                  the same height on every slide: `plexus`, `math`, `spec` or `hierarchy`
    one movie     a step with no spec to read (a measurement), or `--pane none`: the movie centred in the whole content area

WHERE EVERY WORD COMES FROM. The slide title is the pane's own headline (the story's headline when one is saved, else the spec's
name); the pane text is the watcher's pane, typeset; the spec pane is the spec file as recorded. Nothing is typed here.

`--frames-before <tex>` inserts hand-written frames (the title page, a figure slide) before the first generated slide; they use this file's
macros (wdcanvas, wdpane, wdmovie, wdfig) so they keep the two templates' geometry and type sizes. The file sets the page counter to 1
when its first frame is the title page (the preamble starts the count at 2). `--steps none` generates no movie slide.

`--drop learnables` leaves a plexus-pane section out; `--eq-budget 80` keeps equations on one line up to 80 source characters (default 36).

NOTES (`--notes <yaml>`, optional): the one place words are typed, a YAML beside the deck that the session writes after reading the
record. `steps: {"0025": {title: "big picture : specific", insights: "plain text"}}` (or `slides:`) replaces the slide title and adds an INSIGHT
block at the bottom of the right column (yellow heading, plain body, never cut; the pane above it is clipped to the height that is left; a step
with no pane gets the insights alone on the right); `intro: {title, body}` and `summary: {title, body}` are text slides before the first movie
and after the last (a body line that ends with ":" is a yellow heading, a line that starts with "- " a bullet; `sections: [{heading, lines}]`
is read as the same), scaled down when taller than the slide. A step with no spec may carry `pane: {entities: [[name, count, note, code]], dynamics: {note, rows}}`, a plexus-style pane typed in the notes. A step may also carry `trend: up|right|down` (or progress / status quo / regression): a white, regular arrow after the insights heading, and `trend_why:` the few words (five) that say why, after the arrow. Without `--notes` nothing changes. Keys are the record's step file numbers.

THE PANES ARE THE WATCHER'S, NOT REDONE. Each pane is asked of the watcher's own routes (`/api/watch/plexus`, `/math`,
`/hierarchy`): a pane already saved beside its step (`<record>/story/NNNN.plexus.html`, `.math.html`) is read back, a missing one
is made by the watcher -- which launches the story agent, `tools/spec_story.py`, and the hierarchy tool,
`tools/spec_hierarchy.py` -- and this tool waits for it. A watcher on `--watcher` (default http://127.0.0.1:8835) is used when it
answers; otherwise one is started on a free port for the run and stopped after it. `--no-make` reads what is saved and asks nothing.

ONE FIXED TYPE SIZE PER PANE KIND, ON EVERY SLIDE OF EVERY DECK, AND ONE SLIDE PER MOVIE: SCALE below gives each kind's scale, the pane column is
COL_R wide, and what does not fit its height is cut off the bottom and reported (`TRUNCATED <step> <pane>: n % of its height`). Plexus text is
6 pt at the default; math 5.4 pt; spec 4.8 pt, so a spec of 300 lines shows its first ~70. `--scale math=0.5` changes a kind's scale for one run.
The plexus pane shows the entities and the dynamics (name, kind glyph, count), the dynamics in one column unless the slide needs two to fit;
`--full` adds the story, each step's one-line description and its meta line, as the watcher shows them.

Movies are copied to presentation/Movies/<record>_<step>.mp4 with their first frame as the poster PNG, the deck convention
(a `run:` link annotation that pdfpc plays). Pane images land in presentation/<deck>_assets/.
"""
from __future__ import annotations

import argparse
import base64
import glob
import hashlib
import html as htmllib
import json
import math
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from html.parser import HTMLParser

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
PRES = os.path.join(ROOT, "presentation")
sys.path.insert(0, os.path.join(ROOT, "tools"))

# --- the two templates' geometry, in cm of a 16 x 9 cm slide (beamer aspectratio=169); the Janelia theme's title band is the top 1 cm
# and its footer bar the bottom 0.25 cm, with the slide number above it
MARGIN, TOP, HEIGHT = 0.5, 1.3, 7.1           # right margin; top of the content area below the band; its height (ends 0.6 cm above the bottom edge, clear of the footer text)
COL_R, GAP = 7.0, 0.4                         # the pane column's width and the gap between the columns; the movie column is what is left
COL_L = 16 - MARGIN - COL_R - GAP / 2          # the left panel runs from the slide's left edge to the middle of the gap; the movie is centred in it
DYN_ROW_PT = 11.5                              # one row of the plexus pane's dynamics table before the deck's scale: the 10.5 pt line of the 9 pt body plus the 1 pt of its row end
ASSETS_DIR = ""                                # <deck>_assets, set by main
NO_FOOT = False                                # --no-foot: no record/step line at the bottom left
INS_Y = "\\wdiy"                               # where the insights block starts: the bottom of the column (\wdiy) or, --insights-under, one blank line under the pane
PANES = ("plexus", "math", "spec", "hierarchy")
ENTITIES_WORD = "states"                                        # the plexus pane's first heading (`--entities-word states` prints "states": the deck of Cedric's talk calls the state of the sets "states")
SCALE = {"plexus": 0.66, "math": 0.60, "spec": 0.69}          # the FIXED scale of each pane kind's box on every slide of every deck (`--scale kind=value` changes it)
TREND = {"up": r"\ensuremath{\uparrow}", "progress": r"\ensuremath{\uparrow}", "right": r"\ensuremath{\rightarrow}", "status quo": r"\ensuremath{\rightarrow}", "down": r"\ensuremath{\downarrow}", "regression": r"\ensuremath{\downarrow}"}      # --notes `trend:` of a step -> the arrow after the insights heading (white, regular): up = progress, right = status quo, down = regression
NATURAL_PT = {"plexus": 9.0, "math": 9.0, "spec": 7.0}        # the point size of the kind's text before that scale: the slide text is NATURAL_PT x SCALE
COL_R_PT = COL_R * 72.27 / 2.54                                # the pane column's width in points; a pane is set at COL_R_PT / SCALE and then scaled by SCALE
BOOKKEEPING = {"general.name", "general.headline", "general.title", "general.n_frames", "general.record_cap"}   # keys of a run that are not sweep parameters
DROP_COMPACT = {"ps-story", "ps-chg", "ps-w", "ps-m"}   # classes the compact plexus pane leaves out (the prose and the per-step lines under a name)
WATCHER = "http://127.0.0.1:8835"


# ---------------------------------------------------------------- the watcher
class Watcher:
    """The watcher's HTTP routes, and the process behind them when this tool had to start one."""

    def __init__(self, base: str, rec: str):
        self.base, self.rec, self.proc = base.rstrip("/"), rec, None

    def get(self, path: str, timeout: float = 90.0) -> bytes:
        with urllib.request.urlopen(self.base + path, timeout=timeout) as r:
            return r.read()

    def alive(self) -> bool:
        try:
            self.get("/api/watch/state", timeout=8.0)
            return True
        except Exception:                                        # noqa: BLE001
            return False

    def ensure(self) -> None:
        if self.alive():
            print(f"[deck] watcher at {self.base}", flush=True)
            return
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        self.base = f"http://127.0.0.1:{port}"
        env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src") + os.pathsep + os.environ.get("PYTHONPATH", ""))
        self.proc = subprocess.Popen([sys.executable, os.path.join(ROOT, "Plexus_gui.py"), "--host", "127.0.0.1", "--port", str(port)],
                                     cwd=ROOT, env=env, stdout=open(f"/tmp/watcher_deck_{port}.log", "w"), stderr=subprocess.STDOUT)
        print(f"[deck] no watcher answered; started one on port {port} (log /tmp/watcher_deck_{port}.log)", flush=True)
        t0 = time.time()
        while time.time() - t0 < 300:
            if self.alive():
                return
            if self.proc.poll() is not None:
                raise SystemExit("the watcher this tool started has exited; see its log")
            time.sleep(3)
        raise SystemExit("the watcher did not come up in 300 s")

    def stop(self) -> None:
        if self.proc is not None:
            self.proc.terminate()

    def pane(self, kind: str, i: int) -> dict | None:
        """One request: the pane's JSON, or None when the route is still working on it (`busy`)."""
        q = f"?i={i}&rec={urllib.parse.quote(self.rec)}"
        j = json.loads(self.get(f"/api/watch/{kind}{q}"))
        return None if j.get("busy") else j


def gather(w: Watcher, wanted: list[tuple[str, int]], make: bool, wait: float) -> dict:
    """{(kind, i): pane json} for every wanted pane. All requests go out together each round (the watcher works on them in
    background jobs); a pane `pending` -- the story agent still writing -- is asked again until it lands or `wait` seconds pass."""
    out, todo, t0 = {}, list(wanted), time.time()
    while todo:
        again = []
        for kind, i in todo:
            try:
                j = w.pane(kind, i)
            except Exception as e:                               # noqa: BLE001
                print(f"[deck] {kind} pane of row {i}: {type(e).__name__}: {e}", flush=True)
                j = {"text": '<div class="ps-g">the watcher could not draw this pane</div>', "pending": False}
            if j is None or (j.get("pending") and make and time.time() - t0 < wait):
                again.append((kind, i))
                continue
            if j.get("pending"):
                print(f"[deck] {kind} pane of row {i} is still being written; using what the watcher shows now", flush=True)
            out[(kind, i)] = j
        todo = again
        if todo:
            print(f"[deck] waiting for {len(todo)} pane(s) ({int(time.time() - t0)} s)", flush=True)
            time.sleep(6)
    return out


# ---------------------------------------------------------------- HTML of a pane -> a small tree
class Node:
    def __init__(self, tag: str, attrs=None, text: str = ""):
        self.tag, self.attrs, self.text, self.kids = tag, dict(attrs or []), text, []

    def cls(self) -> set:
        return set((self.attrs.get("class") or "").split())

    def hidden(self) -> bool:
        return "display:none" in (self.attrs.get("style") or "").replace(" ", "")

    def plain(self) -> str:
        if self.tag == "#text":
            return self.text
        return "" if self.hidden() else "".join(k.plain() for k in self.kids)

    def walk(self):
        """Every visible descendant, a hidden subtree (a collapsed explanation) skipped whole."""
        for k in self.kids:
            if k.tag != "#text" and not k.hidden():
                yield k
                yield from k.walk()


class _Parser(HTMLParser):
    VOID = {"img", "br", "hr", "input", "meta", "link"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = Node("root")
        self.stack = [self.root]

    def handle_starttag(self, tag, attrs):
        n = Node(tag, attrs)
        self.stack[-1].kids.append(n)
        if tag not in self.VOID:
            self.stack.append(n)

    def handle_endtag(self, tag):
        for k in range(len(self.stack) - 1, 0, -1):
            if self.stack[k].tag == tag:
                del self.stack[k:]
                return

    def handle_data(self, data):
        self.stack[-1].kids.append(Node("#text", text=data))


def parse(text: str) -> Node:
    p = _Parser()
    p.feed(re.sub(r"^<!--key[^>]*-->\n?", "", text))
    return p.root


# ---------------------------------------------------------------- tex helpers
_ESC = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "\u00a0": "~", "\u2009": r"\,", "\u2212": "-", "\u00b7": r"\textperiodcentered{}"}
_UNKNOWN: set = set()


def esc(s: str) -> str:
    out = []
    for ch in re.sub(r"\s+", " ", s):
        if ch in _ESC:
            out.append(_ESC[ch])
        elif ord(ch) < 0x2FF or ch in "\u2013\u2014\u2018\u2019\u201c\u201d\u2026\u2192\u2190\u2248\u2265\u2264\u2260\u00d7\u00b1\u2206\u2202\u221e\u2032":
            out.append(ch)
        else:
            _UNKNOWN.add(ch)
            out.append("?")
    return "".join(out)


def split_tex(tex: str, budget: int) -> str:
    """One equation as a short stack of lines: cut at its top-level `\\qquad`, `\\quad` and `+` / `-` between terms (outside braces and
    brackets), the pieces packed greedily into lines of about `budget` characters of source; a piece that is longer stays whole.
    The algorithm of presentation/make_exp29_deck.py:split_tex, which the older decks use. An equation that is ONE single-column `array` (the story's
    own rows, `row \\\\[4pt] row ...`) is split row by row: the budget counts each row on its own, never across a row end."""
    return "\\begin{array}{@{}l@{}}" + " \\\\ ".join(tex_line_list(tex, budget)) + "\\end{array}"


def tex_line_list(tex: str, budget: int) -> list:
    """The lines `split_tex` stacks: one list entry per typeset line (a story's single-column array is cut at its rows first, then each row packed on its own)."""
    m = re.fullmatch(r"\s*\\begin\{array\}\{(?:@\{\}\s*)?[lc](?:\s*@\{\})?\}(.*)\\end\{array\}\s*", tex, re.S)
    if m:
        rows = [r for r in _split_rows(m.group(1)) if r.strip()]
        if len(rows) > 1:
            return [ln for r in rows for ln in _tex_lines(r.strip(), budget)]
    return _tex_lines(tex, budget)


def eq_stack(tex: str, budget: int) -> str:
    """An equation as typeset lines, EACH LINE in its own box scaled down to 0.72 of the pane's width when it is wider (never up): only an over-wide line shrinks,
    the others of the same equation keep their size."""
    cell = lambda ln: "\\adjustbox{max width=\\dimexpr0.72\\wdw\\relax}{$\\displaystyle " + ln + "$}"      # noqa: E731
    lines = tex_line_list(tex, budget)
    if len(lines) == 1:
        return cell(lines[0])
    return "\\begin{tabular}{@{}l@{}}" + " \\\\[1pt] ".join(cell(ln) for ln in lines) + "\\end{tabular}"


def _split_rows(body: str) -> list:
    """The rows of an array body: cut at each `\\\\` (with its optional `[4pt]`) that is outside every brace group and every nested `\\begin ... \\end`."""
    rows, cur, depth, env, i = [], "", 0, 0, 0
    while i < len(body):
        if body.startswith("\\\\", i) and depth == 0 and env == 0:
            m = re.match(r"\\\\(?:\[[^\]]*\])?", body[i:])
            rows.append(cur)
            cur, i = "", i + m.end()
            continue
        m = re.match(r"\\(begin|end)\{", body[i:])
        if m:
            env += 1 if m.group(1) == "begin" else -1
        elif body[i] == "\\" and i + 1 < len(body):             # `\\{`, `\\}`, `\\,` ...: an escaped character, never a brace
            cur += body[i:i + 2]
            i += 2
            continue
        depth += (body[i] == "{") - (body[i] == "}")
        cur += body[i]
        i += 1
    rows.append(cur)
    return rows


def _tex_lines(tex: str, budget: int) -> list:
    """The lines of one equation row (or of a whole equation without rows): the packing of `split_tex`."""
    segs, cur, depth, par, i = [], "", 0, 0, 0
    while i < len(tex):
        c = tex[i]
        if c == "\\" and i + 1 < len(tex) and tex[i + 1] in "{}":
            cur += tex[i:i + 2]
            i += 2
            continue
        m = re.match(r"\\(left|right|bigl|bigr|Bigl|Bigr|biggl|biggr|Biggl|Biggr|begin|end)(?![a-zA-Z])", tex[i:])   # `\left| ... \right|`, `\Bigl( ... \Bigr)` and `\begin{cases} ... \end{cases}` are one term: a cut inside leaves them unmatched
        if m:
            par += 1 if m.group(1) in ("left", "bigl", "Bigl", "biggl", "Biggl", "begin") else -1
            cur += m.group(0)
            i += m.end()
            continue
        if depth == 0 and par == 0:
            m = re.match(r"\\(qquad|quad)(?![a-zA-Z])", tex[i:])
            if m:
                segs.append(cur)
                cur = ""
                i += m.end()
                continue
            if c in "+-" and i > 0 and tex[i - 1] not in "^_" and cur.strip() and not cur.rstrip().endswith(("=", "(", "[", "\\leftarrow", ",")):
                segs.append(cur)
                cur = c
                i += 1
                continue
        depth += (c == "{") - (c == "}")
        par += (c in "([") - (c in ")]")
        cur += c
        i += 1
    segs.append(cur)
    size = lambda t: len(re.sub(r"\\[a-zA-Z]+", "x", t))                # noqa: E731
    lines = [""]
    for sg in (x.strip() for x in segs if x.strip()):
        if lines[-1] and size(lines[-1]) + size(sg) > budget:
            lines.append(sg)
        else:
            lines[-1] = (lines[-1] + "\\;" + sg) if lines[-1] else sg
    return lines


# ---------------------------------------------------------------- a pane's HTML -> tex
def entity_code_row(code_tex: str) -> str:
    """The grey state list of an entity as a row of its own across the three columns of the entity table: one line, 8 pt, indented 1.5 em, scaled down to the table's width when it is longer."""
    return ("\\multicolumn{3}{@{}l@{}}{\\hspace*{1.5em}\\adjustbox{max width=\\dimexpr\\wdw-1.5em\\relax}{\\fontsize{8}{9}\\selectfont " + code_tex + "}}\\\\[3pt]")


class PaneTex:
    """The watcher's plexus / math pane HTML, typeset. `fetch(src)` returns a local PNG for an image address of the watcher;
    `eqs` is the story's list of equations (tex) in the pane's order, so an equation is typeset as maths and not shown as the watcher's
    bitmap; `compact` leaves out the prose and the lines under each step's name."""

    def __init__(self, fetch, compact: bool, eqs: list | None):
        self.fetch, self.compact, self.eqs = fetch, compact, eqs
        self.headline, self.skip, self.after_eq = "", False, False
        self.two = False                  # compact plexus pane: the dynamics in two columns instead of one
        self.skip_rows: list = []         # lower-case pieces of a dynamics row's text: a row that holds one is left out (the slide's `skip_rows` in the notes)
        self.drop_sections: tuple = ()    # lower-case starts of section headings left out of the plexus pane with their content (`--drop learnables`)
        self.eq_budget = 36               # characters of source per line of a stacked equation in the math pane (`--eq-budget`)
        self.in_head, self.n_head = False, 0

    # inline
    def inline(self, n: Node) -> str:
        if n.tag == "#text":
            return esc(n.text)
        if n.hidden():
            return ""
        t, c = n.tag, n.cls()
        if t == "br":
            return r"\\ "
        if t == "img":
            return self.image(n)
        if "ps-sw" in c:
            return ""
        inner = "".join(self.inline(k) for k in n.kids)
        if "ps-g" in c:
            return "{" + ("\\wdbody\\mdseries" if self.in_head else "") + "\\color{wdgray}" + inner + "}"
        if (c & {"ps-nm", "ps-sy", "ps-en"} and not self.compact) or t in ("b", "strong"):
            return "\\textbf{" + inner + "}"
        if t == "code":
            return "\\texttt{" + inner + "}"
        return inner

    def image(self, n: Node) -> str:
        src = n.attrs.get("src") or ""
        path = self.fetch(src)
        if not path:
            return ""
        if "kind=" in src:                                       # a kind's glyph, as tall as a line of the step's name
            return f"\\raisebox{{-0.35em}}[0pt][0pt]{{\\includegraphics[height=1.5em]{{{path}}}}}"
        w = float(n.attrs.get("width") or 0) * 0.72              # the watcher shows its 200-dpi bitmaps at half their pixels: 0.72 pt per shown pixel at 12 pt
        return f"\\includegraphics[width={w:g}pt]{{{path}}}" if w else f"\\includegraphics{{{path}}}"

    def cell(self, td: Node) -> str:
        """A table cell: its inline text, each block inside it (a one-line description, a meta line) on a line of its own."""
        pad = re.search(r"padding-left:\s*(\d+)px", td.attrs.get("style") or "")
        parts = [f"\\hspace*{{{int(pad.group(1)) * 0.5:g}pt}}"] if pad and int(pad.group(1)) else []
        grey_open = False
        for k in td.kids:
            if "ps-en" in td.cls() and k.tag == "span" and "ps-g" in k.cls() and not grey_open:
                grey_open = True                                  # an entity's grey text (its code name, its states) is a paragraph of its own, indented
                parts.append("\\par{\\leftskip=1.5em\\relax " + self.inline(k))
                continue
            if k.tag == "div":
                if k.hidden() or (self.compact and k.cls() & DROP_COMPACT):
                    continue
                s = self.inline(k)
                if not s.strip():
                    continue
                size = "\\footnotesize\\color{wdgray}" if "ps-m" in k.cls() else ""
                parts.append("\\\\ {" + size + s + "}")
            else:
                parts.append(self.inline(k))
        if grey_open:
            parts.append("\\par}")
        s = "".join(parts)
        return ("\\textbf{" + s + "}") if "ps-en" in td.cls() and not self.compact and not s.startswith("\\textbf") else s

    # blocks
    def block(self, n: Node) -> str:
        if n.tag == "#text" or n.hidden() or self.after_eq:
            return ""
        c = n.cls()
        if n.tag == "table":
            if self.skip:
                return ""
            if self.compact and "ps-mq" in c:                   # the equations; the symbol legend under them is `--full`
                self.after_eq = True
                return self.table(n)
            return self.table(n)
        if n.tag != "div":
            return "".join(self.block(k) for k in n.kids)
        if "ps-hl" in c:
            self.headline = n.plain().strip()
            return ""
        if "ps-h" in c:
            head = n.plain().strip().lower()
            self.skip = (self.compact and head.startswith(("rendering", "words"))) or (bool(self.drop_sections) and head.startswith(self.drop_sections))
            if self.skip:
                return ""
            self.in_head = True
            txt = self.inline(n)
            self.in_head = False
            gap = "\\vspace{14pt}" if self.n_head else ""                       # a blank between one section and the next
            self.n_head += 1
            if head.startswith(("entities", "states")):            # a pane saved before 2026-11-14 says "entities"
                txt = txt.replace("entities", ENTITIES_WORD, 1).replace("states", ENTITIES_WORD, 1)
            return gap + "{\\wdhead\\color{wdyellow}" + txt + "}\\par\\vspace{7pt}\n"      # and one between a heading and its rows
        if self.skip or (self.compact and c & DROP_COMPACT):
            return ""
        if c & {"ps", "ps-mp", "ps-q"} or not c:
            return "".join(self.block(k) for k in n.kids)
        s = self.inline(n)
        if "ps-g" in c:
            s = "{\\color{wdgray}" + s + "}"
        elif "ps-chg" in c:
            s = "{\\color[HTML]{79B8FF}" + s + "}"
        return s + "\\par\\vspace{4pt}\n" if s.strip() else ""

    def table(self, n: Node) -> str:
        rows = [[td for td in tr.kids if td.tag in ("td", "th") and not td.hidden()] for tr in n.walk() if tr.tag == "tr"]
        rows = [r for r in rows if r]
        if not rows:
            return ""
        c = n.cls()
        if "ps-mq" in c:
            return self.math_table(rows)
        if "ps-e" in c:                                          # entities: name (with its code name and states), count, note
            spec, ncol = r"@{}>{\raggedright\arraybackslash}p{0.52\wdw}@{\hspace{5pt}}r@{\hspace{6pt}}>{\raggedright\arraybackslash}p{0.22\wdw}@{}", 3
        elif "ps-i" in rows[0][0].cls():                         # dynamics: a kind's glyph, then the step
            spec, ncol = (r"@{}l@{\hspace{3pt}}>{\raggedright\arraybackslash}p{0.42\wdw}@{}" if self.two else r"@{}l@{\hspace{4pt}}>{\raggedright\arraybackslash}p{0.86\wdw}@{}"), 2
        else:
            ncol = max(len(r) for r in rows)
            spec = "@{}" + "l@{\\hspace{6pt}}" * (ncol - 1) + r"p{0.7\wdw}@{}"
        body = []
        dyn = "ps-i" in rows[0][0].cls()
        for r in rows:
            cells = [self.cell(td) for td in r] + [""] * (ncol - len(r))
            if dyn and self.skip_rows and any(k in " ".join(cells).lower() for k in self.skip_rows):
                continue
            if "ps-e" in c:                                      # an entity: its name, count and note on one row, its state list on a row of its own, ONE line (shrunk to the width when it is longer)
                name, sep, code = cells[0].partition("\\par{\\leftskip=1.5em\\relax ")
                body.append(" & ".join([name] + cells[1:]) + "\\\\[0pt]")
                if sep:
                    code = code[:-len("\\par}")] if code.endswith("\\par}") else code
                    body.append(entity_code_row(code))
                continue
            body.append(" & ".join(cells) + "\\\\[1pt]")
        tab = lambda b: "\\begin{tabular}[t]{" + spec + "}\n" + "\n".join(b) + "\n\\end{tabular}"      # noqa: E731
        if self.two and "ps-i" in rows[0][0].cls():              # the steps are one line each: two columns of them, the first half down the left
            h = (len(body) + 1) // 2
            return ("{\\wdbody\\begin{minipage}[t]{0.49\\wdw}" + tab(body[:h]) + "\\end{minipage}\\hfill\\begin{minipage}[t]{0.49\\wdw}"
                    + tab(body[h:]) + "\\end{minipage}\\par}\n")
        return "{\\wdbody" + tab(body) + "\\par}\n"

    def math_table(self, rows: list) -> str:
        out = []
        for r in rows:
            img = next((x for x in r[0].walk() if x.tag == "img"), None)
            k = int(re.search(r"[?&]k=(\d+)", img.attrs.get("src", "")).group(1)) if img is not None and re.search(r"[?&]k=(\d+)", img.attrs.get("src", "")) else -1
            eq = eq_stack(self.eqs[k], self.eq_budget) if self.eqs and 0 <= k < len(self.eqs) else (self.inline(img) if img is not None else "")
            word = "{\\color{wdyellow}\\sffamily " + (self.inline(r[1]) if len(r) > 1 else "") + "}"
            out.append(f"{eq} & {word}\\\\[8pt]")
        return "{\\wdbody\\begin{tabular}{@{}l@{\\hspace{10pt}}l@{}}\n" + "\n".join(out) + "\n\\end{tabular}\\par}\n"


def spec_tex(text: str) -> str:
    """The spec file as recorded: monospace, two columns of lines, a line's indentation kept; blank lines dropped."""
    lines = [ln.rstrip().expandtabs(2) for ln in text.splitlines() if ln.strip()]
    half = (len(lines) + 1) // 2

    def col(ls):
        out = []
        for ln in ls:
            ind = len(ln) - len(ln.lstrip(" "))
            out.append(f"{{\\leftskip={0.5 * ind:g}em\\hangindent=1.5em\\hangafter=1\\noindent {esc(ln.strip())}\\par}}")      # a long line wraps under its own indent, a little deeper
        return "\n".join(out)
    return ("{\\fontsize{7}{8.2}\\selectfont\\ttfamily\\raggedright\\begin{minipage}[t]{0.49\\wdw}\\raggedright\n" + col(lines[:half])
            + "\n\\end{minipage}\\hspace{3pt}\\begin{minipage}[t]{0.49\\wdw}\\raggedright\n" + col(lines[half:]) + "\n\\end{minipage}\\par}")


# ---------------------------------------------------------------- the record
class Record:
    def __init__(self, path: str):
        self.dir = os.path.abspath(path)
        self.id = os.path.relpath(self.dir, ROOT)
        self.name = os.path.basename(self.dir)
        self.pngs = sorted(glob.glob(os.path.join(self.dir, "png", "[0-9][0-9][0-9][0-9].png")))
        self.steps = [os.path.basename(p)[:4] for p in self.pngs]       # the watcher's row i is steps[i]

    def row(self, step: str) -> int:
        return self.steps.index(step)

    def movie(self, step: str) -> str | None:
        p = os.path.join(self.dir, "mp4", f"{step}.mp4")
        return p if os.path.exists(p) else None

    def spec(self, step: str) -> str:
        p = os.path.join(self.dir, "spec", f"{step}.yaml")
        if not os.path.exists(p):
            return ""
        txt = open(p, errors="replace").read()
        return txt if any(ln.strip() and not ln.lstrip().startswith("#") for ln in txt.splitlines()) else ""

    def spec_name(self, step: str) -> str:
        try:
            import yaml
            name = (yaml.safe_load(open(os.path.join(self.dir, "spec", f"{step}.yaml"))) or {}).get("general", {}).get("name")
            if name:
                return str(name)
        except Exception:                                        # noqa: BLE001
            pass
        p = os.path.join(self.dir, "why", f"{step}.txt")
        if os.path.exists(p):
            for ln in open(p, errors="replace"):
                if ln.startswith("spec "):
                    return ln.split(None, 1)[1].strip()
        return f"step {step}"


def sweep_title(rec: Record, step: str) -> str:
    """The slide title: only the parameters this step changes from the run before it (`spec_story.spec_diff`), as `name value` -- scalar
    parameters of the operators, fields, sets and seeds. A step that adds or removes an operator is a rebuilt rig and has none; a list,
    the schedule and the run's bookkeeping (name, headline, frame counts) are not parameters of a sweep. Empty when nothing of that kind
    changed (the caller then uses the spec's own name)."""
    try:
        import spec_story as SM
        prev = SM.previous_spec(os.path.join(rec.dir, "png", f"{step}.png"))
        if not prev:
            return ""
        out = []
        diff = SM.spec_diff(prev, os.path.join(rec.dir, "spec", f"{step}.yaml"), 10 ** 6)
        if any(ln.startswith("operator ") for ln in diff):         # an operator added or removed: a rebuilt rig, not a sweep of one
            return ""
        for ln in diff:
            m = re.match(r"([^:]+): (.*) -> (.*)$", ln)
            if not m or m.group(1) == "schedule" or m.group(1) in BOOKKEEPING:
                continue
            path, new = m.group(1), m.group(3).strip()
            if new[:1] in "[{(" or len(new) > 24:
                continue
            try:
                new = f"{float(new):.4g}"
            except ValueError:
                pass
            out.append(f"{path.rsplit('.', 1)[-1]} {new}")
        return ", ".join(out[:3]) + (", ..." if len(out) > 3 else "")
    except Exception as e:                                       # noqa: BLE001
        print(f"[deck] sweep parameters of step {step} not read ({type(e).__name__}: {e})", flush=True)
        return ""


def equations_of(rec: Record, step: str) -> list | None:
    """The equations of the step's math pane, as tex in the pane's order, from the story the watcher saved beside the step."""
    try:
        import spec_story as SM
        got = SM.load_saved(os.path.join(rec.dir, "story", f"{step}.json"))
        if got is None:
            return None
        return [row[1] for row in SM.math_equations(got[0], got[1])]
    except Exception as e:                                       # noqa: BLE001
        print(f"[deck] equations of step {step} not read ({type(e).__name__}: {e}); the watcher's bitmaps are used", flush=True)
        return None


# ---------------------------------------------------------------- the deck
def run(*a, **kw):
    return subprocess.run(a, check=True, **kw)


def background_of(png: str) -> str:
    """`r,g,b` of the movie's grey background when the picture has one, else "". The poster's border pixels are counted: one colour
    over at least half of them, grey (its three channels within 40 of each other, a slightly blue grey included) and not near-black, is the background; it is returned
    exactly, so the column behind the movie can be filled with the very same grey."""
    from collections import Counter
    from PIL import Image
    im = Image.open(png).convert("RGB")
    w, h = im.size
    px = im.load()
    edge = [px[x, y] for x in range(w) for y in (0, h - 1)] + [px[x, y] for y in range(h) for x in (0, w - 1)]
    (r, g, b), n = Counter(edge).most_common(1)[0]
    if n >= 0.5 * len(edge) and max(r, g, b) - min(r, g, b) <= 40 and (r + g + b) / 3 >= 40:
        return f"{r},{g},{b}"
    return ""


def poster(ff: str, movie: str, png: str) -> None:
    run(ff, "-v", "error", "-y", "-i", movie, "-frames:v", "1", "-update", "1", png)


def drop_rows(frame_tex: str, need_pt: float, lay: str) -> str:
    """Whole dynamics rows taken off the END of the layout the slide uses (`lay` A: one table, B: two columns) until at least `need_pt` points (before the deck's scale) of its height are gone, so the pane ends on a row above the insights.

    A row is one line of 10.5 pt + 1 pt, or more when its text wraps (about 62 characters to a line in layout A, 29 in B: a little generous, so a row is taken out rather than left clipped): the height of a row is estimated from its text, and in layout B the row is taken from whichever column is taller."""
    pat = re.compile(r"(\\begin\{tabular\}\[t\]\{@\{\}l@\{\\hspace\{" + ("4pt" if lay == "A" else "3pt") + r"\}\}[^\n]*\}\n)((?:\\raisebox[^\n]*\n)+)(\\end\{tabular\})")
    cpl = 62 if lay == "A" else 29
    def height(row):
        txt = row.split(" & ", 1)[-1].rsplit("\\\\", 1)[0]
        n = len(re.sub(r"\\[a-zA-Z]+|[{}]", "", txt).strip())
        return max(1, -(-n // cpl)) * 10.5 + 1.0
    m = list(pat.finditer(frame_tex))
    if not m:
        return frame_tex
    cols = [m_.group(2).splitlines(keepends=True) for m_ in m]                 # layout A: one table; layout B: two
    hs = [[height(r) for r in c] for c in cols]
    start = max(sum(h) for h in hs)
    while cols and start - max(sum(h) for h in hs) < need_pt and any(cols):
        k = max((i for i in range(len(cols)) if cols[i]), key=lambda i: sum(hs[i]))
        cols[k].pop(); hs[k].pop()
    out, last = [], 0
    for m_, c in zip(m, cols):
        out.append(frame_tex[last:m_.start()] + m_.group(1) + "".join(c) + m_.group(3)); last = m_.end()
    return "".join(out) + frame_tex[last:]


def pad_canvas(movie: str, x0: float, col_w: float, panel_w: float, bg: str) -> str:
    """The movie's own picture continued outward by its edge pixels over the whole panel it sits in -> the PNG's path relative to presentation/ ("" when the movie has no grey background).

    A flat fill beside the picture is drawn by the viewer through another route than the picture is, and the two greys differ by 1-2 levels on screen (a visible step at the picture's edge), and by 11 levels where the render's background is a gradient.
    Here the panel is ONE kind of pixel: the picture's pixels are untouched at their place (scale, offsets as `\\wdmovie` sets them: the picture fitted in its box, centred both ways) and every pixel around them repeats the nearest edge pixel, so the grey is
    continuous across the edge whatever the viewer does to images. The part under the picture is hidden by it and is filled flat to keep the file small."""
    if not bg or not ASSETS_DIR:
        return ""
    from PIL import Image
    import numpy as np
    im = np.array(Image.open(os.path.join(PRES, movie + ".png")).convert("RGB"))
    ph, pw = im.shape[:2]
    sc = min(col_w / pw, HEIGHT / ph)                                  # cm per pixel
    cx, cy = x0 + col_w / 2, TOP + HEIGHT / 2
    wc, hc = int(round(panel_w / sc)), int(round(9.0 / sc))
    xo, yo = int(round(cx / sc - pw / 2)), int(round(cy / sc - ph / 2))
    if xo < 0 or yo < 0 or xo + pw > wc or yo + ph > hc:
        return ""
    big = np.pad(im, ((yo, hc - yo - ph), (xo, wc - xo - pw), (0, 0)), mode="edge")
    big[yo + 2:yo + ph - 2, xo + 2:xo + pw - 2] = im[2, 2]               # hidden under the picture
    key = hashlib.md5(big.tobytes()).hexdigest()[:14]
    path = os.path.join(ASSETS_DIR, f"pad_{key}.png")
    if not os.path.exists(path):
        Image.fromarray(big).save(path, optimize=True)
    return os.path.basename(ASSETS_DIR) + "/" + os.path.basename(path)


PREAMBLE = r"""%!TEX program = pdflatex
% GENERATED by tools/watcher_deck.py -- edit the tool, not this file.
\documentclass[aspectratio=169]{beamer}
\usetheme{Janelia}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{helvet}
\renewcommand{\familydefault}{\sfdefault}
\usepackage{amsmath,amssymb,bm,array}
\usepackage{multimedia}
\usepackage[overlay,absolute]{textpos}
\usepackage{xfp,varwidth,adjustbox}
\DeclareUnicodeCharacter{2192}{\ensuremath{\rightarrow}}
\DeclareUnicodeCharacter{2190}{\ensuremath{\leftarrow}}
\DeclareUnicodeCharacter{2248}{\ensuremath{\approx}}
\DeclareUnicodeCharacter{2265}{\ensuremath{\geq}}
\DeclareUnicodeCharacter{2264}{\ensuremath{\leq}}
\DeclareUnicodeCharacter{2260}{\ensuremath{\neq}}
\DeclareUnicodeCharacter{00B1}{\ensuremath{\pm}}
\DeclareUnicodeCharacter{2206}{\ensuremath{\Delta}}
\DeclareUnicodeCharacter{2202}{\ensuremath{\partial}}
\DeclareUnicodeCharacter{221E}{\ensuremath{\infty}}
\DeclareUnicodeCharacter{2032}{\ensuremath{{}^{\prime}}}
\setlength{\TPHorizModule}{1cm}
\setlength{\TPVertModule}{1cm}
\setbeamercolor{background canvas}{bg=black}
\setbeamercolor{normal text}{fg=white,bg=black}
\setbeamercolor{structure}{fg=white}
\usebeamercolor[fg]{normal text}
\AtBeginDocument{\color{white}}
\definecolor{wdgray}{gray}{0.62}
\definecolor{wdyellow}{HTML}{E8C547}
% one title size for every slide, small enough that a two-line headline stays inside the theme's 1 cm band
\renewcommand{\frametitlesize}{\fontsize{9}{10}\selectfont}
% the fixed geometry (tools/watcher_deck.py: MARGIN, TOP, HEIGHT, COL_R, GAP); the left panel starts at the slide's left edge
\newlength{\wdtop}\setlength{\wdtop}{@top@cm}
\newlength{\wdh}\setlength{\wdh}{@height@cm}
\newlength{\wdcoll}\setlength{\wdcoll}{@col_l@cm}
\newlength{\wdcolr}\setlength{\wdcolr}{@col_r@cm}
\newlength{\wdxl}\setlength{\wdxl}{0cm}
\newlength{\wdxr}\setlength{\wdxr}{@xr@cm}
\newlength{\wdcolm}\setlength{\wdcolm}{@col_m@cm}
\newlength{\wdxm}\setlength{\wdxm}{@margin@cm}
\newlength{\wdw}
\newsavebox{\wdbox}
% the pane's two type sizes (before the deck's scale): section headings large and yellow, everything else smaller and white
\newcommand{\wdbody}{\fontsize{9}{10.5}\selectfont}
\newcommand{\wdhead}{\fontsize{13}{15}\selectfont\bfseries}
\makeatletter
% the fixed scale of each pane kind, and the width in points it is typeset at before that scale (the column's width over the scale)
@natural@
\newcommand{\wdscaleof}[1]{\csname wdscale@#1\endcsname}
% a TEXT pane has two layouts, A (preferred) and B (denser). Both are typeset at the kind's natural width and scaled by the kind's
% FIXED scale; a slide takes A when A fits the column's height at that scale and B when it does not. What still does not fit is CUT
% off the bottom (and the right) of the column, and reported (WDCUT): the type size never changes.
\newsavebox{\wdboxb}
\newcommand{\wdfit}[1]{\fpeval{min(1, \strip@pt\wdh/\strip@pt\dimexpr\ht#1+\dp#1\relax, \strip@pt\wdcolr/\strip@pt\wd#1)}}
\newcommand{\wdfith}[1]{\fpeval{min(1, \strip@pt\wdh/\strip@pt\dimexpr\ht#1+\dp#1\relax)}}
\newlength{\wdpaneh}
\newcommand{\wd@emit}[3]{%
  \global\wdpaneh=\fpeval{min(\wd@s*\strip@pt\dimexpr\ht#1+\dp#1\relax, \strip@pt\wdh)}pt\relax% a primitive global assignment: `\global\setlength` is lost with calc loaded, and the insights then sat at the top of the column, over the states (Cedric, 2026-11-15)
  \edef\wd@exh{\fpeval{max(0, \wd@s*\strip@pt\dimexpr\ht#1+\dp#1\relax-\strip@pt\wdh)}}%
  \edef\wd@exw{\fpeval{max(0, \wd@s*\strip@pt\wd#1-\strip@pt\wdcolr)}}%
  \edef\wd@nat{\fpeval{\wd@s*\strip@pt\dimexpr\ht#1+\dp#1\relax}}%
  \typeout{WDCUT kind=#2 step=\wd@step\space cut=\wd@exh\space cutw=\wd@exw\space lay=#3\space nat=\wd@nat}%
  \noindent\clipbox{0pt \wd@exh pt \wd@exw pt 0pt}{\scalebox{\wd@s}{\usebox{#1}}}}
\newcommand{\wdpane}[4]{%
  \def\wd@step{#2}\edef\wd@s{\wdscaleof{#1}}%
  \setlength{\wdw}{\csname wdnat@#1\endcsname}%
  \sbox{\wdbox}{\begin{varwidth}{\wdw}\wdbody #3\end{varwidth}}%
  \sbox{\wdboxb}{\begin{varwidth}{\wdw}\wdbody #4\end{varwidth}}%
  \edef\wd@fa{\fpeval{min(1, \strip@pt\wdh/\strip@pt\dimexpr\ht\wdbox+\dp\wdbox\relax, \strip@pt\wdcolr/\strip@pt\wd\wdbox)}}%
  \ifdim\wd@fa pt<\wd@s pt\wd@emit{\wdboxb}{#1}{B}\else\wd@emit{\wdbox}{#1}{A}\fi}
\makeatother
% the movie panel: the movie centred in it both ways
\pdflinkmargin=0pt   % pdfpc plays a movie in its whole link rectangle, which pdfTeX pads by 1 pt a side: the movie then overflowed its poster (Cedric, 2026-11-15, plexus_exp05)
\newcommand{\wdmovie}[2]{\vbox to \wdh{\vss\hbox to #2{\hss\href{run:#1.mp4?autostart&loop&noprogress}{\includegraphics[width=#2,height=\wdh,keepaspectratio]{#1.png}}\hss}\vss}}
% the page canvas, black, with the movie's own grey (r,g,b as found in its picture) filling the left panel from the slide's left edge to
% width #1, from under the title band to the bottom of the slide; it sits under the theme's band and progress bar, which stay
\newcommand{\wdcanvasbox}{}
\setbeamertemplate{background canvas}{\wdcanvasbox}
\newcommand{\wdcanvas}[3]{\gdef\wdcanvasbox{\color{black}\vrule width\paperwidth height\paperheight depth0pt\ifx\relax#2\relax\else\kern-\paperwidth\ifx\relax#3\relax\color[RGB]{#2}\vrule width #1 height\paperheight depth0pt\else\includegraphics[width=#1,height=\paperheight]{#3}\fi\fi}}
\newcommand{\wdfig}[1]{\vbox to \wdh{\vss\hbox to \wdcolr{\hss\includegraphics[width=\wdcolr,height=\wdh,keepaspectratio]{#1}\hss}\vss}}
% the notes' insights block (--notes): a yellow heading and a plain body at the bottom of the right column; the pane above it gets the height
% that is left (\wdh is shortened inside a group, so the pane's fit and clipping follow)
\newsavebox{\wdibox}
\newlength{\wdih}\newlength{\wdiy}\newlength{\wdhfull}\setlength{\wdhfull}{@height@cm}
\newcommand{\wdinshead}{\fontsize{8.6}{10}\selectfont\bfseries\color{wdyellow}}
\newcommand{\wdinsbody}{\fontsize{5.0}{6}\selectfont\mdseries\color{white}}
\newcommand{\wdinsarrow}{\fontsize{8.6}{10}\selectfont\mdseries\color{white}}
\newcommand{\wdinsights}[3]{\sbox{\wdibox}{\begin{minipage}[t]{\wdcolr}\raggedright{\wdinshead insights}{\wdinsarrow\ #2}{\wdinsbody\ \ #3}\par\vspace{3pt}{\wdinsbody #1\par}\end{minipage}}%
  \setlength{\wdih}{\dimexpr\ht\wdibox+\dp\wdibox\relax}\setlength{\wdiy}{\dimexpr\wdtop+\wdhfull-\wdih\relax}}
% a text slide (--notes intro / summary): headings yellow, bullets, 8 pt, across the content area
\newcommand{\wdtexthead}{\fontsize{9}{11}\selectfont\bfseries\color{wdyellow}}
\newcommand{\wdtextbody}{\fontsize{7.5}{9.5}\selectfont\mdseries\color{white}}
\title{}\author{}\date{}
\begin{document}
% the Janelia theme draws page 1 as a title page; the first slide of this deck is a content slide
\setcounter{page}{2}
"""


def _fill(text: str, vals: dict) -> str:
    for k, v in vals.items():
        text = text.replace(f"@{k}@", str(v))
    return text


def frame(title: str, foot: str, movie: str, pane, kind: str, step: str, bg: str = "", insights: str = "", trend: str = "", trend_why: str = "") -> str:
    """The two templates. `pane` is the right column's tex, or a (layout A, layout B) pair (None: the one-movie template); `kind` names how
    it is set; `bg` is the movie's grey background as `r,g,b`, which then fills the movie's column. `insights` (--notes) is a plain text set
    at the bottom of the right column; a step with no pane then takes the two-column template with the insights alone on the right."""
    arrow = TREND.get(str(trend or "").strip().lower(), "")
    if trend and not arrow:
        raise SystemExit(f"step {step}: unknown trend {trend!r} in the notes; one of up, right, down (progress, status quo, regression)")
    ink = "\\fontsize{4.5}{5}\\selectfont{}" + ("\\color{white}" if bg else "")      # the footer line: small, and white where the movie's grey is under it (the theme's own grey is lost there)
    head = f"\\renewcommand{{\\janeliafootleft}}{{{'' if NO_FOOT else ink + esc(foot)}}}\n\\begin{{frame}}{{{esc(title)}}}\n"
    if pane is None and not insights:
        return (f"\\wdcanvas{{\\paperwidth}}{{{bg}}}{{{pad_canvas(movie, MARGIN, 16 - 2 * MARGIN, 16.0, bg)}}}\n" + head + f"\\begin{{textblock*}}{{\\wdcolm}}(\\wdxm,\\wdtop)\\wdmovie{{{movie}}}{{\\wdcolm}}\\end{{textblock*}}\n\\end{{frame}}\n")
    out = f"\\wdcanvas{{\\wdcoll}}{{{bg}}}{{{pad_canvas(movie, 0.0, COL_L, COL_L, bg)}}}\n" + head + f"\\begin{{textblock*}}{{\\wdcoll}}(\\wdxl,\\wdtop)\\wdmovie{{{movie}}}{{\\wdcoll}}\\end{{textblock*}}\n"
    if pane is not None:
        a, b = pane if isinstance(pane, tuple) else (pane, pane)
        right = f"\\wdfig{{{a}}}" if kind == "hierarchy" else f"\\wdpane{{{kind}}}{{{step}}}{{%\n{a}\n}}{{%\n{b}\n}}"
        if insights:
            out += (f"\\wdinsights{{{esc(insights)}}}{{{arrow}}}{{{esc(trend_why)}}}\n{{\\setlength{{\\wdh}}{{\\dimexpr\\wdhfull-\\wdih-0.3cm\\relax}}"
                    f"\\begin{{textblock*}}{{\\wdcolr}}(\\wdxr,\\wdtop){right}\\end{{textblock*}}}}\n"
                    f"\\begin{{textblock*}}{{\\wdcolr}}(\\wdxr,{INS_Y}){{\\usebox{{\\wdibox}}}}\\end{{textblock*}}\n")
        else:
            out += f"\\begin{{textblock*}}{{\\wdcolr}}(\\wdxr,\\wdtop){right}\\end{{textblock*}}\n"
    else:
        out += f"\\wdinsights{{{esc(insights)}}}{{{arrow}}}{{{esc(trend_why)}}}\n\\begin{{textblock*}}{{\\wdcolr}}(\\wdxr,\\wdtop)\\usebox{{\\wdibox}}\\end{{textblock*}}\n"
    return out + "\\end{frame}\n"


def manual_pane_tex(p: dict) -> str:
    """A plexus-style pane typed in the notes, for a step with no spec to read (--notes `pane: {entities: [[name, count, note, code], ...], dynamics: {note: "...", rows: ["...", ...]}}`): the same two yellow headings,
    the same entity table and dynamics rows as the watcher's plexus pane, a grey bullet where that pane has a kind's glyph, or the glyph itself when a row is written ["text", "kind"] (presentation/icons/<kind>.png)."""
    ent = []
    for r in p.get("entities") or []:
        name, count, note, code = (list(r) + ["", "", ""])[:4]
        ent.append(esc(str(name)) + " & " + esc(str(count)) + " & " + esc(str(note)) + "\\\\[0pt]")
        if code:
            ent.append(entity_code_row("{\\color{wdgray}" + esc(str(code)) + "}"))
    dyn = p.get("dynamics") or {}
    def glyph(t):                                                  # a row is "text" (a grey bullet) or ["text", "kind"] (the kind's glyph presentation/icons/<kind>.png)
        text, kind = (t if isinstance(t, (list, tuple)) else [t, ""])[:2] if not isinstance(t, str) else (t, "")
        mark = ("\\includegraphics[height=1.5em]{icons/" + kind + ".png}") if kind else "\\textcolor{wdgray}{\\textbullet}"
        return "\\raisebox{-0.35em}[0pt][0pt]{" + mark + "} & " + esc(str(text)) + " \\\\[1pt]"
    rows = [glyph(t) for t in dyn.get("rows") or []]
    return ("{\\wdhead\\color{wdyellow}" + ENTITIES_WORD + "}\\par\\vspace{7pt}\n"
            "{\\wdbody\\begin{tabular}[t]{@{}>{\\raggedright\\arraybackslash}p{0.52\\wdw}@{\\hspace{5pt}}r@{\\hspace{6pt}}>{\\raggedright\\arraybackslash}p{0.22\\wdw}@{}}\n" + "\n".join(ent) + "\n\\end{tabular}\\par}\n"
            "\\vspace{14pt}{\\wdhead\\color{wdyellow}dynamics {\\wdbody\\mdseries\\color{wdgray}" + esc(str(dyn.get("note", ""))) + "}}\\par\\vspace{7pt}\n"
            "{\\wdbody\\begin{tabular}[t]{@{}l@{\\hspace{4pt}}>{\\raggedright\\arraybackslash}p{0.86\\wdw}@{}}\n" + "\n".join(rows) + "\n\\end{tabular}\\par}\n")


TABLE_TOKENS = {"\u2713": r"\textcolor[HTML]{3DDC84}{\ensuremath{\checkmark}}", "\u2717": r"\textcolor[HTML]{E5484D}{\ensuremath{\times}}", "\u2191": r"\ensuremath{\uparrow}", "\u2193": r"\ensuremath{\downarrow}"}


def table_tex(table: dict) -> str:
    """A table of a text slide (--notes `table: {header: [...], rows: [[...]], widths: [cm or 0, ...], align: "lplrcrcc", footer: "..."}`): a yellow regular header over a hairline, one row per list; a column
    of align `l` with a width is a paragraph column of that width in cm. In a cell the characters check mark (green), cross (red), up arrow and down arrow are drawn as symbols. `footer`: one grey line under the table."""
    cell = lambda t: "".join(TABLE_TOKENS.get(part) or esc(part) for part in re.split("([\u2713\u2717\u2191\u2193])", str(t)))      # noqa: E731
    head, rows = table.get("header") or [], table.get("rows") or []
    widths, align = table.get("widths") or [], table.get("align") or "l" * len(head or rows[0])
    cols = [(f"p{{{widths[i]}cm}}" if (i < len(widths) and widths[i] and a == "l") else a) for i, a in enumerate(align)]
    out = ["{\\wdtextbody\\renewcommand{\\arraystretch}{1.28}\\begin{tabular}{@{}" + "@{\\hspace{5pt}}".join(cols) + "@{}}"]
    if head:
        out.append(" & ".join("\\textcolor{wdyellow}{" + cell(h) + "}" for h in head) + "\\\\\\hline")
    for r in rows:
        out.append(" & ".join(cell(c) for c in r) + "\\\\")
    out.append("\\end{tabular}\\par}")
    if table.get("footer"):
        out.append("\\vspace{5pt}{\\wdtextbody\\color{wdgray}" + esc(table["footer"]) + "\\par}")
    return "\n".join(out)


def text_frame(title: str, foot: str, body: str, table: dict | None = None) -> str:
    """A text slide of the notes (intro, summary): each line of `body` a paragraph; a line ending with ":" is a yellow heading, a line starting
    with "- " a bullet; a blank line a small gap."""
    paras = []
    for ln in body.splitlines():
        s = ln.strip()
        if not s:
            paras.append("\\vspace{3pt}")
        elif s.endswith(":"):
            paras.append("{\\wdtexthead " + esc(s) + "}\\par\\vspace{2pt}")
        elif s.startswith("- "):
            paras.append("{\\wdtextbody\\leftskip=1.2em\\hangindent=1.2em\\hangafter=1\\noindent\\llap{\\textbullet\\hspace{0.5em}}" + esc(s[2:]) + "\\par}")
        else:
            paras.append("{\\wdtextbody\\noindent " + esc(s) + "\\par}")
    if table:
        paras.append(table_tex(table))
    ink = "\\fontsize{4.5}{5}\\selectfont{}"
    # the text is set at its fixed size and, when it is taller than the content area, scaled down to fit it (never cut): a long summary gets smaller type
    return (f"\\wdcanvas{{\\paperwidth}}{{}}{{}}\n\\renewcommand{{\\janeliafootleft}}{{{'' if NO_FOOT else ink + esc(foot)}}}\n\\begin{{frame}}{{{esc(title)}}}\n"
            f"\\sbox{{\\wdbox}}{{\\begin{{minipage}}[t]{{\\wdcolm}}\\raggedright\n" + "\n".join(paras) + "\n\\end{minipage}}\n"
            f"\\begin{{textblock*}}{{\\wdcolm}}(\\wdxm,\\wdtop)\\scalebox{{\\wdfith{{\\wdbox}}}}{{\\usebox{{\\wdbox}}}}\\end{{textblock*}}\n\\end{{frame}}\n")


def load_notes(path: str) -> tuple[dict, dict]:
    """The notes YAML -> (deck notes, {step: {title, insights}}). Two spellings are read: `steps:` or `slides:` for the per-step map (keys are
    the record's step file numbers, 1 or "0001"); `intro:` / `summary:` with a `body:` text, or with `sections: [{heading, lines: [...]}]`,
    which is turned into the same text (a heading line ending with ":", each line a bullet)."""
    import yaml
    notes = yaml.safe_load(open(path)) or {}
    for key in ("intro", "summary"):
        for sec in (notes.get(key) if isinstance(notes.get(key), list) else [notes.get(key)]):
          if sec and not sec.get("body") and sec.get("sections"):
              lines = []
              for s in sec["sections"]:
                  h = str(s.get("heading", "")).strip()
                  if h:
                      lines.append(h if h.endswith(":") else h + ":")
                  lines += ["- " + str(ln) for ln in (s.get("lines") or [])]
                  lines.append("")
              sec["body"] = "\n".join(lines)
    per = notes.get("steps") or notes.get("slides") or {}
    return notes, {str(k).zfill(4): (v or {}) for k, v in per.items()}


def watcher_numbers(text: str, offset: int) -> str:
    """The deck's "slide N" / "slides N-M" read as the watcher's "movie N-offset" / "movies ...": the deck's intro slide comes first, so deck slide k shows the watcher's movie k-1 (`offset` 1; 0 for a deck without an intro)."""
    def one(m):
        lo = int(m.group(2)) - offset; hi = m.group(3)
        return ("movies " if m.group(1).lower() == "slides" else "movie ") + (f"{lo}-{int(hi) - offset}" if hi else str(lo))
    return re.sub(r"\b(slides?) (\d+)(?:-(\d+))?", one, text)


def export_insights(record: str, step_notes: dict, notes_path: str = "", offset: int = 0) -> list[str]:
    """`--export-insights`: write `<record>/story/NNNN.insights.json` = {title, insights, trend, trend_why} for every step of the notes' steps map that has
    insights and exists in the record, so the watcher's `insights` tab reads what the deck prints. The notes YAML stays the single source; only this one
    file of a step is written (atomically), never its other story files. Returns the report lines (written, and notes steps the record does not have)."""
    rec = Record(record)
    story = os.path.join(rec.dir, "story")
    os.makedirs(story, exist_ok=True)
    report = []
    for s, sn in sorted(step_notes.items()):
        if "insight" in sn:
            raise SystemExit(f"step {s}: the notes key `insight:` is now `insights:`; rename it in {notes_path}")
        text = (sn.get("insights") or "").strip()
        if not text:
            continue
        trend = str(sn.get("trend") or "").strip().lower()
        if trend and trend not in TREND:
            raise SystemExit(f"step {s}: unknown trend {sn.get('trend')!r} in the notes; one of up, right, down (progress, status quo, regression)")
        if s not in rec.steps:
            report.append(f"{s}: not in {rec.id}, skipped")
            continue
        path = os.path.join(story, f"{s}.insights.json")
        with open(path + ".tmp", "w") as f:
            json.dump({"title": str(sn.get("title") or ""), "insights": watcher_numbers(text, offset), "trend": trend, "trend_why": str(sn.get("trend_why") or "").strip()},
                      f, ensure_ascii=False, indent=1)
        os.replace(path + ".tmp", path)
        report.append(f"{s}: written")
    return report


def pdflatex(tex: str, halt: bool = True) -> str:
    """One pdflatex run; `halt=False` for the first run of a deck with no .aux yet: the theme's progress bar divides by the total number of frames, which the .aux has not recorded, and overflows on a late frame of a long deck -- with halt-on-error that run never reaches \\end{document} to write the total, and no later run could succeed."""
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode"] + (["-halt-on-error"] if halt else []) + [os.path.basename(tex)], cwd=PRES, capture_output=True, text=True, errors="replace")      # a LaTeX error can print a raw 8-bit byte, which must not stop the tool
    return r.stdout


def main() -> None:
    global ENTITIES_WORD, NO_FOOT, INS_Y
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("record", help="a record folder, e.g. experiments/exp38_gap_function")
    ap.add_argument("--out", default=None, help="the PDF (default presentation/watcher_<record>.pdf)")
    ap.add_argument("--steps", default="", help="comma-separated step numbers (default: every step that has a movie; none: no movie slide)")
    ap.add_argument("--pane", default="plexus", help="comma-separated panes of the right column, one slide each per movie: plexus, math, spec, hierarchy, none")
    ap.add_argument("--pane-for", action="append", default=[], metavar="STEP=PANE[,PANE]", help="panes of one step, replacing --pane for it")
    ap.add_argument("--full", action="store_true", help="the plexus pane with its story and the line under each step, as the watcher shows it")
    ap.add_argument("--entities-word", default="states", help="the word printed for the plexus pane's first heading (default states, Cedric 2026-11-14; entities for the old word)")
    ap.add_argument("--watcher", default=WATCHER)
    ap.add_argument("--no-make", action="store_true", help="read the panes already saved; never wait for the watcher to make one")
    ap.add_argument("--scale", action="append", default=[], metavar="KIND=SCALE", help=f"the fixed scale of a pane kind, replacing {SCALE}")
    ap.add_argument("--wait", type=float, default=900.0, help="seconds to wait for panes the watcher is making")
    ap.add_argument("--notes", default=None, help="a YAML of typed words: per-step title and insights, an intro and a summary slide (see NOTES above)")
    ap.add_argument("--frames-before", default=None, metavar="TEX", help="a tex file of hand-written frames, inserted verbatim before the first generated slide (a title page, a figure slide)")
    ap.add_argument("--drop", action="append", default=[], metavar="SECTION", help="leave a plexus-pane section out, heading and content (repeatable), e.g. --drop learnables; matched on the start of the heading, case-insensitive")
    ap.add_argument("--eq-budget", type=int, default=36, help="characters of source per line of a stacked equation in the math pane (default 36; a larger number keeps short equations on one line, the ones wider than the column still overflow)")
    ap.add_argument("--insights-under", action="store_true", help="the insights block right under the pane's drawn text, one blank line below it, instead of at the bottom of the column")
    ap.add_argument("--no-foot", action="store_true", help="no record/step line at the bottom left of the slides")
    ap.add_argument("--full-range-movies", action="store_true", help="re-encode the deck's movie copies in full range (yuvj420p, tagged pc): pdfpc shows limited-range movies darker than their poster")
    ap.add_argument("--export-insights", action="store_true", help="write the notes' per-step insights into the record's story folder (NNNN.insights.json, read by the watcher's insights tab) and stop: no PDF, no watcher")
    a = ap.parse_args()
    ENTITIES_WORD = a.entities_word
    NO_FOOT = bool(a.no_foot)
    if a.insights_under:                                      # one blank line (7 pt, a line of the plexus pane at its scale) under the pane's drawn text
        INS_Y = "\\dimexpr\\wdtop+\\wdpaneh+7pt\\relax"

    notes, step_notes = load_notes(a.notes) if a.notes else ({}, {})
    if a.export_insights:
        if not a.notes:
            raise SystemExit("--export-insights needs --notes <yaml>")
        lines = export_insights(a.record, step_notes, a.notes, 1 if notes.get("intro") else 0)
        print("\n".join(f"[deck] insights {ln}" for ln in lines) + f"\n[deck] {sum(ln.endswith('written') for ln in lines)} insights file(s) written in {os.path.join(a.record, 'story')}")
        return

    rec = Record(a.record)
    out = os.path.abspath(a.out or os.path.join(PRES, f"watcher_{rec.name}.pdf"))
    deck = os.path.splitext(os.path.basename(out))[0]
    assets = os.path.join(PRES, f"{deck}_assets")
    global ASSETS_DIR
    ASSETS_DIR = assets
    os.makedirs(assets, exist_ok=True)
    os.makedirs(os.path.join(PRES, "Movies"), exist_ok=True)
    want = [] if a.steps == "none" else [s.zfill(4) for s in a.steps.split(",") if s] or [s for s in rec.steps if rec.movie(s)]
    for s in want:
        if s not in rec.steps or not rec.movie(s):
            raise SystemExit(f"step {s} has no movie in {rec.id}/mp4")
    per_step = {s: [p for p in a.pane.split(",") if p] for s in want}
    for spec in a.pane_for:
        s, _, ps = spec.partition("=")
        per_step[s.zfill(4)] = [p for p in ps.split(",") if p]
    for ps in per_step.values():
        for p in ps:
            if p not in PANES + ("none",):
                raise SystemExit(f"unknown pane {p!r}; one of {', '.join(PANES)}, none")

    # 1. the panes, from the watcher: the plexus pane of every step (its headline is the slide title), plus each pane asked for
    kinds = {(k, rec.row(s)) for s in want for k in {p for p in per_step[s] if p in ("plexus", "math", "hierarchy")} if rec.spec(s)}
    w = Watcher(a.watcher, rec.id)
    got = {}
    try:
        if kinds:
            w.ensure()
            got = gather(w, sorted(kinds), not a.no_make, a.wait)
        cache: dict = {}

        def fetch(src: str) -> str | None:
            """A watcher image address -> a PNG beside the deck, once per address."""
            if src in cache:
                return cache[src]
            path = None
            try:
                data = w.get(src)
                if data[:4] == b"\x89PNG":
                    path = os.path.join(assets, "img_" + hashlib.sha1(src.encode()).hexdigest()[:14] + ".png")
                    open(path, "wb").write(data)
            except Exception as e:                               # noqa: BLE001
                print(f"[deck] image {src[:60]}: {type(e).__name__}", flush=True)
            cache[src] = os.path.relpath(path, PRES) if path else None
            return cache[src]

        # 2. the slides
        ff = __import__("imageio_ffmpeg").get_ffmpeg_exe()
        frames, log_notes = [], []
        fb_idx = -1
        if a.frames_before:
            fb_idx = len(frames)                                  # the hand-written frames are ONE block: the row dropping below must never cut the panes inside it
            frames.append(open(a.frames_before).read())
        if notes.get("intro"):
            frames.append(text_frame(notes["intro"].get("title", ""), rec.id, notes["intro"].get("body", "")))
        for s in want:
            name = f"{rec.name}_{s}"
            mov = os.path.join(PRES, "Movies", name + ".mp4")
            if not os.path.exists(mov) or os.path.getmtime(mov) < os.path.getmtime(rec.movie(s)) or (a.full_range_movies and not os.path.exists(mov + ".fullrange")):
                shutil.copyfile(rec.movie(s), mov)
                if a.full_range_movies:                           # FULL-RANGE COPY: pdfpc showed the limited-range movie 3-5 levels darker than its poster and the grey canvas around it (Cedric, 2026-11-15)
                    run(ff, "-v", "error", "-y", "-i", mov, "-vf", "scale=out_range=full", "-pix_fmt", "yuvj420p", "-color_range", "pc",
                        "-c:v", "libx264", "-crf", "16", "-movflags", "+faststart", mov[:-4] + ".tmp.mp4")
                    os.replace(mov[:-4] + ".tmp.mp4", mov)
                    open(mov + ".fullrange", "w").close()
                poster(ff, mov, mov[:-4] + ".png")
            movie = f"Movies/{name}"
            bg = background_of(mov[:-4] + ".png")
            foot = f"{rec.id}  step {s}"
            sn = step_notes.get(s, {})
            title = sn.get("title") or sweep_title(rec, s) or rec.spec_name(s)
            if "insight" in sn:
                raise SystemExit(f"step {s}: the notes key `insight:` is now `insights:`; rename it in {a.notes}")
            insights = (sn.get("insights") or "").strip()
            panes = [p for p in per_step[s] if p != "none"]
            if sn.get("pane") and (not panes or not rec.spec(s)):                # a pane typed in the notes for a step with no spec to read
                tex = manual_pane_tex(sn["pane"])
                frames.append(frame(title, foot + "  notes pane", movie, (tex, tex), "plexus", s, bg, insights, sn.get("trend", ""), sn.get("trend_why", "")))
                log_notes.append(f"{s}: plexus pane typed in the notes")
                continue
            if not panes or not rec.spec(s):
                frames.append(frame(title, foot, movie, None, "", s, bg, insights, sn.get("trend", ""), sn.get("trend_why", "")))
                log_notes.append(f"{s}: movie only" + ("" if panes == [] else " (no spec to read)"))
                continue
            for kind in panes:
                if kind == "spec":
                    tex = spec_tex(rec.spec(s))
                elif kind == "hierarchy":
                    j = got[("hierarchy", rec.row(s))]
                    m = re.search(r"base64,([A-Za-z0-9+/=]+)", j.get("text", ""))
                    if not m:
                        frames.append(frame(title, foot, movie, None, "", s, bg, insights, sn.get("trend", ""), sn.get("trend_why", "")))
                        log_notes.append(f"{s}: hierarchy: {PaneTex(fetch, True, None).inline(parse(j.get('text', '')))[:70] or 'none'}; movie only")
                        continue
                    png = os.path.join(assets, f"hierarchy_{s}.png")
                    open(png, "wb").write(base64.b64decode(m.group(1)))
                    tex = os.path.relpath(png, PRES)
                else:
                    j = got[(kind, rec.row(s))]
                    pt = PaneTex(fetch, not a.full, equations_of(rec, s) if kind == "math" else None)
                    pt.skip_rows = [str(x).lower() for x in (sn.get("skip_rows") or [])]
                    pt.drop_sections, pt.eq_budget = tuple(x.lower() for x in a.drop), a.eq_budget
                    tex = pt.block(parse(j["text"]))
                    if kind == "plexus" and not a.full:                  # layout A: the dynamics in one column; layout B: in two, for the slides that need it
                        pt2 = PaneTex(fetch, True, None)
                        pt2.two = True
                        pt2.skip_rows, pt2.drop_sections = pt.skip_rows, pt.drop_sections
                        tex = (tex, pt2.block(parse(j["text"])))
                    if not (tex[0] if isinstance(tex, tuple) else tex).strip():
                        frames.append(frame(title, foot, movie, None, "", s, bg, insights, sn.get("trend", ""), sn.get("trend_why", "")))
                        log_notes.append(f"{s}: {kind} pane is empty; movie only")
                        continue
                frames.append(frame(title, foot + f"  {kind} pane", movie, tex, kind, s, bg, insights, sn.get("trend", ""), sn.get("trend_why", "")))
                log_notes.append(f"{s}: {kind}")
        summ = notes.get("summary") or []
        for sec in (summ if isinstance(summ, list) else [summ]):
            frames.append(text_frame(sec.get("title", ""), rec.id, sec.get("body", ""), sec.get("table")))
    finally:
        w.stop()

    # 3. typeset: every pane kind at its fixed scale; a second run settles the slide count in the footer
    texf = os.path.join(PRES, deck + ".tex")
    scale = dict(SCALE)
    for spec in a.scale:
        k, _, v = spec.partition("=")
        scale[k] = float(v)
    geom = dict(natural="".join(f"\\expandafter\\def\\csname wdscale@{k}\\endcsname{{{v}}}\\expandafter\\def\\csname wdnat@{k}\\endcsname{{{COL_R_PT / v:.1f}pt}}\n"
                                for k, v in scale.items()),
                top=TOP, height=HEIGHT, col_l=f"{COL_L:g}", col_r=COL_R, margin=MARGIN, xr=16 - MARGIN - COL_R, col_m=16 - 2 * MARGIN)
    def write():
        open(texf, "w").write(_fill(PREAMBLE, geom) + "\n".join(frames) + "\\end{document}\n")
    write()
    log = pdflatex(texf, halt=False)
    # a plexus pane that does not fit above its insights is cut at the ROW above the one it would be clipped through: the cut height the typeset run reports
    # (and the layout it used) decides how many whole dynamics rows are taken out of that slide's pane, and the deck is typeset again; up to six times
    log_notes += [f"{st}: plexus pane natural height {float(nat):.0f} pt, layout {lay}, of {HEIGHT * 72.27 / 2.54:.0f} pt of column" for k, st, lay, nat in re.findall(r"WDCUT kind=(\w+) step=(\d+) cut=[\d.]+ cutw=[\d.]+ lay=(\w) nat=([\d.]+)", log) if k == "plexus"]
    cutre = r"WDCUT kind=(\w+) step=(\d+) cut=([\d.]+) cutw=([\d.]+) lay=(\w)"
    dropped: dict = {}
    for _ in range(6):
        todo = {st: (float(cut), lay) for k, st, cut, cw, lay in re.findall(cutre, log) if k == "plexus" and float(cut) > 0.5}
        if not todo:
            break
        for i, f in enumerate(frames):
            if i == fb_idx:
                continue
            m = re.search(r"\\wdpane\{plexus\}\{(\d+)\}", f)
            if m and m.group(1) in todo:
                frames[i] = drop_rows(f, todo[m.group(1)][0] / scale["plexus"], todo[m.group(1)][1])
                dropped[m.group(1)] = dropped.get(m.group(1), 0) + 1
        write()
        log = pdflatex(texf)
    log_notes += [f"{st}: plexus pane ends on a whole dynamics row above the insights ({n} pass{'es' if n > 1 else ''} of row dropping)" for st, n in sorted(dropped.items())]
    pdflatex(texf)
    log = pdflatex(texf)
    pdf = texf[:-4] + ".pdf"
    errs = [ln for ln in log.splitlines() if ln.startswith("!")]
    if not os.path.exists(pdf) or errs:
        print("\n".join(errs[:10]) or log[-1500:])
        raise SystemExit("pdflatex failed; see " + texf)
    if os.path.abspath(pdf) != out:
        shutil.copyfile(pdf, out)
    print("\n".join("[deck] " + n for n in log_notes))
    used = {k for k, _ in re.findall(r"WDCUT kind=(\w+) step=(\d+)", log)}
    for k in sorted(used):
        print(f"[deck] {k} pane: fixed scale {scale[k]:g}, i.e. text of {NATURAL_PT.get(k, 10) * scale[k]:.1f} pt, column {COL_R:g} cm wide")
    for k, st, cut, cw in re.findall(r"WDCUT kind=(\w+) step=(\d+) cut=([\d.]+) cutw=([\d.]+)", log):
        if float(cut) > 0.5 or float(cw) > 0.5:
            print(f"[deck] TRUNCATED {st} {k}: {float(cut) / (HEIGHT * 72.27 / 2.54 + float(cut)):.0%} of its height cut off the bottom" + (f", {float(cw):.0f} pt off the right" if float(cw) > 0.5 else ""))
    if _UNKNOWN:
        print("[deck] characters the font cannot set were replaced by '?': " + " ".join(sorted(_UNKNOWN)))
    print(f"[deck] {len(frames)} slides -> {out}")


if __name__ == "__main__":
    main()
