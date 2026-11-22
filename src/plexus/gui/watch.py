"""A read-only window onto what this session is doing to the page -- served BY the page, at
`/watch`, so it needs no second port.

WHY IT LIVES HERE AND NOT ON A PORT OF ITS OWN. It did have its own port, and the port was the
problem: a server bound to 0.0.0.0 inside this container is still invisible until the editor
forwards it, and an unforwarded port reads as ERR_CONNECTION_REFUSED, which looks exactly like a
crashed server. `/watch` is on the port that already works.

IT STILL CANNOT DISTURB THE RUN. The hazard was never the port, it was the MAIN page: it polls
`/api/scene/state`, sees a version this session's build bumped, and re-seeds to match -- and a
re-seed stops an in-flight run. This page polls nothing but the four routes below, all of which
only read files off disk. There is no route here that changes anything.

THE PROBLEM IT SOLVES. The page and the API drive ONE scene on ONE VTK thread, so a person
watching at `/` and a session building through `/api/...` fight: the page polls, sees the version
the session's build just bumped, re-seeds itself to match, and a re-seed STOPS an in-flight run.
Measured three times in a row -- `[view] a run was in flight and was stopped so the scene could be
re-seeded` -- and it reads exactly like a run stuck at frame 0.

So the session works on its own server and this shows what it sees: the newest picture it took and
the journal of what it did. Read-only by construction -- there is no route here that changes
anything -- so watching cannot disturb the work.

It serves what `gui_drive.py` writes, ONE FOLDER PER KIND OF OUTPUT with the step number as the
filename, so a step reads across the folders and a kind reads down one:
  builder/png/NNNN.png     the picture
  builder/spec/NNNN.yaml   the spec that made it
  builder/why/NNNN.txt     the time and the reason
  builder/mp4/NNNN.mp4     the movie, when the step ran one
  <record>/journal.txt            one line per action, per experiment

A STEP IS FOUR FILES, not one. The picture says what the scene looked like, the yaml says what
was built, the why says what it was for and when, and the mp4 -- when the step ran one -- says
what it did. Walking prev/next walks all four together, because any one of them alone is a
record of nothing.
"""
from __future__ import annotations

import glob
import html
import json
import os
import re
import struct
import threading
import time
# up from src/plexus/gui/ to the repo
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# WHICH EXPERIMENT'S RECORD THIS IS. The builder used to be one flat pile at `builder/`, which
# works until a second line of work starts and its steps interleave with the first -- the record
# then reads as one story told by two people. `PLEXUS_BUILDER` names the folder, so a new
# experiment is a new directory rather than a new numbering scheme, and the watcher and the writer
# read the SAME variable so they cannot drift apart.
HISTORY = os.environ.get("PLEXUS_BUILDER") or os.path.join(REPO, "experiments", "exp05_spheroid_from_words")
SHOTS = os.path.join(HISTORY, "png")
# ONE WATCHER, EVERY RECORD. `PLEXUS_BUILDER` is only the record a page opens on: the page names the record it
# wants with `rec=<root>/<folder>` on every call (the human, 2026-09-26: "one watcher can toggle between
# experiments in experiments/ and builder/"), so two tabs on one server can watch two experiments and neither
# moves the other -- a server-wide switch would have. A record is any folder under these roots with the record
# layout (a `png/` of steps); `rec` is looked up in that list, never joined as a path, so a query string reaches
# nothing else on disk.
RECORD_ROOTS = ("experiments", "builder")


def records():
    """Every record the watcher can show: [{id, root, name, n}], newest activity first within each root."""
    out = []
    for root in RECORD_ROOTS:
        base = os.path.join(REPO, root)
        if not os.path.isdir(base):
            continue
        rows = []
        for name in os.listdir(base):
            d = os.path.join(base, name)
            png = os.path.join(d, "png")
            # AN EXPERIMENT IS LISTED FROM ITS FIRST DAY, before it has a step (the human, 2026-09-26: "add them to
            # the watcher selector even if still empty" -- exp06, 09, 14, 15 had just been launched): an experiment
            # folder by name (`experiments/expNN_*`, `builder/exp_*`), or any folder that already holds steps
            # ONLY PICTURES THAT EXIST: a record being rebuilt by another session holds, for a moment, a link whose
            # picture is gone (exp14's png/0013.png, 2026-09-26) -- its mtime raised and took the whole list down
            # NAMES ONLY: the pictures are links into /groups, and statting each of 600 of them made this list take 30 s
            # (the drop-down looked empty meanwhile); a dangling link is skipped where a step is read, in `shots`
            steps = ([f for f in os.listdir(png) if len(f) == 8 and f.endswith(".png") and f[:4].isdigit()]
                     if os.path.isdir(png) else [])
            is_exp = os.path.isdir(d) and (name[:3] == "exp" and name[3:5].isdigit() if root == "experiments"
                                           else name.startswith("exp_"))
            if not steps and not is_exp:
                continue
            rows.append({"id": f"{root}/{name}", "root": root, "name": name, "n": len(steps)})
        out += sorted(rows, key=lambda r: r["name"])
    return out


def _rec_id(folder: str) -> str:
    rel = os.path.relpath(os.path.realpath(folder), os.path.realpath(REPO))
    return rel if not rel.startswith("..") else folder


def record_dir(q=None) -> str:
    """The record a request asks for (`rec=<root>/<folder>`, as `records()` names them), else the server's own. Checked by one stat of the folder, not by listing
    `records()`: that list counts the steps of every record on a network disk, 7 s when the machine is quiet and over 100 s when it is loaded, and every
    request -- the state, each pane -- waited for it (Cedric, 2026-11-07: a click on previous step changed nothing for seconds)."""
    rec = ((q or {}).get("rec", [""])[0] or "").strip()
    for root in RECORD_ROOTS:
        name = rec[len(root) + 1:] if rec.startswith(root + "/") else ""
        if name and "/" not in name and name not in (".", "..") and os.path.isdir(os.path.join(REPO, rec)):
            return os.path.join(REPO, rec)
    return HISTORY


def journal_path(folder: str) -> str:
    return os.path.join(folder, "journal.txt")
# THE JOURNAL LIVES INSIDE THE RECORD, so it is per-experiment by construction. It was a fixed
# path under log/gui_runs/, which two sessions running at once both appended to -- the watcher on
# 8826 then showed the cilium campaign's lines interleaved with the bacterium's, and neither
# session could tell which line was its own. One folder per experiment already isolates the four
# step files; the journal is the fifth thing that has to be there.
JOURNAL = os.path.join(HISTORY, "journal.txt")


_SHOTS_CACHE: dict = {}


def shots(folder: str | None = None):
    """Every picture taken, oldest first -- the order they were made, which is the order to walk.

    Sorted by NAME here, not by time: these names are a step counter this session assigns in the
    order it builds, and that order is the thing to walk. Modification time would reorder the
    history if a file were ever touched or copied.
    """
    # a link whose picture is gone (a record mid-rebuild) is not a step: every reader of a step stats its picture.
    # CACHED 10 s per folder: the page polls every 2 s, and statting 150 links into /groups took ~7 s a poll.
    key = folder or HISTORY
    hit = _SHOTS_CACHE.get(key)
    if hit is not None and time.time() - hit[0] < 10.0:
        return hit[1]
    out = sorted(f for f in glob.glob(os.path.join(key, "png", "[0-9][0-9][0-9][0-9].png")) if os.path.exists(f))
    _SHOTS_CACHE[key] = (time.time(), out)
    return out


def sidecar(png: str, kind: str) -> str:
    """The step's other output of kind `kind`, by number. The picture is the spine of the walk and
    everything else is found from it."""
    ext = {"spec": ".yaml", "why": ".txt", "mp4": ".mp4", "caption": ".txt", "story": ".json"}[kind]
    # the record the picture is IN (<record>/png/NNNN.png), not the server's default one
    return os.path.join(os.path.dirname(os.path.dirname(png)), kind, os.path.basename(png)[:4] + ext)


def _read(path: str) -> str:
    """The text beside a picture, or nothing. A step written before this record existed simply has
    no why and no spec, and that is shown as such rather than as an error."""
    if not os.path.exists(path):
        return ""
    with open(path, errors="replace") as f:
        return f.read()


def newest_shot():
    fs = shots()
    return fs[-1] if fs else None


# --------------------------------------------------------------------------- the `plexus` pane
# THE SPEC AS THE PAPER'S THREE QUESTIONS, not as 250 lines of YAML. The `spec` tab already shows
# the file verbatim, which is the right thing to keep -- it is the artefact, and it is what a
# re-run would read -- but it is not what a person wants to look at while watching a run: the box
# and the timestep are on line 4, the sets are two hundred lines further down, and the activities
# are named but not explained. `tools/spec_summary.py` answers the three questions plexus2.tex
# asks in its own order -- HEADER what is being run, ENTITIES what exists, OPERATORS what happens,
# each with its equation and the parameter values THIS spec gives it -- and everything in it is
# read from the spec and from the live operator registry, so it cannot drift from either.
#
# ON DEMAND AND CACHED, for two reasons. `summarise` parses the spec and walks the registry, and
# the page polls every 1.5 s: computing it on every tick would pay that cost forever for a pane
# nobody may have open. The cache is keyed on the file's mtime, so a step whose spec is rewritten
# is re-summarised and a step that is merely looked at again is free.
_SUMMARY_CACHE: dict = {}


def _why_context(why_path: str | None) -> str:
    """The step's `why`, without the time / spec / camera / movie header lines a recorder adds: the one
    place a run says what it is MEANT to be, handed to the story's agent as context."""
    if not why_path or not os.path.exists(why_path):
        return ""
    keep = []
    for ln in open(why_path, errors="replace").read().splitlines():
        head = ln.split(None, 1)[0] if ln.strip() else ""
        if head in ("time", "spec", "camera", "movie", "render"):
            continue
        keep.append(ln[4:].strip() if head == "why" else ln.strip())
    # THE TWO SHOTS OF ONE RUN SHARE ONE STORY: the driver adds "(before the run)" / "(after the run)"
    # to the same why, and nothing else about the run differs between them.
    txt = re.sub(r"\s*\((before|after) the run\)\s*$", "", " ".join(k for k in keep if k))
    return txt[:500]


def _story_module():
    """`tools/spec_story.py`, imported by path and kept (it shares `spec_summary`'s module)."""
    m = globals().get("_SPEC_STORY_MOD")
    if m is None:
        import importlib.util
        _summary_module()
        p = os.path.join(REPO, "tools", "spec_story.py")
        spec = importlib.util.spec_from_file_location("plexus_spec_story", p)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        globals()["_SPEC_STORY_MOD"] = m
    return m


def _story(path: str, why_path: str | None, story_path: str | None = None, png: str | None = None):
    """(summary, facts, story or None) for one step. A story SAVED beside the step is used as it is --
    never rebuilt while walking the record; otherwise it is read from the agent's cache, or started in
    the background (and saved beside the step when it lands) and None until then."""
    sm = _story_module()
    s = _summarise(path)
    saved = sm.load_saved(story_path)
    if saved is not None:
        # THE PANE'S RULE: every equation with every variable and constant defined. A story saved before the
        # legend was enforced (or whose agent skipped it) is completed in the background and re-saved.
        sm.ensure_symbols(saved[0], saved[1], story_path)
        return s, saved[0], saved[1]
    F = sm.facts(s, context=_why_context(why_path), diff=sm.spec_diff(sm.previous_spec(png), path))
    st = sm.ensure(F, save_to=story_path)
    # NO "BEING WRITTEN" PANE (the human, 2026-10-18: it sat there for good and blinked): until the agent's story lands -- or when it
    # failed, which is not retried before RETRY_AFTER_S -- the pane shows the story the facts alone give (`fallback_story`)
    return s, F, (st if st is not None else sm.fallback_story(F))


def spec_summary(path: str, q_index: int = 0, rec: str = "", why_path: str | None = None,
                 story_path: str | None = None, png: str | None = None, lead: str = "") -> tuple:
    """`tools/spec_summary.py`'s reading of one spec file as HTML, or a message saying why not.

    Imported BY PATH because `tools/` is not a package -- the same way `spec_summary` itself
    imports `tools/operator_equations.py` for the equation table. A failure here must never reach the
    page as a traceback: this pane is a convenience on a read-only window, and the `spec` tab
    beside it still holds the file itself.
    """
    if not path or not os.path.exists(path):
        return '<div class="ps-g">no spec recorded for this step</div>', False
    try:
        _s, _F, _st = _story(path, why_path, story_path, png)
        ready = not (_st or {}).get("fallback") or not _story_module().pending(_F)
    except Exception:                                            # noqa: BLE001
        _s = _F = _st = None
        ready = True
    key = (path, os.path.getmtime(path), int(q_index or 0), rec, ready,
           os.path.getmtime(story_path) if story_path and os.path.exists(story_path) else 0.0)
    hit = _SUMMARY_CACHE.get(path)
    if hit is not None and hit[0] == key:
        return hit[1], not ready
    # A MEASUREMENT STEP HAS NO SPEC, AND THAT IS NOT AN ERROR. `tools/builder_figure.py` writes a
    # two-line comment stub into `spec/` for every graph it records, so that `_next_index` cannot
    # hand the number to a later run and interleave two things under one index. Summarising it
    # raised `AttributeError: 'NoneType' object has no attribute 'get'` -- yaml parses a file of
    # pure comments to None -- and a watcher reading that would think the tool was broken rather
    # than that this step is a graph. Answered before the parse instead.
    with open(path, errors="replace") as fh:
        body = [ln for ln in fh.read().splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if not body:
        txt = ('<div class="ps-g">This step is a MEASUREMENT, not a run: it has a picture and a '
               'why, and no spec to summarise. What it measures is in the why beside it.</div>')
        _SUMMARY_CACHE[path] = (key, txt)
        return txt, False
    try:
        i = int(q_index or 0)
        sm = _story_module()
        if _F is None:
            raise RuntimeError("the spec could not be read as facts")
        # THE EQUATION'S ADDRESS CARRIES A HASH OF ITS TEXT. `/api/watch/eq?i=&k=` names a POSITION,
        # and the route answers with `Cache-Control: max-age=86400`, so when an operator's equation
        # was corrected the browser kept showing the old image for that position for a day: slide 11
        # read "autocatalytic reactions in cyclic competition" -- the new title -- above Gray-Scott's
        # da/dt and du/dt, while the server was serving May-Leonard's three rates the whole time.
        # With the hash in the URL, new text is a new address and cannot be answered from cache.
        import hashlib as _hl
        _eqs = sm.equations(_F, _st)
        _hashes = [_hl.sha1(f"{EQ_DPI}:{e}".encode()).hexdigest()[:12] for e in _eqs]
        err = sm.error(_F)
        text = sm.render_html(_s, _F, _st, why=_read(why_path) if why_path else "",
                              icon_url=lambda k: f"/api/watch/icon?kind={k}",
                              eq_url=lambda k: f"/api/watch/eq?i={i}&k={k}&h={_hashes[k] if k < len(_hashes) else 0}"
                                               + (f"&rec={rec}" if rec else ""),
                              eq_size=lambda k: eq_size(_eqs[k]) if k < len(_eqs) else (0, 0),
                              sym_url=sym_url,
                              note=(f"the plain-English description could not be written ({err}); the steps "
                                    f"below are in the order the run takes them") if err else "")
    except Exception as e:                                       # noqa: BLE001
        text = (f'<div class="ps-g">spec_summary could not read this step: '
                f'{html.escape(type(e).__name__)}: {html.escape(str(e))}<br>'
                f'The spec itself is in the tab beside this one.</div>')
    if lead:
        text = f'<div class="ps-m" style="font-size:12px;margin-bottom:6px">{html.escape(lead)}</div>' + text
    _SUMMARY_CACHE[path] = (key, text)
    return text, not ready


PAGE = """<!doctype html><html><head><meta charset=utf-8><title>Plexus — watching</title>
<style>
 body{background:#0b0d10;color:#ccd3da;font:13px/1.6 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;padding:14px 18px}
 h1{font-size:15px;font-weight:600;margin:0 0 2px;color:#e6edf3}
 .sub{color:#7d8894;font-size:12px;margin-bottom:10px}
 .wrap{display:flex;gap:18px;align-items:flex-start;flex-wrap:wrap}
 .why{background:#12171d;border-left:3px solid #7fb069;border-radius:0 5px 5px 0;padding:8px 12px;
      margin-bottom:10px;white-space:pre-wrap;font-size:12px;color:#c8d2dc;max-width:min(62vw,820px)}
 .col{display:flex;flex-direction:column;position:relative}
 #lfstamp{position:absolute;top:6px;left:6px;z-index:5;font:11px/1.4 ui-monospace,Consolas,monospace;color:#e6edf3;
   background:rgba(0,0,0,.6);padding:1px 7px;border-radius:3px;pointer-events:none;display:none}
 video{border:1px solid #2a3139;background:#000;max-width:min(62vw,820px)}
 .tabs button{font-size:12px;padding:3px 10px}
 img{border:1px solid #2a3139;background:#000;max-width:min(62vw,820px)}
 #live3d{border-color:#4a6b8a;min-height:320px;min-width:320px;user-select:none;-webkit-user-drag:none}
 pre{background:#0f1319;border:1px solid #222a33;border-radius:5px;padding:10px 12px;
     max-height:74vh;overflow:auto;font-size:11.5px;color:#b9c3cd;flex:1;min-width:300px;margin:0;
     /* WRAPPED, because a `why` is a paragraph and not a log line. Unwrapped, reading one meant
        dragging a horizontal scrollbar left to right for every sentence. `pre-wrap` keeps the
        newlines that make the time/spec/camera header readable while letting long prose fold. */
     white-space:pre-wrap;overflow-wrap:anywhere}
 /* THE `plexus` PANE: a STORY (headline, paragraph), then ENTITIES with a right-aligned count, then
    the GNN -- every step of the run in the story's order, each with its kind's icon (2026-10-11;
    it was ENTITIES then ACTIVITIES grouped by kind, which only the developer could read). It is a PANE and not a `pre`,
    because a summary is a small document and a `pre` can only be a block of text. */
 .ps-pane{background:#0f1319;border:1px solid #222a33;border-radius:5px;padding:12px 14px;
          max-height:74vh;overflow:auto;flex:1;min-width:300px;font-size:12.5px;color:#ccd3da}
 .ps-h{font-size:15px;font-weight:700;color:#f2c94c;margin:14px 0 6px}
 /* THE NAMES ARE WHAT A READER SCANS FOR, so they carry the headings' size, bold and white; the
    headings above them are yellow so the two levels never read as one. */
 .ps-en{color:#ffffff;font-weight:700;font-size:13.5px}
 .ps-h:first-child{margin-top:2px}
 /* THE `insights` PANE: the arrow and the few words after its heading are white and regular, at the heading's size; the body is plain */
 .ps-ins{color:#ffffff;font-weight:400}
 .ps-story p{margin:0 0 8px}
 /* NOT `width:100%`. Stretched to the pane, the count was flung to the far right and `cell` and
    `4,000` were read as two separate columns with a hand's width of nothing between them --
    the row says ONE thing and has to be read as one. `auto` puts the number where the name ends. */
 .ps-e,.ps-a{border-collapse:collapse;width:auto}
 .ps-e td{padding:1px 0;vertical-align:top}
 /* ENTITY NAMES LINE UP WITH ACTIVITY NAMES: the activities table spends its first column on the
    32-px icon (plus 2 px), so the entities table starts that far in. */
 .ps-e{margin-left:34px}
 .ps-n{text-align:right;padding:0 10px 0 22px !important;white-space:nowrap;color:#ccd3da}
 .ps-g{color:#6f7a86}
 /* A CLICK IS ANSWERED AT ONCE: the picture dims and a line above it says what is loading and for how long (Cedric, 2026-11-07) */
 #loading{color:#e0a030;font-size:12px;min-height:16px;margin:2px 0}
 .dim{opacity:.35;transition:opacity .15s}
 /* A RELATION IS NOT A POPULATION, and the row says so in its own colour. `half_edge` is 520,032
    of nothing a reader should count as entities -- it is the map between the two sets above it,
    and reading it as a third body is the one misreading this table can produce. */
 .ps-rel{color:#d08770;font-weight:700;font-size:13.5px}
 /* WHAT EACH ELEMENT CARRIES, under its row: the state an equation below is written in. */
 .ps-st{color:#6f7a86;font-size:11.5px;padding:0 0 3px 0}
 .ps-a td{padding:1px 0;vertical-align:top}
 .ps-i{width:32px;padding-top:4px !important}
 /* NO `invert`. These files are coloured line art on an OPAQUE BLACK square -- invisible on a
    black slide, a dark tile on this pane, and a WHITE tile once inverted. The route strips the
    background instead (alpha = brightness), so what arrives here is already transparent. */
 .ps-i img{height:24px;border:0;background:none;max-width:none}
 .ps-k{color:#6f7a86;padding-top:6px !important;padding-bottom:2px !important}
 .ps-t{color:#ffffff;font-weight:700;font-size:13.5px;padding-left:2px !important}
 /* An equation is a PNG from the same LaTeX the deck uses, white on transparent, so it needs
    neither a border nor a background of its own. Sized in em so it tracks the text beside it. */
 .ps-q{max-width:100%;overflow-x:auto}
 /* THE STORY (tools/spec_story.py): a headline and a paragraph a reader who is not the developer
    can follow, then each GNN step as a name, a sentence, and a small grey line saying when it runs,
    its code name and what it acts on. */
 .ps-hl{font-size:16px;font-weight:700;color:#ffffff;margin:2px 0 4px}
 .ps-story{font-size:13px;line-height:1.55;color:#d6dde4;margin:0 0 8px;max-width:78ch}
 .ps-nm{color:#ffffff;font-weight:700}
 /* ONE LINE, ALWAYS: a description that does not fit is cut with an ellipsis (the human: "no two line"). */
 .ps-w{color:#ccd3da;font-weight:400;font-size:12.5px;max-width:72ch;white-space:nowrap;overflow:hidden;
       text-overflow:ellipsis}
 /* A NAME IS A LINK to its class's source on GitHub, drawn as the name it is, underlined only on hover. */
 a.ps-ln{color:inherit;text-decoration:none}
 a.ps-ln:hover{text-decoration:underline}
 /* THE DIFFERENCE FROM THE PREVIOUS RUN, in blue: yellow belongs to the headings (entities, GNN). */
 .ps-chg{color:#79b8ff;font-size:12.5px;margin:-2px 0 8px;max-width:78ch}
 .ps-m{color:#6f7a86;font-weight:400;font-size:11px;margin-bottom:6px}
 details.ps-q summary{color:#6f7a86;font-weight:400;font-size:11px;cursor:pointer}
 /* A COLOUR IN THE RENDERING SECTION, as a small swatch beside its name. */
 .ps-sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:3px;vertical-align:-1px;
        border:1px solid #2f3945}
 /* THE EQUATION'S SYMBOLS, under it: each as it appears, then what it is in plain English. */
 .ps-sym{color:#aab4be;font-weight:400;font-size:11.5px;line-height:1.45}
 .ps-sy{color:#ffffff;font-weight:600;display:inline-block;min-width:3.2em}
 /* A LEGEND SYMBOL IS TYPESET like the equation above it (the same LaTeX, the same dpi), inline with its meaning. */
 .ps-syi{border:0;background:none;vertical-align:middle;opacity:.92;margin-right:4px}
 /* THE MATH PANE: one equation per row, its locating word in a column to the right, then the legends. */
 .ps-mq td{padding:3px 0;vertical-align:middle}
 .ps-mi img{border:0;background:none;max-width:none;opacity:.92}
 .ps-mw{color:#f2c94c;font-weight:600;font-size:12.5px;padding-left:18px !important;white-space:nowrap}
 .ps-ex{margin:4px 0 8px 18px;padding:7px 11px;max-width:760px;background:#12171d;border-left:3px solid #f2c94c;border-radius:0 5px 5px 0;font-size:12.5px;line-height:1.5;color:#c9d1d9;white-space:normal}
 .ps-ex,.ps-ex *{font-weight:400 !important}
 .ps-exh{color:#f2c94c}
 .ps-ex img.ps-tx{border:0;background:none;max-width:100%;vertical-align:middle;opacity:.95}
 .ps-ex code{background:#1b222b;padding:0 4px;border-radius:3px;font-size:11.5px}
 .ps-mrow:hover td{background:#141a21}
 .ps-ml{margin-left:34px}
 .ps-ml td{padding:1px 10px 1px 0;vertical-align:top}
 .ps-q img{border:0;background:none;max-width:none;margin:1px 0 5px;opacity:.92}
 .t{color:#6f7a86}
 b{color:#7fb069;font-weight:600}
 button{background:#1d2530;color:#ccd3da;border:1px solid #2f3945;border-radius:4px;
        padding:4px 12px;margin-right:6px;cursor:pointer;font:inherit}
 button:hover{background:#26303c}
 button.on{background:#2d4a2d;border-color:#3f6b3f}
 button.ic{padding:2px 4px;margin-right:3px;vertical-align:middle;line-height:0}
 button.ic svg{width:18px;height:18px;fill:none;stroke:currentColor;stroke-width:1.5;stroke-linejoin:round;stroke-linecap:round}

 /* THE FLAT LOOK (Cedric, 2026-11-22): the page, the movie and every pane are the grey of the scene (141, 152, 165), no borders, white text; a button is
    one tone darker, the one that is on is near-white; the pane tabs sit on the row of the selection buttons, above their pane. */
 :root{--bg:#8d98a5;--btn:#707b88;--btn2:#7a8593;--on:#ecf0f4;--ink:#263038;--sub:#e0e6ec;--card:#7e8996}
 body{background:var(--bg);color:#ffffff;display:grid;grid-template-columns:auto minmax(300px,1fr);column-gap:18px;align-items:start}
 body>h1,body>.sub,body>.t,body>#d3bar{grid-column:1 / -1}
 body>#ctl{grid-column:1;grid-row:3}
 body>.tabs{grid-column:2;grid-row:3;margin-bottom:8px}
 .wrap{display:contents}
 .wrap>.col:first-child{grid-column:1}
 .wrap>.col:last-child{grid-column:2;min-width:300px}
 h1{color:#ffffff}
 .sub,.t,.ps-g,.ps-k,.ps-m,.ps-st,.ps-sym,details.ps-q summary{color:var(--sub)}
 .why{background:var(--card);border-left:3px solid #ffffff;color:#ffffff}
 video,img{border:0;background:none}
 .ps-pane,pre{background:var(--bg);border:0;color:#ffffff}
 .ps-h,.ps-mw,.ps-exh{color:#ffffff}
 .ps-n,.ps-w,.ps-story{color:#ffffff}
 .ps-ex{background:var(--card);border-left:3px solid #ffffff;color:#ffffff}
 .ps-ex code{background:var(--btn)}
 .ps-mrow:hover td{background:var(--card)}
 .ps-chg{color:#ffffff}
 .ps-sw{border:1px solid var(--btn)}
 button,select{background:var(--btn);color:#ffffff;border:0;border-radius:4px}
 button:hover{background:var(--btn2)}
 button.on{background:var(--on);color:var(--ink)}
 #loading{color:#ffffff}
 b{color:#ffffff}
</style></head><body>
<h1>Plexus &mdash; watching the session</h1>
<div class=sub id=meta>&hellip;</div>
<div class=row id=ctl style="margin-bottom:8px">
 <select id=recsel onchange="pickRec(this.value)" title="which experiment's record to watch"
         style="padding:3px 6px;margin-right:10px;font:inherit"></select>
 <button onclick="step(-10)" title="back ten steps">&laquo;</button>
 <button onclick="step(-1)">&lt;</button>
 <button onclick="step(1)">&gt;</button>
 <button onclick="step(10)" title="forward ten steps">&raquo;</button>
 <button onclick="live()" id=livebtn>live</button>
 <select id=lfsel style="display:none;background:#1d2530;color:#ccd3da;border:1px solid #2f3945;border-radius:4px;padding:3px 6px;margin-right:10px;font:inherit"
         onchange="lfRun=this.value;showLatest()" title="which running job"></select>
 <button onclick="open3d()" id=d3btn title="4D: the run's own frames, turnable, with the time slider">4D</button>
 <span id=d3cams style="display:none">
  <button class=ic onclick="view3d('top')" title="top view: looking down the up axis"><svg viewBox="0 0 20 20"><rect x="3" y="14" width="14" height="3"/><path d="M10 2v9M6.5 7.5 10 11l3.5-3.5"/></svg></button>
  <button class=ic onclick="view3d('bottom')" title="bottom view: looking up the up axis"><svg viewBox="0 0 20 20"><rect x="3" y="3" width="14" height="3"/><path d="M10 18V9M6.5 12.5 10 9l3.5 3.5"/></svg></button>
  <button class=ic onclick="view3d('nominal')" title="oblique: the movie's own camera"><svg viewBox="0 0 20 20"><path d="M4 14l7-3.5 6 2.5-7 3.5z"/><path d="M2 2l6 6M4.5 8.5H8V5"/></svg></button>
  <button class=ic onclick="view3d('side')" title="side view: level with the membrane"><svg viewBox="0 0 20 20"><rect x="9" y="8.5" width="9" height="3"/><path d="M1 10h6M4.5 6.5 8 10l-3.5 3.5"/></svg></button>
  <button class=ic id=d3glass onclick="glass3d()" title="glass / solid: translucent skins with the atoms inside, or opaque skins alone"><svg viewBox="0 0 20 20"><circle cx="10" cy="10" r="7"/><circle cx="8" cy="9" r="1"/><circle cx="12" cy="11" r="1"/><circle cx="10" cy="13" r="1"/></svg></button>
  <button class=ic id=d3cut onclick="cut3d()" title="slice: keep a thick disc of the scene through its centre (a rectangle seen from the side); the two buttons after it thin or thicken it"><svg viewBox="0 0 20 20"><circle cx="10" cy="10" r="8" stroke-dasharray="2.2 2" style="opacity:.55"/><path d="M2.4 7.5H17.6V12.5H2.4Z" style="fill:currentColor;fill-opacity:.38"/></svg></button>
  <button class=ic id=d3thin onclick="slab3d(0.67)" title="slice thinner: the slab keeps a thinner disc" style="display:none"><svg viewBox="0 0 20 20"><rect x="3" y="8" width="14" height="4"/><path d="M10 2v4M8 4.5l2 2 2-2M10 18v-4M8 15.5l2-2 2 2"/></svg></button>
  <button class=ic id=d3thick onclick="slab3d(1.5)" title="slice thicker: the slab keeps a thicker disc" style="display:none"><svg viewBox="0 0 20 20"><rect x="3" y="5" width="14" height="10"/><path d="M10 1v4M8 3l2-2 2 2M10 19v-4M8 17l2 2 2-2"/></svg></button>
  <button class=ic id=d3def onclick="def3d()" title="deformation: colour the dots by |F|, the size of their deformation gradient against the first frame (red-white-blue, red = compressed, blue = stretched, white = undeformed), with no outline, instead of by cell" style="display:none"><svg viewBox="0 0 20 20"><path d="M10 2c.9 3.3-3.5 5.1-3.5 9.2a3.5 3.5 0 0 0 7 0c0-2-.9-3.3-1.9-4.5-.1 1.4-.8 2.4-1.6 2.7.6-2.7.3-5-.1-7.4z"/></svg></button>
  <button class=ic id=d3look onclick="look3d()" style="display:none"></button>
  <button class=ic id=d3dot1 onclick="dot3d(0.6)" title="dot size: small (0.6 x the view's own)" style="display:none"><svg viewBox="0 0 20 20" style="fill:currentColor;stroke:none"><circle cx="7" cy="8" r="1.1"/><circle cx="12" cy="6" r="1.1"/><circle cx="9" cy="13" r="1.1"/><circle cx="14" cy="12" r="1.1"/></svg></button>
  <button class=ic id=d3dot2 onclick="dot3d(1)" title="dot size: the view's own" style="display:none"><svg viewBox="0 0 20 20" style="fill:currentColor;stroke:none"><circle cx="7" cy="8" r="1.8"/><circle cx="12" cy="6" r="1.8"/><circle cx="9" cy="13" r="1.8"/><circle cx="14" cy="12" r="1.8"/></svg></button>
  <button class=ic id=d3dot3 onclick="dot3d(1.7)" title="dot size: large (1.7 x the view's own)" style="display:none"><svg viewBox="0 0 20 20" style="fill:currentColor;stroke:none"><circle cx="7" cy="8" r="2.7"/><circle cx="12" cy="6" r="2.7"/><circle cx="9" cy="13" r="2.7"/><circle cx="14" cy="12" r="2.7"/></svg></button>
  <button class=ic id=d3iso1 onclick="iso3d(0.5)" title="surface threshold: loose (half the spec's level -- a fuller body, thin sheets kept)" style="display:none"><svg viewBox="0 0 20 20"><circle cx="10" cy="10" r="8"/><circle cx="10" cy="10" r="2.5" style="fill:currentColor;stroke:none"/></svg></button>
  <button class=ic id=d3iso2 onclick="iso3d(1)" title="surface threshold: the spec's own level" style="display:none"><svg viewBox="0 0 20 20"><circle cx="10" cy="10" r="6"/><circle cx="10" cy="10" r="2.5" style="fill:currentColor;stroke:none"/></svg></button>
  <button class=ic id=d3iso3 onclick="iso3d(2)" title="surface threshold: tight (twice the spec's level -- only the dense core)" style="display:none"><svg viewBox="0 0 20 20"><circle cx="10" cy="10" r="4"/><circle cx="10" cy="10" r="2.5" style="fill:currentColor;stroke:none"/></svg></button>
 </span>
 <span id=pos class=t style="display:none"></span>
 <span id=d3note class=t></span>
 <span id=drvnote class=t style="color:#7aa2f7"></span>
</div>
<div class="row tabs">
 <button onclick="pane('plexus')" id=b_plexus class=on>plexus</button>
 <button onclick="pane('math')" id=b_math>math</button>
 <button onclick="pane('hierarchy')" id=b_hierarchy>hierarchy</button>
 <button onclick="pane('spec')" id=b_spec>spec</button>
 <button onclick="pane('insights')" id=b_insights>insights</button>
  </div>
<div class=row id=d3bar style="margin-bottom:8px;display:none">
 <span class=t>frame</span>
 <input type=range id=d3frame min=0 max=0 value=0 step=1 style="width:520px" oninput="scrub3d(this.value)">
 <span id=d3ft class=t></span>
 <span class=t>&nbsp;(&larr; &rarr; step one frame)</span>
</div>
<div class=t style="font-size:10.5px;margin:-2px 0 8px"><span id=mpos></span></div>
<div class=wrap>
 <div class=col>
  <div id=lfstamp></div>
  <div id=loading></div>
  <img id=shot>
  <img id=live3d style="display:none;cursor:grab">
  <video id=mov controls loop muted autoplay playsinline style="display:none"></video>
  <div class=row id=spdbar style="margin-top:4px;gap:4px">
   <span class=t>speed</span>
   <button onclick="setSpd(0.5)" id=spd_0.5>x0.5</button>
   <button onclick="setSpd(1)" id=spd_1 class=on>x1</button>
   <button onclick="setSpd(2)" id=spd_2>x2</button>
   <button onclick="setSpd(3)" id=spd_3>x3</button>
   <button onclick="setSpd(5)" id=spd_5>x5</button>
  </div>
  <script>
  // PLAYBACK SPEED (Cedric, 2026-10-13): one rate for every movie the pane shows; a new source resets a
  // <video>'s rate to 1, so it is re-applied on every loadedmetadata.
  window.spd = 1;
  function setSpd(r) {
    window.spd = r;
    const mv = document.getElementById('mov'); mv.playbackRate = r;
    document.querySelectorAll('#spdbar button').forEach(b => b.classList.toggle('on', b.id === 'spd_' + r));
  }
  document.getElementById('mov').addEventListener('loadedmetadata', e => { e.target.playbackRate = window.spd; });
  </script>
 </div>
 <div class=col style="flex:1;min-width:300px">
  <div id=plexus class=ps-pane></div>
  <div id=math class=ps-pane style="display:none"></div>
  <div id=hierarchy class=ps-pane style="display:none"></div>
  <pre id=spec style="display:none"></pre>
  <div id=insights class=ps-pane style="display:none"></div>
 </div>
</div>
<script>
// `idx` is which picture is shown; -1 means FOLLOW THE NEWEST, which is what a watcher wants by
// default. Pressing prev/next pins it to one shot so the view stops jumping while it is read.
// `shown` OPENS ON `plexus`, not on `why`. The why says what a step was FOR and the plexus pane
// says what it IS -- the box, the sets and the activities with their equations and this spec's own
// parameter values -- and that is the thing to have in view while a picture is being looked at.
let idx = -1, total = 0, shown = 'plexus';
// WHICH RECORD THIS TAB WATCHES: `rec=<root>/<folder>` (experiments/... or builder/...), kept in the page's own URL
// so a reload stays on it, and sent with every call -- the server holds no "current record", so two tabs on one
// watcher can watch two experiments. Empty = the record the server was started on (PLEXUS_BUILDER).
let rec = new URLSearchParams(location.search).get('rec') || '';
function R(){ return rec ? '&rec=' + encodeURIComponent(rec) : ''; }
async function loadRecords(){
 try{
  const j = await (await fetch('/api/watch/records')).json();
  const sel = document.getElementById('recsel');
  const cur = rec || j.default;
  sel.innerHTML = '';
  for(const root of ['experiments', 'builder']){
   const g = document.createElement('optgroup'); g.label = root;
   for(const r of j.records.filter(x => x.root === root)){
    const o = document.createElement('option'); o.value = r.id; o.textContent = r.name + '  (' + r.n + ')';
    if(r.id === cur) o.selected = true;
    g.appendChild(o);
   }
   if(g.children.length) sel.appendChild(g);
  }
 }catch(e){ /* the list is a convenience; the page still shows its record */ }
}
function pickRec(v){
 rec = v;
 history.replaceState(null, '', location.pathname + '?rec=' + encodeURIComponent(v));
 if(d3) close3d();
 idx = -1; total = 0; movies = []; pxAt = null; pxWant = null; pxShown = null; mxAt = null; mxWant = null; mxShown = null; hiAt = null; hiWant = null; hiShown = null; czAt = null; czWant = null;
 const mv = document.getElementById('mov'); mv.removeAttribute('src'); mv.dataset.k = '';
 document.getElementById('shot').removeAttribute('src');
 tick();
}
loadRecords();
// THE 3-D VIEW. A step's picture is one camera; this re-opens that step's own SPEC on this
// server, seeds it, and serves the live render so it can be turned and zoomed. Server-side VTK
// streamed as a PNG, which is what the page itself does -- there is no WebGL here and a point
// cloud of 100,000 particles is not something to ship to a browser frame by frame.
//
// IT IS THE ONLY THING IN THIS PAGE THAT CHANGES ANYTHING, and it says so: seeding a scene takes
// over this server's one VTK thread. That is why it is a button and not the default.
let d3 = null;   // {i, azim, elev, zoom, drag}
function open3d(){
 // LATEST FRAME ON: the 3-D view opens on the running job's saved state, not on a recorded step
 const live = (lfOn && lfRun && lfState) ? lfRun : null;
 if(total === 0 && !live) return;
 const i = live ? -2 : ((idx < 0) ? total - 1 : idx);
 if(d3 && d3.i === i){ close3d(); return; }
 // THE FRAME THE MOVIE IS AT, as a fraction of the run: the 3-D view opens there, not at the seed. A still (the
 // run's own 3d.png) is its last frame.
 const mv = document.getElementById('mov');
 let f = 1.0;
 if(mv && mv.style.display !== 'none' && mv.duration > 0) f = Math.max(0, Math.min(1, mv.currentTime / mv.duration));
 d3 = {i: i, azim: 20, elev: 8, zoom: 1.3, roll: 180, drag: null, name: null, frame: null, nk: 0, tns: null, f: f, cut: false, glass: false, look: null, shown: false, live: live, lt: 0};
 if(live) d3.frame = null;
 document.getElementById('d3cut').className = 'ic';
 document.getElementById('d3thin').style.display = 'none'; document.getElementById('d3thick').style.display = 'none';
 document.getElementById('d3glass').className = 'ic';
 // A COUNTER WHILE VTK BUILDS, because a button that does nothing visible for six seconds is
 // indistinguishable from a broken one. The server prints the same thing to its log.
 const t0 = Date.now();
 const note = document.getElementById('d3note');
 const tim = setInterval(() => {
  if(d3) note.textContent = ' — VTK is building the scene… ' + ((Date.now()-t0)/1000).toFixed(1) + ' s';
 }, 200);
 note.textContent = ' — VTK is building the scene… 0.0 s';
 loading('4D: VTK is building the scene and reading the recorded frames');          // the line above the picture, with the seconds, until the first frame is drawn
 fetch(live ? '/api/watch/open3d?live=' + encodeURIComponent(live) + R() : '/api/watch/open3d?i=' + i + R()).then(r => r.json()).then(j => {
  clearInterval(tim);
  if(j.error){
   note.textContent = ' — ' + j.error; d3 = null; loading('');
   // THE SERVER RESTARTS ITSELF when the step needs operators added after it started: wait until it answers
   // again, then ask for the same step's 3-D view once more
   //
   // EACH CHECK TIMES OUT. Through VS Code's port forwarding a request to a server that is down can HANG rather than
   // fail, and one hung check stalled the whole retry: exp11's step 18 restarted the watcher and never reopened
   // (2026-09-27). Three seconds a check, for up to two minutes, the wait shown as it runs.
   if(j.restarting){
    const t_r = Date.now();
    const again = () => {
     const s_ = Math.round((Date.now() - t_r) / 1000);
     if(s_ > 120){ note.textContent = ' — the watcher did not come back within 2 min; reload the page'; return; }
     note.textContent = ' — the watcher is restarting to load new operators… ' + s_ + ' s (the 3D view reopens by itself)';
     const ac = new AbortController(); const to = setTimeout(() => ac.abort(), 3000);
     fetch('/api/watch/records', {signal: ac.signal, cache: 'no-store'})
       .then(r => { clearTimeout(to); if(r.ok) setTimeout(open3d, 500); else setTimeout(again, 1500); })
       .catch(() => { clearTimeout(to); setTimeout(again, 1500); });
    };
    setTimeout(again, 3000);
   }
   return;
  }
  // THE NAME IS CARRIED ON EVERY RENDER, because opening a spec is not seeding it: the render
  // route builds the scene itself when it is told which one, and without that it answers "no
  // scene is open; seed one first".
  d3.name = j.name;
  d3.nk = j.n_kept || 0; d3.tns = j.total_ns || null;
  if(j.live){ d3.lt = j.live.mtime; lfStamp3d(live); }
  // THE MOVIE'S OWN CAMERA when the spec states one: the view opens where the movie looks, and the oblique button
  // returns there. The fixed default above (roll 180) was chosen for the Platynereis larva and drew exp04 upside down.
  if(j.camera){ d3.cam = j.camera; d3.azim = j.camera.azim; d3.elev = j.camera.elev; d3.zoom = j.camera.zoom; d3.roll = j.camera.roll; }
  document.getElementById('d3cams').style.display = '';
  d3.glass = !!j.glass;                       // the look the step's own spec draws; the button flips it
  document.getElementById('d3glass').className = 'ic' + (d3.glass ? ' on' : '');
  d3.skin = !!j.skin_look;                    // a scene of skins (exp02, exp04): the same dots / surface button, dots hide the skins; it opens on its surface
  d3.look = j.look || j.skin_look || null;    // 'dots' first when the view has a surface to switch to; null: dots only
  // ONE BUTTON IN THAT PLACE: dots / surface when the view can contour (every MPM point cloud), else glass / solid for
  // a scene of skins (Cedric, 2026-10-27: the glass button read as dots/surface and did nothing on a point cloud)
  glassButton();
  d3.iso = 1; d3.dot = 0.6;                   // opens on the small dots (the first of the three size buttons)
  look3dButton();
  isoButtons();
  dotButtons(); defButton();
  // FEWER THAN FIVE KEPT FRAMES HAVE NO SLIDER: three frames are not a run to scrub (the movie is not played either, MIN_MOVIE_FRAMES
  // in this file), so the view stays on the scene as seeded and the time bar is not shown.
  if(d3.nk >= 5){
   d3.frame = Math.round(d3.f * (d3.nk - 1));
   const sl = document.getElementById('d3frame');
   sl.max = d3.nk - 1; sl.value = d3.frame;
   document.getElementById('d3bar').style.display = '';
   label3d();
  }
  note.textContent = ' — step ' + curShot.replace(/\\.png$/, '') + ' · ' + j.name + ' ready in ' + (j.seconds || '?')
    + ' s: drag to turn, wheel to zoom, the slider moves through the run, 3D again to close';
  // THE MOVIE STAYS UP UNTIL THE FIRST 3-D FRAME HAS ARRIVED: the panel was swapped to an empty <img> here, so a
  // render that took minutes (exp25 step 0013's last frame, 2026-10-13) was an empty black panel all that time
  drvEv('4d-scene-ready', (j.seconds || '?') + ' s on the server, ' + (j.n_kept || 0) + ' frames kept');
  loading('4D: drawing the first frame');
  render3d();
 }).catch(e => { clearInterval(tim); note.textContent = ' — 4D failed: ' + e; d3 = null; loading(''); });
}
// THE RUN'S TIME AT THE SLIDER: kept frame k of nk spans the whole run, so t = k / (nk - 1) x its length
// IN THE RUN'S OWN SCALE: a channel runs for nanoseconds and a tissue for days -- exp13's growth read
// "of 960,600,000,000,000 ns" when the label assumed the first
function tunit(ns){
 const u = [[86400e9, 'days'], [3600e9, 'h'], [60e9, 'min'], [1e9, 's'], [1e6, 'ms'], [1e3, 'µs'], [1, 'ns']];
 for(const [f, n] of u) if(ns >= f) return [f, n];
 return [1, 'ns'];
}
function label3d(){
 if(!d3 || d3.frame === null) return;
 let t = '';
 if(d3.tns){
  const [f, n] = tunit(d3.tns), T = d3.tns / f;
  t = ' — t = ' + (T * d3.frame / Math.max(d3.nk - 1, 1)).toFixed(T < 10 ? 2 : 1) + ' of ' + T.toFixed(1) + ' ' + n;
 }
 document.getElementById('d3ft').textContent = (d3.frame + 1) + ' of ' + d3.nk + t;
}
function scrub3d(v){
 if(!d3) return;
 d3.frame = Math.max(0, Math.min(d3.nk - 1, parseInt(v)));
 document.getElementById('d3frame').value = d3.frame;
 label3d();
 render3d();
}
window.addEventListener('keydown', e => {
 if(!d3 || d3.frame === null) return;
 if(e.target && e.target.tagName === 'INPUT' && e.target.type !== 'range') return;
 if(e.key === 'ArrowRight'){ scrub3d(d3.frame + 1); e.preventDefault(); }
 if(e.key === 'ArrowLeft'){ scrub3d(d3.frame - 1); e.preventDefault(); }
});
// FOUR FIXED CAMERAS. Elevation is measured from the membrane plane toward the spec's up axis (+89 = straight down
// from above; +-89 and not 90 because the orbit's up vector is degenerate along the axis). Top, bottom and side keep
// the current azimuth, zoom and frame; the oblique one is the movie's own camera, zoom included.
function view3d(k){
 if(!d3) return;
 const c = d3.cam || {elev: 40, azim: 30, zoom: 1.0, roll: d3.roll};
 if(k === 'nominal'){ d3.azim = c.azim; d3.elev = c.elev; d3.zoom = c.zoom; }
 else d3.elev = {top: 89, bottom: -89, side: 0}[k];
 d3.roll = c.roll;
 render3d();
}
// THE SLICE: a toggle, kept while the view turns and the frames change; the server cuts along the view each render
// THE GLASS: a toggle like the slice, carried on every render
function glass3d(){
 if(!d3) return;
 d3.glass = !d3.glass;
 document.getElementById('d3note').textContent = ' \u2014 rendering ...';
 document.getElementById('d3glass').className = 'ic' + (d3.glass ? ' on' : '');
 render3d();
}
// DOTS / SURFACE: a toggle like the glass, carried on every render; shown whenever the view built a contour (bio_view)
// THE ICON SHOWS THE ACTIVE LOOK, the size and style of the camera icons beside it: a cluster of dots, or a smooth
// cell outline for the isosurface
const LOOK_ICON = {
 dots: '<svg viewBox="0 0 20 20" style="fill:currentColor;stroke:none"><circle cx="6" cy="7" r="1.6"/><circle cx="10.5" cy="5" r="1.6"/>'
     + '<circle cx="14.5" cy="8" r="1.6"/><circle cx="8" cy="11" r="1.6"/><circle cx="12.5" cy="12" r="1.6"/>'
     + '<circle cx="5" cy="14.5" r="1.6"/><circle cx="9.5" cy="15.5" r="1.6"/><circle cx="15" cy="15" r="1.6"/></svg>',
 iso: '<svg viewBox="0 0 20 20"><path d="M10 2.5c4 0 7.5 2.6 7.5 6.6 0 4.6-3.4 8.4-7.8 8.4C5.6 17.5 2.5 14.6 2.5 10.8 2.5 6 5.8 2.5 10 2.5z"/>'
     + '<path d="M6.5 7.5c1-1.4 2.3-2 3.8-2" style="opacity:.6"/></svg>'
};
function look3dButton(){
 const b = document.getElementById('d3look');
 b.style.display = (d3 && d3.look) ? '' : 'none';
 if(!d3 || !d3.look) return;
 const iso = d3.look === 'iso';
 b.innerHTML = LOOK_ICON[iso ? 'iso' : 'dots'];
 b.className = 'ic' + (iso ? ' on' : '');
 b.title = d3.skin ? (iso ? 'dots / surface: showing the surface skins -- click for the points alone'
                          : 'dots / surface: showing every compartment as points, the skins hidden -- click for the surface')
                   : (iso ? 'dots / surface: showing the surface contoured per body -- click for the dots'
                          : 'dots / surface: showing the live points coloured per body -- click for the surface');
}
// THE GLASS BUTTON: in the place of the dots / surface button for a scene with no look, and beside it (only while the surface shows) for a scene of skins
function glassButton(){
 const b = document.getElementById('d3glass');
 b.style.display = (d3 && d3.look && (!d3.skin || d3.look === 'dots')) ? 'none' : '';
}
// THE SURFACE THRESHOLD: three buttons, x0.5 / x1 / x2 of the spec's own iso level, shown while the surface is on
function isoButtons(){
 const on = !!(d3 && d3.look === 'iso' && !d3.skin);
 [[0.5, 'd3iso1'], [1, 'd3iso2'], [2, 'd3iso3']].forEach(([f, id]) => {
  const b = document.getElementById(id);
  b.style.display = on ? '' : 'none';
  b.className = 'ic' + (on && Math.abs((d3.iso || 1) - f) < 1e-9 ? ' on' : '');
 });
}
// THE DOT SIZE: three buttons, x0.6 / x1 / x1.7 of the view's own dot, shown while the dots are on
function dotButtons(){
 const on = !!(d3 && d3.look === 'dots');
 [[0.6, 'd3dot1'], [1, 'd3dot2'], [1.7, 'd3dot3']].forEach(([f, id]) => {
  const b = document.getElementById(id);
  b.style.display = on ? '' : 'none';
  b.className = 'ic' + (on && Math.abs((d3.dot || 1) - f) < 1e-9 ? ' on' : '');
 });
}
// THE FLAME BUTTON: dots coloured by |F|, the size of the deformation gradient (red-white-blue, no outline), instead of by cell, shown while the dots are on, on a point cloud
function defButton(){
 const b = document.getElementById('d3def');
 const on = !!(d3 && d3.look === 'dots' && !d3.skin);
 b.style.display = on ? '' : 'none';
 b.className = 'ic' + (on && d3.def ? ' on' : '');
}
function def3d(){
 if(!d3 || d3.look !== 'dots') return;
 d3.def = !d3.def;
 defButton();
 document.getElementById('d3note').textContent = d3.def ? ' \u2014 measuring the deformation ...' : ' \u2014 rendering ...';
 render3d();
}
function dot3d(f){
 if(!d3 || d3.look !== 'dots') return;
 d3.dot = f;
 dotButtons();
 document.getElementById('d3note').textContent = ' \u2014 rendering ...';
 render3d();
}
function iso3d(f){
 if(!d3 || d3.look !== 'iso' || d3.skin) return;
 d3.iso = f;
 isoButtons();
 document.getElementById('d3note').textContent = ' — contouring the cells…';
 render3d();
}
function look3d(){
 if(!d3 || !d3.look) return;
 d3.look = (d3.look === 'iso') ? 'dots' : 'iso';
 look3dButton();
 glassButton();
 isoButtons();
 dotButtons(); defButton();
 if(d3.look === 'iso') document.getElementById('d3note').textContent = d3.skin ? ' \u2014 rendering ...' : ' — contouring the cells…';
 render3d();
}
// THE SLICE IS A SLAB of cells (a thick disc for a sphere) fixed in the world, of half thickness `slab` x the scene's radius;
// the two buttons beside the slice button thin and thicken it while it is on.
function slab3d(f){
 if(!d3 || !d3.cut) return;
 d3.slab = Math.max(0.03, Math.min(1.5, (d3.slab || 0.2) * f));
 document.getElementById('d3note').textContent = ' \u2014 rendering ...';
 render3d();
}
function cut3d(){
 if(!d3) return;
 d3.cut = !d3.cut;
 document.getElementById('d3thin').style.display = d3.cut ? '' : 'none';
 document.getElementById('d3thick').style.display = d3.cut ? '' : 'none';
 document.getElementById('d3note').textContent = ' \u2014 rendering ...';
 document.getElementById('d3cut').className = 'ic' + (d3.cut ? ' on' : '');
 render3d();
}
function close3d(){
 d3 = null;
 if(lfOn) setTimeout(showLatest, 50);
 document.getElementById('d3cams').style.display = 'none';
 document.getElementById('d3look').style.display = 'none';
 ['d3def', 'd3iso1', 'd3iso2', 'd3iso3', 'd3dot1', 'd3dot2', 'd3dot3'].forEach(id => { document.getElementById(id).style.display = 'none'; });
 document.getElementById('d3bar').style.display = 'none';
 document.getElementById('live3d').style.display = 'none';
 document.getElementById('d3note').textContent = '';
 tick();
}
// ONE RENDER IN FLIGHT AT A TIME, AND THE LAST GOOD FRAME STAYS UP.
//
// A render of a few hundred thousand particles takes seconds on the server's one VTK thread. A
// drag fires a mousemove every few milliseconds, so setting `img.src` on each one starts dozens
// of overlapping requests that each cancel the last image load -- and an <img> whose load was
// cancelled shows NOTHING. That is the black panel: not a failed render, a hundred successful
// ones none of which was ever allowed to finish.
//
// So: at most one request in flight, the newest camera queued behind it, and the picture decoded
// into an offscreen Image that is swapped in only once it is complete. The view then lags the
// mouse by one render instead of blanking.
let busy = false, pending = false;
function render3d(){
 if(!d3) return;
 if(busy){ pending = true; return; }
 busy = true;
 const url = '/api/watch/render?name=' + encodeURIComponent(d3.name || '')
   + '&azim=' + d3.azim.toFixed(1) + '&elev=' + d3.elev.toFixed(1)
   + '&zoom=' + d3.zoom.toFixed(3) + '&roll=' + d3.roll
   + (d3.live ? '&live=' + encodeURIComponent(d3.live) + '&frame=1' : (d3.frame !== null ? '&frame=' + d3.frame : '')) + '&cut=' + (d3.cut ? 1 : 0) + (d3.cut ? '&slab=' + (d3.slab || 0.2) : '') + '&glass=' + (d3.glass ? 1 : 0)
   + (d3.look ? '&look=' + d3.look : '') + (d3.look === 'iso' && !d3.skin && d3.iso ? '&iso=' + d3.iso : '') + (d3.look === 'dots' && d3.dot ? '&dot=' + d3.dot : '') + (d3.look === 'dots' && !d3.skin ? '&deform=' + (d3.def ? 1 : 0) + (d3.def ? '&ink=0' : '') : '') + '&t=' + Date.now();
 const pre = new Image();
 pre.onload = () => {
  if(d3){
   document.getElementById('live3d').src = pre.src;
   if(!d3.shown){                              // the first frame: only now does the 3-D panel replace the movie
    d3.shown = true; loading(''); drvEv('4d-first-frame');
    document.getElementById('shot').style.display = 'none';
    document.getElementById('mov').style.display = 'none';
    document.getElementById('live3d').style.display = '';
   }
   const nt = document.getElementById('d3note');
   if(nt.textContent === ' — contouring the cells…' || nt.textContent === ' \u2014 rendering ...' || nt.textContent === ' \u2014 measuring the deformation ...') nt.textContent = '';
  }
  busy = false;
  if(pending){ pending = false; render3d(); }
 };
 pre.onerror = () => {
  busy = false; pending = false; loading('');
  document.getElementById('d3note').textContent = ' — the render failed; see the server log';
 };
 pre.src = url;
}
(function(){
 const im = document.getElementById('live3d');
 im.addEventListener('dragstart', e => e.preventDefault());   // or the browser drags the IMAGE
 im.addEventListener('mousedown', e => { if(d3){ d3.drag = {x: e.clientX, y: e.clientY}; im.style.cursor='grabbing'; e.preventDefault(); }});
 window.addEventListener('mouseup', () => { if(d3){ d3.drag = null; document.getElementById('live3d').style.cursor='grab'; }});
 window.addEventListener('mousemove', e => {
  if(!d3 || !d3.drag) return;
  // THE PICTURE MOVES WITH THE HAND: a rightward drag turns the camera left, a downward drag lifts it. A picture
  // rolled 180 deg is turned over on screen, so both signs flip with it.
  const s = Math.cos(d3.roll * Math.PI / 180) >= 0 ? 1 : -1;
  d3.azim -= s * (e.clientX - d3.drag.x) * 0.5;
  d3.elev = Math.max(-89, Math.min(89, d3.elev + s * (e.clientY - d3.drag.y) * 0.5));
  d3.drag = {x: e.clientX, y: e.clientY};
  render3d();
 });
 im.addEventListener('wheel', e => {
  if(!d3) return;
  e.preventDefault();
  d3.zoom = Math.max(0.2, Math.min(12, d3.zoom * (e.deltaY < 0 ? 1.12 : 1/1.12)));
  render3d();
 }, {passive: false});
})();
// THE PAGE CAN BE DRIVEN THROUGH THE API (Cedric, 2026-11-07: "I would like to see you interacting with it through an API"): a script queues a command at
// `/api/watch/drive?do=step&arg=-1` and THIS page, polling that route every second, runs it on its own buttons -- so the person watching sees the page move -- and
// reports what it showed and when (`drive?ev=...`, read back with `drive?log=1`): the milliseconds from the command to the loading line, to the picture, to each pane.
let drvN = null, drv = {cmd: '', t0: 0}, recId = '';
function drvEv(ev, d){
 if(!drv.t0) return;
 fetch('/api/watch/drive?ev=' + ev + '&cmd=' + encodeURIComponent(drv.cmd) + '&ms=' + (Date.now() - drv.t0) + (d ? '&d=' + encodeURIComponent(d) : ''));
}
function drvSnap(){
 const t = id => (document.getElementById(id).textContent || '').trim().slice(0, 90);
 return JSON.stringify({idx: idx, total: total, shown: shown, loading: t('loading'), mpos: t('mpos'), meta: t('meta'), px: t('plexus'), mx: t('math'), d3: !!d3,
                        video: document.getElementById('mov').style.display !== 'none', def: !!(d3 && d3.def),
                        toolbar: ['d3cut', 'd3def', 'd3look', 'd3dot1', 'd3dot2', 'd3dot3', 'd3glass'].filter(k => document.getElementById(k).style.display !== 'none').join(' '), dim: document.getElementById('mov').classList.contains('dim') || document.getElementById('shot').classList.contains('dim')});
}
function drvRun(c){
 drv = {cmd: c.do + (c.arg ? ' ' + c.arg : ''), t0: Date.now()};
 document.getElementById('drvnote').textContent = ' \u25b6 API: ' + drv.cmd;
 if(c.do === 'step') step(parseInt(c.arg || '1', 10));
 else if(c.do === 'go'){ lfOff(); idx = parseInt(c.arg, 10); step(0); }
 else if(c.do === 'pane') pane(c.arg);
 else if(c.do === 'open3d') open3d();
 else if(c.do === 'def') def3d();
 else if(c.do === 'live') live();
 drvEv('after-click', drvSnap());
 if(c.do === 'snap') drvEv('snap', drvSnap());
}
async function drvPoll(){
 try{
  const j = await (await fetch('/api/watch/drive?rec=' + encodeURIComponent(recId) + (drvN === null ? '' : '&after=' + drvN))).json();
  if(drvN === null){ drvN = j.n; return; }          // a page opened now does not replay the commands queued before it
  for(const c of j.cmds){ drvN = c.n; if(!c.rec || !recId || c.rec === recId) drvRun(c); }       // a command with `rec=` is for the page showing that record only
 }catch(e){ /* the next second asks again */ }
}
setInterval(drvPoll, 1000);
function pane(w){
 shown = w;
 for(const k of ['plexus','math','hierarchy','spec','insights']){
  document.getElementById(k).style.display = (k === w) ? '' : 'none';
  document.getElementById('b_' + k).className = (k === w) ? 'on' : '';
 }
 if(w === 'plexus') loadPlexus();
 if(w === 'math') loadMath();
 if(w === 'insights') loadInsights();
 if(w === 'hierarchy') loadHierarchy();
 tick();
}
// THE `hierarchy` PANE (Cedric, 2026-11-01, in the place of `causal`): the containment of the step's spec -- which sets live inside which, how
// many per parent -- as a graph and, beside it, the table of counts with the total of MPM points. Drawn by tools/spec_hierarchy.py in the look
// the spec's `plotting.hierarchy` names (circlepack by default); the picture comes back inline and is asked again only for another step.
let hiAt = null, hiWant = null, hiShown = null;
// A PANE NEVER SHOWS ANOTHER STEP'S TEXT (Cedric, 2026-11-06): going from one step to the next left the previous step's hierarchy, maths or plexus
// text on screen until the new one arrived, so it read as the new step's. `blankFor` empties the pane, with a line saying what it is waiting
// for, the moment it is asked for a step other than the one its text is for (`shown`); a re-ask for the same step (a story still being
// written) leaves the text alone.
// A SMALL * ON THE plexus AND math TABS (Cedric, 2026-11-07) when the pane on show is SAVED beside its step (`story/NNNN.plexus.html`, `.math.html`): it was read from there, or has just been written there.
function star(k, on){
 document.getElementById('b_' + k).innerHTML = k + (on ? '<span style="font-size:70%;vertical-align:super">*</span>' : '');
}
function blankFor(el, shown, i, msg){
 if(shown === i) return;
 // A PLACEHOLDER ALREADY UP FOR THIS STEP IS KEPT, so its clock keeps counting through the retries
 if(el.dataset.for === String(i) && el.querySelector('[data-w]')) return;
 el.innerHTML = '<div class="ps-g" data-w="' + Date.now() + '">' + msg + '<span class=el></span></div>'; el.dataset.t = ''; el.dataset.for = String(i);
}
// WHAT IS LOADING, IN ONE LINE ABOVE THE PICTURE, with the seconds it has taken, and the picture dimmed while it waits: the page used to keep the old
// step on screen until the new one arrived, so a click looked like a click that did nothing (Cedric, 2026-11-07: "I do not like to click and wait
// without seeing any changes"). `loading('')` ends it.
let loadT0 = 0, loadMsg = '';
const LOAD_BOXES = ['shot', 'mov', 'live3d'];
function paintLoading(){
 if(loadT0) document.getElementById('loading').textContent = '\u25cf ' + loadMsg + ' \u2026 ' + ((Date.now() - loadT0) / 1000).toFixed(0) + ' s';
}
function loading(msg){
 const el = document.getElementById('loading');
 if(!msg){ if(loadT0) drvEv('picture-loaded'); el.textContent = ''; loadT0 = 0; LOAD_BOXES.forEach(k => document.getElementById(k).classList.remove('dim')); return; }
 if(!loadT0) loadT0 = Date.now();
 loadMsg = msg; paintLoading();
 LOAD_BOXES.forEach(k => document.getElementById(k).classList.add('dim'));
}
setInterval(() => {
 paintLoading();
 const now = Date.now();
 document.querySelectorAll('[data-w]').forEach(e => { const c = e.querySelector('.el'); if(c) c.textContent = ' ' + Math.round((now - e.dataset.w) / 1000) + ' s'; });
}, 500);
let hiCtl = null;
async function loadHierarchy(){
 const i = (idx < 0) ? total - 1 : idx;
 if(total === 0 || hiAt === i || hiWant === i) return;
 hiWant = i;
 const el = document.getElementById('hierarchy');
 if(hiShown !== i){ blankFor(el, null, i, 'hierarchy rendering \u2026'); hiShown = null; hiAt = null; }
 if(hiCtl) hiCtl.abort();
 const ctl = hiCtl = new AbortController();
 try{
  const j = await (await fetch('/api/watch/hierarchy?i=' + i + R(), {signal: ctl.signal})).json();
  if(hiWant !== i) return;
  if(j.busy){ setTimeout(loadHierarchy, 400); return; }
  const tx = j.text || '<div class="ps-g">no hierarchy for this step</div>';
  if(el.dataset.t !== tx){ el.innerHTML = tx; el.dataset.t = tx; }
  hiAt = i; hiShown = i;
 }catch(e){ /* the interval retries; the line above says what it waits for */ }
 finally{ if(hiWant === i) hiWant = null; }
}
// THE `math` PANE (Cedric, 2026-10-12): every equation of the run at once, in one notation, each located by one
// word, then one legend of the variables and the constants -- fetched like the plexus pane, asked again while
// the agent is still writing it.
let mxAt = null, mxWant = null, mxShown = null, mxCtl = null;
async function loadMath(){
 const i = (idx < 0) ? total - 1 : idx;
 if(total === 0 || mxAt === i || mxWant === i || Date.now() < mxRetry) return;
 mxWant = i;
 const el = document.getElementById('math');
 // THE STEP LEFT BEHIND IS FORGOTTEN WHEN ITS PANE IS BLANKED: going back to it must fetch it again, not find it "already shown" over a placeholder
 if(mxShown !== i){ blankFor(el, null, i, 'reading the equations\u2026'); mxShown = null; mxAt = null; star('math', false); }
 if(mxCtl) mxCtl.abort();                  // the request of the step left behind gives its connection back at once
 const ctl = mxCtl = new AbortController();
 try{
  const j = await (await fetch('/api/watch/math?i=' + i + R(), {signal: ctl.signal})).json();
  if(mxWant !== i) return;
  if(j.busy){ setTimeout(loadMath, 400); return; }       // the server is still drawing it in the background: ask again, the placeholder keeps counting
  const tx = j.text || '<div class="ps-g">no spec recorded for this step</div>';
  if(el.dataset.t !== tx){ el.innerHTML = tx; el.dataset.t = tx; }
  mxShown = i; star('math', j.saved); drvEv('math-pane', 'saved=' + j.saved + ' pending=' + j.pending);
  if(!j.pending) mxAt = i; else { mxRetry = Date.now() + 10000; setTimeout(loadMath, 10100); }
 }catch(e){ /* the interval retries; keep whatever is on screen */ }
 finally{ if(mxWant === i) mxWant = null; }
}
// THE `insights` PANE reads a small file saved beside the step (`story/NNNN.insights.json`, written from the deck's notes), so it is asked again
// every 4 s while it is on screen: an export made after the page opened shows up without a reload. The text is rewritten only when it changed.
let inAt = null, inT = 0, inWant = null, inCtl = null;
async function loadInsights(){
 const i = (idx < 0) ? total - 1 : idx;
 if(total === 0 || inWant === i || (inAt === i && Date.now() - inT < 4000)) return;
 inWant = i;
 const el = document.getElementById('insights');
 if(inAt !== i){ blankFor(el, null, i, 'reading the insights…'); inAt = null; }
 if(inCtl) inCtl.abort();
 const ctl = inCtl = new AbortController();
 try{
  const j = await (await fetch('/api/watch/insights?i=' + i + R(), {signal: ctl.signal})).json();
  if(inWant !== i) return;
  const tx = j.text || '<div class="ps-g">no insights written for this step</div>';
  if(el.dataset.t !== tx){ el.innerHTML = tx; el.dataset.t = tx; }
  inAt = i; inT = Date.now();
 }catch(e){ /* the next tick asks again */ }
 finally{ if(inWant === i) inWant = null; }
}
// THE `plexus` PANE IS FETCHED, NOT POLLED. The other three panes are files read off disk and ride
// along on `state` every 1.5 s; this one parses the spec and walks the operator registry, so it
// gets its own route and is asked for only when the step it belongs to changes. `pxAt` remembers
// which step the text on screen is FOR, which is what stops a re-render on every tick and, more
// importantly, stops step N's summary being left on screen while the picture has moved to N+1.
let pxAt = null, pxWant = null, pxShown = null, pxRetry = 0, mxRetry = 0, pxCtl = null;
async function loadPlexus(){
 const i = (idx < 0) ? total - 1 : idx;
 if(total === 0 || pxAt === i || pxWant === i || Date.now() < pxRetry) return;
 pxWant = i;
 const el = document.getElementById('plexus');
 if(pxShown !== i){ blankFor(el, null, i, 'reading the spec\u2026'); pxShown = null; pxAt = null; star('plexus', false); }
 if(pxCtl) pxCtl.abort();
 const ctl = pxCtl = new AbortController();
 try{
  const j = await (await fetch('/api/watch/plexus?i=' + i + R(), {signal: ctl.signal})).json();
  // A REPLY FOR A STEP THAT IS NO LONGER SHOWN IS DROPPED. Walking the record fast fires several
  // of these and they can land out of order; without this the pane settles on whichever was
  // slowest rather than on the step being looked at.
  if(pxWant !== i) return;
  if(j.busy){ setTimeout(loadPlexus, 400); return; }     // the server is still drawing it in the background: ask again, the placeholder keeps counting
  // THE PANE IS REWRITTEN ONLY WHEN ITS TEXT CHANGED: re-setting the same html every tick made it blink (2026-10-18)
  const tx = j.text || '<div class="ps-g">no spec recorded for this step</div>';
  if(el.dataset.t !== tx){ el.innerHTML = tx; el.dataset.t = tx; }
  pxShown = i; star('plexus', j.saved); drvEv('plexus-pane', 'saved=' + j.saved + ' pending=' + j.pending);
  // A STORY STILL BEING WRITTEN IS ASKED FOR AGAIN, but every 10 s and not every tick: the agent takes minutes, and the pane
  // already shows the story the facts give.
  if(!j.pending) pxAt = i; else { pxRetry = Date.now() + 10000; setTimeout(loadPlexus, 10100); }
 }catch(e){ /* the interval retries; keep whatever is on screen */ }
 finally{ if(pxWant === i) pxWant = null; }
}
function live(){ lfOff(); idx = -1; tick(); }
// THE LATEST FRAME OF A RUNNING JOB. A run writes its live view (a still every few hundred frames) into its data folder while
// it simulates, long before its movie and trajectory exist, so the record has nothing to show for it. This button asks the
// server for the newest such still of any running job and shows it in the picture panel, refreshed every 15 s; the
// selector appears when more than one job is running. Any step, movie or live button turns it off.
let lfOn = false, lfRun = '', lfMeta = '', lfCut = false, lfLook = 'ink', lfState = false, stepRun = '';
// LATEST FRAME IN THE 3-D VIEW TOO (the human, 2026-10-18): with latest frame on, the 3D button opens the running job's saved state, the
// view can be turned and sliced like any scene, and every 15 s the newer state, when there is one, is drawn and the stamp updated.
async function lfStamp3d(run){
 let j; try{ j = await (await fetch('/api/watch/live?x=' + Date.now())).json(); }catch(e){ return; }
 const r = (j.runs || []).find(x => x.run === run); if(!r) return;
 lfStamp('frame ' + r.frame + '/' + (r.total || '?'));
 return r;
}
setInterval(async () => {
 if(!(lfOn && d3 && d3.live)) return;
 const r = await lfStamp3d(d3.live);
 if(r && r.mtime > d3.lt){ d3.lt = r.mtime; document.getElementById('d3note').textContent = ' \u2014 rendering ...'; render3d(); }
}, 15000);
function lfOff(){ lfOn = false; document.getElementById('lfsel').style.display = 'none';

 document.getElementById('lfstamp').style.display = 'none'; }
// THE STAMP over the picture's top-left corner says what the picture is: the job, its frame, the simulated time and how old the
// saved state is -- so a latest frame cannot be mistaken for a recorded step.
function lfStamp(t){ const e = document.getElementById('lfstamp'); e.textContent = t; e.style.display = t ? 'block' : 'none'; }
async function showLatest(){
 if(!lfOn || d3) return;
 let j;
 try{ j = await (await fetch('/api/watch/live?x=' + Date.now())).json(); }catch(e){ return; }
 const sel = document.getElementById('lfsel'), runs = j.runs || [];
 document.getElementById('mov').pause(); document.getElementById('mov').style.display = 'none'; document.getElementById('shot').style.display = '';
 sel.innerHTML = runs.map(r => '<option value="' + r.run + '">' + r.run + ' &mdash; frame ' + r.frame
                  + (r.total ? ' of ' + r.total : '') + '</option>').join('');
 // THE STEP'S OWN RUN FIRST (step 0092 is sph1, step 0093 is sph2): latest frame shows the run the step on screen is about, and the
 // selector appears only when that run is not computing and several others are
 const mine = runs.find(r => r.run === stepRun);
 const pick = mine || runs.find(r => r.run === lfRun) || runs[0];
 const im = document.getElementById('shot'), mv = document.getElementById('mov');
 if(!pick){
  lfMeta = 'no running job has written a frame yet';
  lfStamp('no frame yet');
  document.getElementById('meta').innerHTML = lfMeta; sel.style.display = 'none'; return;
 }
 lfRun = pick.run; lfState = !!pick.state; sel.value = pick.run; sel.style.display = (!mine && runs.length > 1) ? '' : 'none';
 const what = 'latest frame of ' + pick.run + ' &nbsp;&middot;&nbsp; frame ' + pick.frame + (pick.total ? ' of ' + pick.total : '')
        + ' &nbsp;&middot;&nbsp; ' + Math.round(pick.age_s / 60) + ' min ago';
 mv.pause(); mv.style.display = 'none'; im.style.display = '';
 lfStamp('frame ' + pick.frame + '/' + (pick.total || '?'));
 if(pick.state){
  // THE JOB DUMPED ITS STATE: render it here, in the ink look, cut in half and centred -- not the job's own off-centre live view
  const stamp0 = document.getElementById('lfstamp').textContent;
  lfStamp(stamp0 + ' \u00b7 rendering ...');
  let r; try{ r = await (await fetch('/api/watch/liverender?run=' + encodeURIComponent(pick.run) + '&cut=' + (lfCut ? 1 : 0) + '&look=' + lfLook + '&x=' + Date.now())).json(); }catch(e){ return; }
  if(r.state === 'ready'){ lfMeta = what + ' &nbsp;&middot;&nbsp; rendered from the job&apos;s saved state' + (lfCut ? ', slice' : ', whole') + ', ' + lfLook; document.getElementById('meta').innerHTML = lfMeta;
   lfStamp(stamp0); im.src = r.url; return; }
  lfMeta = what + ' &nbsp;&middot;&nbsp; rendering ... (30-60 s)'; document.getElementById('meta').innerHTML = lfMeta;
  if(r.state === 'rendering') setTimeout(() => { if(lfOn) showLatest(); }, 6000);
  if(r.state !== 'failed') return;
 }
 lfMeta = what + ' &nbsp;&middot;&nbsp; the running job&apos;s own live view';
 document.getElementById('meta').innerHTML = lfMeta;
 im.src = '/api/watch/liveimg?run=' + encodeURIComponent(pick.run) + '&t=' + pick.mtime;
}
setInterval(() => { if(lfOn) showLatest(); }, 15000);
// `d` IS ANY STRIDE, which is why the ten-step buttons needed no new function: << is step(-10)
// and >> is step(10), through the same wrap. The record runs to a hundred and fifty steps and
// walking it one at a time to reach a rung from a few hours ago is the only thing it was awkward
// at.
let movies = [];
let curShot = '';                 // the picture file of the step on show, e.g. 0012.png: printed with the step's number
function step(d){
 lfOff();
 if(total === 0) return;
 if(idx < 0) idx = total - 1;           // stepping back from live starts at the newest
 // WRAPS, rather than stopping dead at either end. The record is a loop to walk, not a list with
 // walls: next past the last step returns to the first and prev before the first goes to the
 // last, so flipping through it never needs a decision about which button has stopped working.
 idx = (idx + d + total) % total;
 // ANSWERED AT ONCE, before the server says anything: the picture dims, the line above it names the step being loaded, and the panes that
 // are open blank to a placeholder with a clock -- and ask for the new step in parallel with the state, not after it.
 loading('step ' + (idx + 1) + ' of ' + total);
 if(shown === 'plexus') loadPlexus();
 if(shown === 'math') loadMath();
 if(shown === 'insights') loadInsights();
 if(shown === 'hierarchy') loadHierarchy();
 tick();
}
// THE PAGE SURVIVES THE SERVER RESTARTING UNDER IT, which it does often: Python caches its
// imports, so every new operator or renderer change needs the process replaced, and until now that
// left this tab dead -- a blank page, and a fresh link had to be asked for each time. Now a failed
// poll is a STATE rather than an exception: the banner says so, the picture already on screen
// stays put, and the next successful poll clears it. The URL never changes, so the tab can simply
// be left open.
let down = false;
let tickCtl = null;
async function tick(){
 let j;
 const ctl = new AbortController();
 if(tickCtl) tickCtl.abort();             // a newer click supersedes the state still being waited for
 tickCtl = ctl;
 try{
  const want = rec;
  j = await (await fetch('/api/watch/state?i=' + idx + R(), {signal: ctl.signal})).json();
  if(tickCtl === ctl) tickCtl = null;
  if(want !== rec) return;                 // a reply for the record this tab has just left
 }catch(e){
  if(e && e.name === 'AbortError') return;
  if(tickCtl === ctl) tickCtl = null;
  loading('');
  if(!down){ down = true;
   document.getElementById('meta').innerHTML =
     '<span style="color:#e08214">server restarting &mdash; this page will reconnect by itself, '
     + 'nothing to reopen</span>'; }
  return;                                  // keep whatever is on screen; the interval retries
 }
 if(down){ down = false; }
 total = j.n_shots; recId = j.rec || '';
 stepRun = ((j.why || '').match(/spec\\s+(\\S+)/) || [])[1] || '';      // the run this step is about, for the latest-frame mode
 movies = j.movies || [];
 const mAt = movies.indexOf(idx < 0 ? total - 1 : idx);
 // A STEP WITHOUT A MOVIE PRINTS ITS PICTURE'S NUMBER (`pic 0012`, the file name in the record), so the step on show can be named
 // when there is no `movie N of M` to read it from (exp02: three-frame movies are not played).
 const picId = (j.shot_name || '').replace(/\\.png$/, '');
 document.getElementById('mpos').textContent =
   mAt >= 0 ? `movie ${mAt + 1} of ${movies.length} (step ${(idx < 0 ? total - 1 : idx) + 1}, pic ${picId})`
            : (picId ? `pic ${picId}` + (movies.length ? `  \u00b7  ${movies.length} movies` : '')
                     : (movies.length ? `${movies.length} movies` : 'no movies yet'));
 document.getElementById('livebtn').className = (idx < 0) ? 'on' : '';
 curShot = j.shot_name || '';
 document.getElementById('pos').textContent =
   (total ? (idx < 0 ? `live — ${total} of ${total}` : `${j.index + 1} of ${total}`) : '')
   + (j.shot_name ? `   step ${j.shot_name.replace(/\\.png$/, '')}` : '')
   // the why's first line after "step NNNN =": which run this is (e.g. "batch 3 · arm 4 · <run>"), so a step
   // can be matched to the deck without opening the why pane
   + ((j.why || '').split(String.fromCharCode(10))[0].replace(/^step\\s+\\d+\\s+=\\s+/, '').trim()
      ? '   ' + (j.why || '').split(String.fromCharCode(10))[0].replace(/^step\\s+\\d+\\s+=\\s+/, '').trim() : '');
 document.getElementById('meta').innerHTML = lfOn ? lfMeta :
   j.shot ? `${j.shot_name} &nbsp;&middot;&nbsp; ${j.shot_age}s ago`
          + (j.mp4_play ? ' &nbsp;&middot;&nbsp; showing the movie' : '') : 'no picture yet';
 const im = document.getElementById('shot');
 // THE OTHER THREE FILES OF THE STEP. The why is always in view because it is the one thing a
 // picture cannot show; the spec sits behind a tab because it is 250 lines.
 { const sp = document.getElementById('spec'), t = j.spec || '(no spec recorded for this step)';      // written only when it changed, not on every tick
   if(sp.textContent !== t) sp.textContent = t; }
 // ... and the fourth pane, which is fetched rather than carried on `state`. Asked for on every
 // tick so that FOLLOWING THE RECORD LIVE keeps it current: `loadPlexus` returns immediately
 // unless the step on screen has actually changed.
 if(shown === 'plexus') loadPlexus();
 if(shown === 'math') loadMath();
 if(shown === 'insights') loadInsights();
 if(shown === 'hierarchy') loadHierarchy();
 // THE MOVIE REPLACES THE PICTURE WHEN THERE IS ONE, rather than sitting under it. A step that
 // ran has a still of its LAST frame and a film of the whole run, and showing both makes the
 // reader compare a frame against the thing it was taken from; the film is strictly the better
 // record. A step that only built something has no film, and then the still is all there is.
 const mv = document.getElementById('mov');
 if(lfOn){ loading(''); return; }       // the latest frame of a running job owns the panel while it is on
 if(d3){ if(d3.shown) loading(''); return; }       // the 3-D view owns the panel while it is open (its own line stays until its first frame)
 if(j.mp4_play){
  // KEYED BY MTIME AS WELL AS POSITION, exactly as the still below is. The movie is served
  // with a day of cache so that stepping back and forth does not re-download 75 MB each
  // way, and without a buster in the URL the browser can answer a later request for the
  // same position from that cache -- which shows the wrong run under the right caption.
  const _k = rec + ':' + j.index + '.' + (j.mp4_mtime || j.shot_mtime || 0);
  if(mv.dataset.k !== _k){ mv.src = '/api/watch/mp4?i=' + j.index + '&t=' + (j.mp4_mtime || j.shot_mtime || 0) + R();
                           mv.dataset.k = _k; mv.load(); loading('movie ' + (j.index + 1) + ' of ' + total); }   // cleared by the video's own loadeddata
  else loading('');
  mv.style.display = ''; im.style.display = 'none';
 } else {
  mv.removeAttribute('src'); mv.removeAttribute('data-i'); mv.style.display = 'none';
  im.style.display = '';
  if(j.shot){
   const u = '/api/watch/shot?i=' + j.index + '&t=' + j.shot_mtime + R();
   if(im.getAttribute('src') !== u){ im.src = u; loading('picture ' + (j.index + 1) + ' of ' + total); }   // cleared by the image's own load
   else loading('');
  } else { im.removeAttribute('src'); loading(''); }          // an experiment with no step yet: nothing, not the last record's picture
 }
}
['loadeddata', 'error'].forEach(ev => document.getElementById('mov').addEventListener(ev, () => loading('')));
['load', 'error'].forEach(ev => document.getElementById('shot').addEventListener(ev, () => loading('')));
document.addEventListener('keydown', e => {
 if(e.key === 'ArrowLeft') step(-1);
 if(e.key === 'ArrowRight') step(1);
});
tick();
// only follow along while LIVE; a pinned picture must stay put
// `|| down` IS WHAT MAKES A PINNED VIEW RECONNECT. Following along is only wanted while LIVE, so
// a pinned picture must stay put -- but that also meant a pinned tab never polled again, and so
// never noticed the server coming back. It polls while it is down whatever it is showing.
setInterval(() => { if((idx < 0 || down) && !tickCtl) tick(); }, 2000);
</script></body></html>"""




# ---------------------------------------------------------------- the live frame of a running job

LIVE_MAX_AGE_S = 12 * 3600       # a still older than this belongs to a job that died, not to one still computing
_LIVE_RUN = re.compile(r"^[A-Za-z0-9_]+/[A-Za-z0-9_.\-]+$")


def live_frames() -> list:
    """The newest live-view still of every run that is computing now: a run writes `still_NN_fFRAME.png` into its data
    folder every few hundred frames while it simulates and removes them when it finishes, so the stills present ARE
    the running jobs. Newest first: run (`group/name`), frame, total frames (from the spec in config/<group>/), age."""
    from plexus.paths import graphs_data_path
    out = []
    now = time.time()
    for d in glob.glob(os.path.join(graphs_data_path(), "*", "*")):
        best = None
        for f in glob.glob(os.path.join(d, "still_*_f*.png")):
            m = re.search(r"_f(\d+)\.png$", f)
            if m and (best is None or int(m.group(1)) > best[0]):
                best = (int(m.group(1)), f)
        st = os.path.join(d, "live_state.npz")
        has_state = os.path.exists(st) and not os.path.exists(os.path.join(d, "trajectory.npz")) \
            and now - os.path.getmtime(st) <= LIVE_MAX_AGE_S
        if has_state:                                    # the dump wins a tie on FRAME: the job writes the still and the dump in the same frame, in either order
            try:
                import numpy as _np
                _sf = int(_np.load(st)["frame"])
                if best is None or _sf >= best[0]:
                    best = (_sf, st)
            except Exception:                                                              # noqa: BLE001
                has_state = False
        if best is None or now - os.path.getmtime(best[1]) > LIVE_MAX_AGE_S:
            continue
        group, name = os.path.basename(os.path.dirname(d)), os.path.basename(d)
        total, dt_, ts_ = 0, 0.1, 60.0
        for cand in (os.path.join(d, "spec.yaml"), os.path.join(REPO, "config", group, name + ".yaml")):
            if os.path.exists(cand):
                txt = _read(cand)
                mt = re.search(r"^\s*n_frames:\s*(\d+)", txt, re.M)
                if mt:
                    total = int(mt.group(1))
                    md = re.search(r"^\s*dt:\s*([0-9.eE+-]+)", txt, re.M)
                    mu = re.search(r"time_s:\s*([0-9.eE+-]+)", txt)
                    dt_ = float(md.group(1)) if md else dt_
                    ts_ = float(mu.group(1)) if mu else ts_
                    break
        out.append({"run": f"{group}/{name}", "frame": best[0], "total": total, "sim_min": round(best[0] * dt_ * ts_ / 60.0, 1),
                    "state": bool(has_state and best[1] == st),
                    "mtime": int(os.path.getmtime(best[1])), "age_s": int(now - os.path.getmtime(best[1]))})
    return sorted(out, key=lambda r: r["age_s"])


def live_file(q):
    """The newest still of the run in `?run=group/name` -- the run name matched against a whitelist pattern, then looked up
    among the stills `live_frames` found, never joined into a path as given."""
    run = (q.get("run", [""])[0] or "")
    if not _LIVE_RUN.match(run):
        return None
    from plexus.paths import graphs_data_path
    d = os.path.realpath(graphs_data_path(*run.split("/")))
    if not d.startswith(os.path.realpath(graphs_data_path()) + os.sep):
        return None
    fs = sorted(glob.glob(os.path.join(d, "still_*_f*.png")),
                key=lambda f: int((re.search(r"_f(\d+)\.png$", f) or [0, 0])[1]) if re.search(r"_f(\d+)\.png$", f) else -1)
    return fs[-1] if fs else None


_LIVE_JOBS: dict = {}            # (run, state mtime, cut) -> "running" | "done" | "failed"
LIVE_RENDER_DIR = "/tmp/plexus_live"
_LW: dict = {}                   # the render worker: {"p": process, "q_in": queue, "q_out": queue, "lock": lock}


_LIVE_LOOKS = {"ink"}            # the looks the monitor keeps rendered ahead of the click; "soft" joins once someone asks for it


def _live_png(run: str, mtime: int, cut: bool = True, look: str = "ink") -> str:
    return os.path.join(LIVE_RENDER_DIR, f"{run.replace('/', '__')}_{mtime}_{'cut' if cut else 'whole'}{'_soft' if look == 'soft' else ''}.png")


def _live_worker_main(q_in, q_out, repo):
    """The render worker (a spawned process): imports the studio once, then renders what it is sent, one still at a time."""
    import importlib.util
    import sys
    sys.path.insert(0, os.path.join(repo, "src"))
    sp = importlib.util.spec_from_file_location("exp23_studio", os.path.join(repo, "tools", "exp23_studio.py"))
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    import pyvista, plexus.gui.flat_ink                                  # noqa: F401,E401  (the heavy imports, paid once)
    sp2 = importlib.util.spec_from_file_location("exp27_soft_render", os.path.join(repo, "tools", "exp27_soft_render.py"))
    soft = importlib.util.module_from_spec(sp2)
    sp2.loader.exec_module(soft)
    while True:
        job = q_in.get()
        if job is None:
            return
        key, run, npz, png, cut, look = job
        try:
            if look == "soft":                                           # the job's own renderer (LiveMovie), not the studio's
                soft.render_soft(run, npz, png, cut, m)
            else:
                m.render_live(run, npz, png, cut)
            q_out.put((key, os.path.exists(png)))
        except Exception as e:                                           # noqa: BLE001
            print(f"[live-render] {run}: {type(e).__name__}: {e}", flush=True)
            q_out.put((key, False))


def _live_worker():
    """Start the render worker and the thread that marks its results, once."""
    import multiprocessing as mp
    import threading
    if _LW.get("p") is not None and _LW["p"].is_alive():
        return _LW
    ctx = mp.get_context("spawn")                                        # not a fork: this process holds VTK and CUDA state
    q_in, q_out = ctx.Queue(), ctx.Queue()
    p = ctx.Process(target=_live_worker_main, args=(q_in, q_out, REPO), daemon=True)
    p.start()
    _LW.update(p=p, q_in=q_in, q_out=q_out)

    def collect():
        while True:
            key, ok = q_out.get()
            _LIVE_JOBS[key] = "done" if ok else "failed"
    threading.Thread(target=collect, daemon=True).start()
    return _LW


def _live_submit(run: str, mt: int, cut: bool, npz: str, look: str = "ink") -> None:
    key = (run, mt, cut, look)
    if os.path.exists(_live_png(run, mt, cut, look)) or _LIVE_JOBS.get(key) in ("running", "failed"):
        return
    os.makedirs(LIVE_RENDER_DIR, exist_ok=True)
    _LIVE_JOBS[key] = "running"
    _live_worker()["q_in"].put((key, run, npz, _live_png(run, mt, cut, look), cut, look))


_LIVE_MON: dict = {}


def _prewarm() -> None:
    """What the first click would otherwise wait for, done when the first page opens: the imports of the story module (the operator registry and the equation
    tools) and of matplotlib's pyplot, which take minutes on a slow network disk."""
    try:
        _story_module()
        import matplotlib.pyplot                                 # noqa: F401  the 4D view's first frame imports it inside the VTK thread: 5 minutes of the first click on a slow network disk
    except Exception as e:                                       # noqa: BLE001
        print(f"[watch] prewarm: {type(e).__name__}: {e}", flush=True)


def _live_monitor():
    """PRE-RENDER: every 10 s, every running job's newest saved state is rendered in both its looks (the slab and the whole), so a
    click on the latest frame finds the picture ready instead of waiting for a render."""
    import threading
    if _LIVE_MON.get("on"):
        return
    _LIVE_MON["on"] = True
    threading.Thread(target=_prewarm, daemon=True).start()

    def loop():
        while True:
            try:
                from plexus.paths import graphs_data_path
                for r in live_frames():
                    if r.get("state"):
                        st = os.path.join(graphs_data_path(*r["run"].split("/")), "live_state.npz")
                        mt = int(os.path.getmtime(st))
                        for look in sorted(_LIVE_LOOKS):
                            for cut in (False, True):
                                _live_submit(r["run"], mt, cut, st, look)
            except Exception:                                                  # noqa: BLE001
                pass
            time.sleep(10)
    threading.Thread(target=loop, daemon=True).start()


def g_watch_liverender(h, q):
    """A running job's dumped state (`live_state.npz`, written by `state_dump`) drawn by tools/exp23_studio.py in the ink look: the
    reply is `ready` with the picture's URL, or `rendering` and the page asks again. The pictures are made by a long-lived worker
    process (imports paid once) and, from the first page load on, ahead of the click (`_live_monitor`)."""
    run = (q.get("run", [""])[0] or "")
    if not _LIVE_RUN.match(run):
        return {"state": "error"}
    from plexus.paths import graphs_data_path
    st = os.path.join(graphs_data_path(*run.split("/")), "live_state.npz")
    if not os.path.exists(st):
        return {"state": "nostate"}
    mt = int(os.path.getmtime(st))
    cut = (q.get("cut", ["1"])[0] or "1") in ("1", "on", "true")
    look = "soft" if (q.get("look", ["ink"])[0] or "ink") == "soft" else "ink"
    _LIVE_LOOKS.add(look)
    png = _live_png(run, mt, cut, look)
    if os.path.exists(png):
        return {"state": "ready", "url": f"/api/watch/liverenderimg?run={run}&t={mt}&cut={int(cut)}&look={look}", "mtime": mt}
    if _LIVE_JOBS.get((run, mt, cut, look)) == "failed":
        return {"state": "failed"}
    _live_submit(run, mt, cut, st, look)
    return {"state": "rendering"}


def live_render_file(q):
    run = (q.get("run", [""])[0] or "")
    if not _LIVE_RUN.match(run):
        return None
    try:
        mt = int(q.get("t", ["0"])[0])
    except ValueError:
        return None
    f = _live_png(run, mt, (q.get("cut", ["1"])[0] or "1") in ("1", "on", "true"),
                  "soft" if (q.get("look", ["ink"])[0] or "ink") == "soft" else "ink")
    return f if os.path.exists(f) else None


def g_watch_live(h, q):
    _live_monitor()                      # the first page that asks starts the pre-rendering of the running jobs' states
    return {"runs": live_frames()}


# ---------------------------------------------------------------- the four read-only routes

def g_watch(h, q):
    """The page itself. Returned as raw HTML by the server's picture path, not as JSON."""
    return PAGE


_MOVIES_CACHE: dict = {}


def _movies(folder: str, fs: list) -> list:
    """Positions in `fs` whose step has a playable movie, kept 10 s per folder like `shots`: two stats of a network disk per step, every poll."""
    hit = _MOVIES_CACHE.get(folder)
    if hit is not None and time.time() - hit[0] < 10.0 and hit[1] == len(fs):
        return hit[2]
    out = [k for k, ff in enumerate(fs) if _playable(sidecar(ff, "mp4"))]
    _MOVIES_CACHE[folder] = (time.time(), len(fs), out)
    return out


def g_watch_state(h, q):
    """Which step is shown and the texts beside it."""
    folder = record_dir(q)
    fs = shots(folder)
    try:
        i = int(q.get("i", ["-1"])[0])
    except (ValueError, TypeError):
        i = -1
    i = (len(fs) - 1) if (i < 0 or i >= len(fs)) else i
    f = fs[i] if fs else None
    # WHICH STEPS HAVE A MOVIE, so the second button row can walk those and skip the rest. A
    # measurement graph and a pre-run camera shot are steps too, and in a record of 320 they
    # outnumber the runs -- stepping one at a time to get from one movie to the next is the thing
    # the first row is bad at. Published as positions in `fs`, which is exactly what `idx` indexes.
    st = {"n_shots": len(fs), "index": i if fs else 0, "rec": _rec_id(folder),
          "movies": _movies(folder, fs)}
    if f:
        st.update(shot=True, shot_name=os.path.basename(f),
                  shot_mtime=int(os.path.getmtime(f)),
                  shot_age=int(time.time() - os.path.getmtime(f)),
                  why=_read(sidecar(f, "why")), spec=_read(sidecar(f, "spec")),
                  # the step's saved movie description (`caption/NNNN.txt`), when the record carries one
                  caption=_read(sidecar(f, "caption")),
                  mp4=os.path.exists(sidecar(f, "mp4")),
                  mp4_play=_playable(sidecar(f, "mp4")),
                  # THE MOVIE'S OWN TIME, so a movie re-rendered under an unchanged still is fetched
                  # afresh: the page keyed the <video> on the still's mtime, and a re-render (exp_02
                  # step 0052, the shells closed) played from the browser's cache as the old file.
                  mp4_mtime=(int(os.path.getmtime(sidecar(f, "mp4"))) if os.path.exists(sidecar(f, "mp4")) else 0))
    return st


# A MOVIE OF FEWER THAN FIVE FRAMES IS NOT PLAYED. Three frames at 60 fps last 0.05 s: the <video> shows one frame and loops
# the other two too fast to see, which is a flicker and not a movie (exp02, the cell render of 3 frames, 2026-10-31). The page
# shows the still instead and the movies-only row skips the step. The count is read from the file's sample table, not decoded.
MIN_MOVIE_FRAMES = 5
_FRAME_COUNTS: dict = {}


def _mp4_frames(path):
    """Frames in an mp4, from moov/trak/mdia/minf/stbl/stsz (the sample count), or None when the file does not say."""
    try:
        key = (path, os.path.getmtime(path))
    except OSError:
        return None
    if key in _FRAME_COUNTS:
        return _FRAME_COUNTS[key]

    def boxes(f, lo, hi):
        pos = lo
        while pos + 8 <= hi:
            f.seek(pos)
            size, typ = struct.unpack(">I4s", f.read(8))
            hdr = 8
            if size == 1:
                size, hdr = struct.unpack(">Q", f.read(8))[0], 16
            elif size == 0:
                size = hi - pos
            if size < hdr:
                return
            yield typ, pos + hdr, pos + size
            pos += size

    def walk(f, lo, hi, chain):
        for typ, a, b in boxes(f, lo, hi):
            if typ == chain[0]:
                if len(chain) == 1:
                    yield a
                else:
                    yield from walk(f, a, b, chain[1:])

    n = None
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            for a in walk(f, 0, f.tell(), [b"moov", b"trak", b"mdia", b"minf", b"stbl", b"stsz"]):
                f.seek(a + 8)
                n = struct.unpack(">I", f.read(4))[0]
                break
    except (OSError, struct.error):
        n = None
    if not n:                                    # 0 samples: a fragmented mp4 keeps them in moof boxes, so the count is unknown
        n = None
    _FRAME_COUNTS[key] = n
    return n


def _playable(path):
    """Whether the step's movie exists and has enough frames to play; an mp4 whose count cannot be read is played."""
    if not os.path.exists(path):
        return False
    n = _mp4_frames(path)
    return n is None or n >= MIN_MOVIE_FRAMES


def _nth(q, kind):
    fs = shots(record_dir(q))
    try:
        i = int(q.get("i", ["-1"])[0])
    except (ValueError, TypeError):
        i = -1
    f = (fs[i] if 0 <= i < len(fs) else (fs[-1] if fs else None))
    if not f:
        return None
    return f if kind == "png" else sidecar(f, kind)


def g_watch_shot(h, q):
    return _nth(q, "png")


def g_watch_mp4(h, q):
    return _nth(q, "mp4")


ICON_DIR = os.path.join(REPO, "presentation", "icons")
EQ_CACHE = os.path.join(REPO, "log", "watch_eq")
# THE EQUATION'S PRINT SIZE, as dvipng's resolution. 105 set them larger than the text around them
# (user, 2026-09-25: "diminish the size"); 80 sits them at the pane's reading size. It is part of
# the cache key and of the image URL, so a change here is never served from an old picture.
EQ_DPI = 200                 # rendered at 200 dpi and shown at half its pixels (Cedric, 2026-10-12: the 80-dpi
                             # images read blurred); the size attributes come from the PNG itself
# THE KINDS THE DECK HAS AN ICON FOR. A whitelist and not a path join, because `kind` arrives in a
# query string: `?kind=../../etc/passwd` must reach nothing. It is also the honest list -- a kind
# with no icon draws nothing rather than a broken image.
ICON_KINDS = ("aggregate", "broadcast", "die", "divide", "exchange", "field", "field_dyn",
              "lateral", "rewire", "seed", "set", "structural")


_SYM_TEX: dict = {}       # hash -> LaTeX of a legend symbol this server has typeset (the route serves only these)


def sym_url(tex: str):
    """(url, (w, h)) for a legend symbol typeset at EQ_DPI, or None when it does not typeset. The LaTeX is registered
    under its hash so `/api/watch/tex?h=` can serve it without any LaTeX arriving from the browser."""
    import hashlib
    if not tex:
        return None
    p = eq_png(tex)
    if not p:
        return None
    h = hashlib.sha1(f"{EQ_DPI}:{tex}".encode()).hexdigest()[:16]
    _SYM_TEX[h] = tex
    tf = os.path.join(EQ_CACHE, f"sym_{h}.tex")              # a SAVED pane names symbols by hash after a restart too
    if not os.path.exists(tf):
        with open(tf, "w") as f:
            f.write(tex)
    return f"/api/watch/tex?h={h}", eq_size(tex)


def watch_tex(q) -> str | None:
    """`/api/watch/tex?h=H` -- a legend symbol typeset earlier by this server, by its hash."""
    h = (q.get("h", [""])[0] or "").strip()
    tex = _SYM_TEX.get(h)
    if tex is None and re.fullmatch(r"[0-9a-f]{16}", h):    # registered by an earlier server: read from disk, never the URL
        tf = os.path.join(EQ_CACHE, f"sym_{h}.tex")
        tex = open(tf).read() if os.path.exists(tf) else None
    return eq_png(tex) if tex else None


def eq_size(tex: str):
    """(width, height) in pixels of the typeset equation, from the cached PNG's header; (0, 0) when it cannot be set."""
    p = eq_png(tex)
    if not p:
        return (0, 0)
    try:
        import struct
        with open(p, "rb") as fh:
            head = fh.read(24)
        return struct.unpack(">II", head[16:24]) if head[:8] == b"\x89PNG\r\n\x1a\n" else (0, 0)
    except Exception:                                            # noqa: BLE001
        return (0, 0)


def eq_png(tex: str) -> str | None:
    """A LaTeX equation as a transparent PNG, white on nothing, cached on disk. Path, or None.

    WHY LATEX AND NOT THE BROWSER. There is no network in this container (a CDN fetch returns
    nothing at all) and no KaTeX or MathJax vendored in the tree, so a page cannot typeset maths
    on its own. There IS a full TeX here, because the deck is built with it -- `latex` and
    `dvipng` are both on PATH -- so the pane sets its equations with the very same typesetter the
    slides use, and the two cannot look different.

    WHY THE TEX NEVER COMES FROM THE QUERY STRING. This shells out to `latex`. The route addresses
    an equation by WHICH ONE IT IS -- step i, equation k -- and looks the body up server-side from
    the spec, so nothing a browser sends is ever compiled.

    Cached under `log/watch_eq/<sha1>.png`: the same equation appears on many steps, and a spec
    re-opened is the commonest thing a watcher does.
    """
    import hashlib
    import shutil
    import subprocess
    import tempfile
    if not tex.strip():
        return None
    os.makedirs(EQ_CACHE, exist_ok=True)
    out = os.path.join(EQ_CACHE, hashlib.sha1(f"{EQ_DPI}:{tex}".encode()).hexdigest()[:20] + ".png")
    if os.path.exists(out):
        return out
    # ONE UNBREAKABLE BOX ON A WIDE PAGE. Set as bare inline maths in an article of default text
    # width, a long equation was broken at a relation like a sentence -- May-Leonard's came out as
    # "(1 - p -" on one line and "a u)" on the next, which reads as two wrong formulas. `\mbox`
    # forbids the break and a 100 cm page gives it room; `-T tight` then crops to the ink, so the
    # page size never shows.
    doc = (r"\documentclass[12pt]{article}\usepackage{amsmath,amssymb,bm}"
           r"\usepackage[paperwidth=100cm,paperheight=30cm,margin=1cm]{geometry}"
           r"\pagestyle{empty}\begin{document}"
           r"\mbox{$\displaystyle " + tex + r"$}\end{document}")
    tmp = tempfile.mkdtemp(prefix="plexus_eq_")
    try:
        with open(os.path.join(tmp, "e.tex"), "w") as f:
            f.write(doc)
        r = subprocess.run(["latex", "-interaction=nonstopmode", "-halt-on-error", "e.tex"],
                           cwd=tmp, capture_output=True, text=True, timeout=40)
        if not os.path.exists(os.path.join(tmp, "e.dvi")):
            print(f"[watch] latex failed on an equation: {(r.stdout or '')[-300:]}", flush=True)
            return None
        # -bg Transparent + -fg white: the watcher's page is black, and a PNG with a white box
        # around every equation is what a default render gives.
        # ONE DPI FOR EVERY EQUATION, AND THE PAGE NEVER RESCALES ONE. At 220 the images ran from
        # 308 to 1194 px wide, and `max-width:100%` then shrank the wide ones to the pane while
        # leaving the narrow ones untouched -- so a rewire's short formula was set twice the size of
        # the lateral's long one, on the same slide. A uniform font is a uniform SCALE, which means
        # no CSS size constraint at all: rendered here at a dpi whose widest output fits a pane, and
        # displayed at its natural pixel size. A formula wider than the pane scrolls rather than
        # shrinking, because shrinking it is what made the two disagree.
        subprocess.run(["dvipng", "-D", str(EQ_DPI), "-T", "tight", "-bg", "Transparent",
                        "-fg", "rgb 1.0 1.0 1.0", "-o", "e.png", "e.dvi"],
                       cwd=tmp, capture_output=True, text=True, timeout=40)
        src = os.path.join(tmp, "e.png")
        if not os.path.exists(src):
            return None
        shutil.move(src, out)
        return out
    except Exception as e:                                       # noqa: BLE001
        print(f"[watch] equation render unavailable ({type(e).__name__}: {e})", flush=True)
        return None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


_SUMMARISED: dict = {}


def _summarise(path: str) -> dict:
    """`spec_summary.summarise(path)`, and when THIS process's operator registry is older than the spec,
    the same call in a fresh interpreter.

    THE REGISTRY IS FROZEN AT THE WATCHER'S START. `plexus.operators` registers every operator once,
    when it is first imported, and a registration cannot be repeated (`register_operator` refuses a
    second variant of the same name). A watcher left running while a session adds operators -- exp04
    added `shape_match`, `slab_barrier`, `residue_slab` and `friction_zone` to `channel_ops.py` after
    its watcher started -- then failed every new spec with "operator 'shape_match' not in registry",
    while the same summary in a fresh process read all 75 operators. So a registry miss is answered
    by a subprocess that imports the library as it is NOW; its JSON is rendered here as before, and
    cached on the spec's mtime like the rest, so it is paid once per step.
    """
    key = (path, os.path.getmtime(path))
    hit = _SUMMARISED.get(path)
    if hit is not None and hit[0] == key:
        return hit[1]
    try:
        s = _summary_module().summarise(path)
    except Exception as e:                                       # noqa: BLE001
        if "not in registry" not in str(e) and "no variant" not in str(e):
            raise
        import subprocess
        import sys as _sys
        code = ("import importlib.util, json, sys; "
                "p = sys.argv[1]; t = sys.argv[2]; "
                "spec = importlib.util.spec_from_file_location('plexus_spec_summary', t); "
                "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
                "print('@@JSON@@' + json.dumps(m.summarise(p), default=str))")
        env = dict(os.environ)
        env["PYTHONPATH"] = os.path.join(REPO, "src") + os.pathsep + env.get("PYTHONPATH", "")
        r = subprocess.run([_sys.executable, "-c", code, path, os.path.join(REPO, "tools", "spec_summary.py")],
                           capture_output=True, text=True, timeout=300, cwd=REPO, env=env)
        line = next((ln for ln in r.stdout.splitlines() if ln.startswith("@@JSON@@")), None)
        if line is None:
            raise RuntimeError(f"{e} -- and a fresh interpreter could not summarise it either: "
                               f"{(r.stderr or '').strip().splitlines()[-1:] or ['no output']}") from e
        s = json.loads(line[len("@@JSON@@"):])
    _SUMMARISED[path] = (key, s)
    return s


def _summary_module():
    """`tools/spec_summary.py`, imported by path and kept, so the registry walk is paid once."""
    m = globals().get("_SPEC_SUMMARY_MOD")
    if m is None:
        import importlib.util
        p = os.path.join(REPO, "tools", "spec_summary.py")
        spec = importlib.util.spec_from_file_location("plexus_spec_summary", p)
        m = importlib.util.module_from_spec(spec)
        import sys as _sys
        _sys.modules["plexus_spec_summary"] = m            # `spec_story` reuses this one, not a second copy
        spec.loader.exec_module(m)
        globals()["_SPEC_SUMMARY_MOD"] = m
    return m


_PANE_JOBS: dict = {}
_PANE_LAST: dict = {}


def _pane_async(key, fn, wait: float = 1.5) -> dict:
    """A PANE THAT TAKES LONG IS DRAWN IN THE BACKGROUND, and the request answers within `wait` seconds either way. The plexus and math panes typeset
    equations and call the story agent: 5 s to 100 s when the machine is loaded, and every such request held one of the browser's six connections to the
    server, so the picture, the movie and the next click queued behind them (Cedric, 2026-11-07: the panes and the 4D view "never show", a click on
    previous step changed nothing for seconds). The reply is the finished pane, or `busy: true` -- the page asks again a moment later and finds the job's result."""
    job = _PANE_JOBS.get(key)
    last = _PANE_LAST.get(key)
    if job is None and last is not None and time.time() - last[0] < 20.0 and last[1].get("pending"):
        return last[1]                       # a pane whose story the agent is still writing was drawn a moment ago: the page asks again every 10 s, and each redraw cost 30 s of the loaded machine
    if job is None:
        job = {"done": threading.Event(), "res": None, "t": time.time()}
        _PANE_JOBS[key] = job

        def run():
            try:
                job["res"] = fn()
                _PANE_LAST[key] = (time.time(), job["res"])
                print(f"[watch] {key[0]} pane drawn in {time.time() - job['t']:.1f} s", flush=True)
            except Exception as e:                                # noqa: BLE001
                job["res"] = {"text": f'<div class="ps-g">this pane could not be drawn: {html.escape(type(e).__name__)}: {html.escape(str(e)[:200])}</div>',
                              "pending": False, "saved": False}
            finally:
                job["done"].set()
        threading.Thread(target=run, daemon=True).start()
    if job["done"].wait(wait):
        if _PANE_JOBS.get(key) is job:
            del _PANE_JOBS[key]
        return job["res"]
    return {"text": "", "pending": True, "busy": True, "saved": False}


# PANE_VERSION: bump when a pane's drawing changes, so the saved panes (`story/NNNN.<kind>.html`) are redrawn once. It stands in for the code's
# mtimes, which used to be in the key: an edit to any operator file or one push then threw away every saved pane (Cedric, 2026-11-05: a saved
# plexus or math pane "to appear faster").
PANE_VERSION = "2026-11-14a"                       # 2026-11-14a: the plexus pane's first heading reads "states", not "entities" (Cedric)


def _pane_cache(story_path: str | None, kind: str, spec_path: str | None, extra: str):
    """(path, key) of a pane's SAVED HTML beside the step's story (`story/NNNN.<kind>.html`), and the key it must carry: the story's and the
    spec's mtimes, PANE_VERSION, the equation resolution and the URL parameters. A pane is drawn once and read back from disk after that
    (Cedric, 2026-10-12: "we should have a saved file to be read fast"); a hit is two stats and one read. Delete the file, or bump
    PANE_VERSION, to force a redraw."""
    if not story_path or not os.path.exists(story_path):
        return None, None
    parts = [os.path.getmtime(story_path)]
    if spec_path and os.path.exists(spec_path):
        parts.append(os.path.getmtime(spec_path))
    key = "|".join(f"{x:.3f}" for x in parts) + f"|{PANE_VERSION}|{EQ_DPI}|{extra}"
    return story_path[:-5] + f".{kind}.html", key


def _pane_read(path, key):
    try:
        with open(path) as f:
            head = f.readline().rstrip("\n")
            return f.read() if head == f"<!--key {key}-->" else None
    except OSError:
        return None


def _pane_write(path, key, text):
    try:
        with open(path + ".tmp", "w") as f:
            f.write(f"<!--key {key}-->\n{text}")
        os.replace(path + ".tmp", path)
    except OSError:
        pass


def _pane_key(kind: str, q) -> tuple:
    return (kind, tuple(sorted((k, tuple(v)) for k, v in q.items() if k in ("i", "rec"))))


def _saved_pane_fast(kind: str, q):
    """A pane SAVED beside its step is served as it is until the story module has loaded. Checking a saved pane's key needs that module (`resolve`), and its first
    import -- the operator registry -- took 245 s on a loaded machine, so the panes of a record whose panes were all saved still showed "reading..." for four
    minutes after every restart. Once the module is in, the exact key check is cheap and takes over (a pane whose story changed since is redrawn then)."""
    if globals().get("_SPEC_STORY_MOD") is not None:
        return None
    png = _nth(q, "png")
    if not png:
        return None
    try:
        with open(sidecar(png, "story")[:-5] + f".{kind}.html") as f:
            if f"|{PANE_VERSION}|" not in f.readline():       # saved by an older pane version: not served, the redraw takes it (2026-11-14: the "states" heading)
                return None
            return {"text": f.read(), "pending": False, "saved": True}
    except OSError:
        return None


def g_watch_plexus(h, q):
    """The whole pane is a background job, cache hit included: even reading a saved pane first imports the story module (the operator registry), which
    held the first request for 120 s on a loaded machine. Answers within 1.5 s: the pane, or `busy`."""
    return _saved_pane_fast("plexus", q) or _pane_async(_pane_key("plexus", q), lambda: _plexus_pane(q))


def _plexus_pane(q):
    """`/api/watch/plexus?i=N` -- step N's spec, read as the paper's three questions.

    A ROUTE OF ITS OWN RATHER THAN A FIELD ON `state`, because `state` is polled every 1.5 s and
    this is the only thing the watcher serves that costs more than reading a file. Read-only like
    every other route here: it parses a spec and queries the registry, and writes nothing.
    """
    fs = shots(record_dir(q))
    try:
        i = int(q.get("i", ["-1"])[0])
    except (ValueError, TypeError):
        i = -1
    i = (len(fs) - 1) if (i < 0 or i >= len(fs)) else i
    # THE RECORD'S OWN ID, whether or not the page sent `rec=`: a page opened on the server's default record sends none, one that picked it from the list sends its id, and
    # the saved pane's key and links carried the raw query value -- so the two forms overwrote each other's file and the pane was drawn again at every switch (Cedric, 2026-11-07).
    rec = _rec_id(record_dir(q))
    # A STEP WITHOUT A SPEC (a measurement, a figure) tells the run its note names, else the run before it.
    r = _story_module().resolve(_nth(q, "png")) or {}
    spec = r.get("spec") or _nth(q, "spec")
    cpath, ckey = _pane_cache(r.get("story"), "plexus", spec, f"{i}|{rec}|{r.get('lead', '')}")
    if cpath:
        hit = _pane_read(cpath, ckey)
        if hit is not None:
            return {"text": hit, "pending": False, "saved": True}
    def draw():
        text, pending = spec_summary(spec, q_index=i, rec=rec, why_path=r.get("why"),
                                     story_path=r.get("story"), png=r.get("png"), lead=r.get("lead", ""))
        if cpath and not pending:
            _pane_write(cpath, ckey, text)
        return {"text": text, "pending": bool(pending), "saved": bool(cpath and not pending)}
    return draw()


def g_watch_math(h, q):
    return _saved_pane_fast("math", q) or _pane_async(_pane_key("math", q), lambda: _math_pane(q))


def _math_pane(q):
    """`/api/watch/math?i=N` -- step N's equations at once, in one notation, located and with one legend
    (tools/spec_story.py `render_math_html`); written in the background when the saved story lacks it."""
    fs = shots(record_dir(q))
    try:
        i = int(q.get("i", ["-1"])[0])
    except (ValueError, TypeError):
        i = -1
    i = (len(fs) - 1) if (i < 0 or i >= len(fs)) else i
    sm = _story_module()
    r = sm.resolve(_nth(q, "png")) or {}
    p = r.get("spec") or _nth(q, "spec")
    if not p or not os.path.exists(p):
        return {"text": '<div class="ps-g">no spec recorded for this step</div>', "pending": False}
    with open(p, errors="replace") as fh:                        # a spec of pure comments is a MEASUREMENT step (as the plexus pane answers), not a broken spec
        if not [ln for ln in fh.read().splitlines() if ln.strip() and not ln.lstrip().startswith("#")]:
            return {"text": '<div class="ps-g">This step is a MEASUREMENT, not a run: no spec, so no equations.</div>', "pending": False}
    rec = _rec_id(record_dir(q))                                  # the record's own id, as in `g_watch_plexus`
    cpath, ckey = _pane_cache(r.get("story"), "math", p, f"{i}|{rec}|{r.get('lead', '')}")
    if cpath:
        hit = _pane_read(cpath, ckey)
        if hit is not None:
            return {"text": hit, "pending": False, "saved": True}
    def draw():
        try:
            _s, F, st = _story(p, r.get("why"), r.get("story"), r.get("png"))
        except Exception as e:                                       # noqa: BLE001
            return {"text": f'<div class="ps-g">the spec could not be read: {html.escape(str(e))}</div>', "pending": False}
        fb = bool(st.get("fallback"))                                # no agent story yet: the registered equations, no agent call
        if sm.missing_math(F, st) and not fb:
            sm.ensure_math(F, st, r.get("story"))
        pending = sm.pending(F) if fb else sm.math_pending(r.get("story"))
        import hashlib as _hl
        rows = sm.math_equations(F, st)
        hs = [_hl.sha1(f"{EQ_DPI}:{t}".encode()).hexdigest()[:12] for _i, t, _o, _w in rows]

        def _sz(k):                      # the one notation when it typesets, else the step's own (as `watch_meq` serves)
            return (eq_size(rows[k][1]) if eq_png(rows[k][1]) else eq_size(rows[k][2])) if k < len(rows) else (0, 0)
        text = sm.render_math_html(F, st, lambda k: f"/api/watch/meq?i={i}&k={k}&h={hs[k] if k < len(hs) else 0}"
                                                     + f"&rec={rec}", pending=pending,
                                   meq_size=_sz, sym_url=sym_url)
        lead = r.get("lead", "")
        if lead:
            text = f'<div class="ps-g" style="margin-bottom:6px">{html.escape(lead)}</div>' + text
        saved = bool(cpath and not pending and not sm.missing_math(F, st))
        if saved:
            _pane_write(cpath, ckey, text)
        return {"text": text, "pending": bool(pending), "saved": saved}
    return draw()


_HIER_CACHE: dict = {}


_DRIVE: dict = {"n": 0, "cmds": [], "log": [], "pages": {}}
_DRIVE_OK = ("step", "go", "pane", "open3d", "def", "live", "snap")


def g_watch_drive(h, q):
    """`/api/watch/drive` -- a script drives the open page, and reads back what the page did. `?do=step&arg=-1&rec=<record id>` queues a command (only the page showing that record runs it; no `rec` -- every page) (`step`, `go` to a 0-based
    position, `pane` plexus|math|hierarchy|spec, `open3d`, `def` the flame button, `live`, `snap`); the page polls `?after=<n>` for the ones it has not run and answers with
    `?ev=<name>&cmd=...&ms=...&d=...`; `?log=1` returns those answers and whether a page is polling. It queues commands for a page and changes nothing
    else, but anyone who reaches this port can use it, as they can already open a step in 3-D."""
    one = lambda k: (q.get(k, [""])[0] or "")
    if one("do"):
        if one("do") not in _DRIVE_OK:
            return {"error": f"do= one of {', '.join(_DRIVE_OK)}"}
        _DRIVE["n"] += 1
        _DRIVE["cmds"] = (_DRIVE["cmds"] + [{"n": _DRIVE["n"], "do": one("do"), "arg": one("arg"), "rec": one("rec")}])[-20:]
        _DRIVE["log"] = [e for e in _DRIVE["log"] if e["t"] > time.time() - 600]
        return {"queued": _DRIVE["n"], "pages": sorted(r for r, t in _DRIVE["pages"].items() if time.time() - t < 5)}
    if one("ev"):
        try:
            ms = int(float(one("ms") or 0))
        except ValueError:
            ms = 0
        _DRIVE["log"] = (_DRIVE["log"] + [{"t": time.time(), "cmd": one("cmd"), "ev": one("ev"), "ms": ms, "d": one("d")[:600]}])[-300:]
        return {"ok": True}
    if one("log"):
        return {"page_polling": bool(_DRIVE["pages"]) and time.time() - max(_DRIVE["pages"].values()) < 5, "n": _DRIVE["n"],
                "pages": sorted(r for r, t in _DRIVE["pages"].items() if time.time() - t < 5), "log": _DRIVE["log"]}
    _DRIVE["pages"][one("rec")] = time.time()                    # which records have a page open, so a command can name the one it is for
    if "after" not in q:
        return {"n": _DRIVE["n"], "cmds": []}
    after = int(one("after") or 0)
    return {"n": _DRIVE["n"], "cmds": [c for c in _DRIVE["cmds"] if c["n"] > after]}


def g_watch_hierarchy(h, q):
    return _pane_async(_pane_key("hier", q), lambda: _hierarchy_pane(q))


def _hierarchy_pane(q):
    """`/api/watch/hierarchy?i=N` -- the hierarchy pane of step N: the graph of tools/spec_hierarchy.py for the spec the `plexus` pane
    reads (`spec_story.resolve`), as an inline PNG; for a MOVIE step -- its note names a run that stores 5 rows or more -- two
    panels, the entities at the start of the run and at its end. Cached per spec file, run trajectory (mtime), look and tool version."""
    import base64
    import importlib.util
    r = _story_module().resolve(_nth(q, "png")) or {}
    p = r.get("spec") or _nth(q, "spec")
    if not p or not os.path.exists(p):
        return {"text": '<div class="ps-g">this step saved no spec</div>'}
    tool = os.path.join(REPO, "tools", "spec_hierarchy.py")
    sp = importlib.util.spec_from_file_location("spec_hierarchy", tool)
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    why = r.get("why") or _nth(q, "why")
    run = mod.run_dir_of(why)
    traj = os.path.join(run, "trajectory.npz") if run else None
    key = (os.path.realpath(p), os.path.getmtime(p), os.path.getmtime(tool), os.path.getmtime(traj) if traj else 0.0,
           os.path.getmtime(os.path.join(REPO, "tools", "spec_summary.py")))
    if key not in _HIER_CACHE:
        print(f"[watch] hierarchy rendering ... {os.path.relpath(p, REPO)}" + (f"  (start and end of {os.path.relpath(run, REPO)})" if run else ""), flush=True)
        _t0 = time.time()
        try:
            png = mod.png_bytes(p, why=why)
        except Exception as e:                                   # noqa: BLE001
            print(f"[watch] hierarchy: {type(e).__name__}: {e}", flush=True)
            return {"text": f'<div class="ps-g">the hierarchy could not be drawn: {html.escape(type(e).__name__)}: {html.escape(str(e)[:160])}</div>'}
        print(f"[watch] hierarchy rendered in {time.time() - _t0:.1f} s", flush=True)
        if len(_HIER_CACHE) >= 12:
            _HIER_CACHE.clear()
        _HIER_CACHE[key] = ('<div class="ps-g">this spec declares no containment (no set inside another)</div>' if png is None else
                            '<img style="max-width:100%;border:0" src="data:image/png;base64,' + base64.b64encode(png).decode() + '">')
    return {"text": _HIER_CACHE[key]}


INSIGHTS_ARROW = {"up": "↑", "progress": "↑", "right": "→", "status quo": "→", "down": "↓", "regression": "↓"}   # the trend words of the deck's notes -> the arrow after the heading


def g_watch_insights(h, q):
    """`/api/watch/insights?i=N` -- the insights saved beside step N, `<record>/story/NNNN.insights.json` = {title, insights, trend, trend_why}: what the deck
    shows in its yellow "insights" section. The file is written by `tools/watcher_deck.py --export-insights` from the deck's notes YAML, the single source.
    A plain file read that imports nothing, so it answers at once even while the story module is still loading; a step without the file answers with a
    one-line note."""
    png = _nth(q, "png")
    try:
        with open(sidecar(png, "story")[:-5] + ".insights.json") as f:
            d = json.load(f)
        body = str(d["insights"]).strip()
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return {"text": '<div class="ps-g">no insights written for this step</div>', "pending": False, "saved": False}
    arrow = INSIGHTS_ARROW.get(str(d.get("trend", "")).strip().lower(), "")
    tail = " ".join(x for x in (arrow, html.escape(str(d.get("trend_why", "")).strip())) if x)
    paras = "".join("<p>" + html.escape(p.strip()).replace("\n", "<br>") + "</p>" for p in re.split(r"\n\s*\n", body) if p.strip())
    head = '<div class="ps-h">insights' + (f' <span class="ps-ins">{tail}</span>' if tail else "") + "</div>"
    return {"text": head + f'<div class="ps-story">{paras}</div>', "pending": False, "saved": True}


def watch_meq(q) -> str | None:
    """`/api/watch/meq?i=N&k=K` -- the K-th equation of the MATH pane, typeset; the one notation when it has
    been written and typesets, else the step's own equation."""
    try:
        k = int(q.get("k", ["0"])[0])
        sm = _story_module()
        r = sm.resolve(_nth(q, "png")) or {}
        p = r.get("spec") or _nth(q, "spec")
        if not p or not os.path.exists(p):
            return None
        _s, F, st = _story(p, r.get("why"), r.get("story"), r.get("png"))
        rows = sm.math_equations(F, st)
        if not (0 <= k < len(rows)):
            return None
        _i, tex, orig, _w = rows[k]
        return eq_png(tex) or eq_png(orig)
    except Exception as e:                                       # noqa: BLE001
        print(f"[watch] math equation {q.get('k')}: {type(e).__name__}: {e}", flush=True)
        return None


def g_watch_records(h, q):
    """`/api/watch/records` -- every record this watcher can show, and the one it opens on."""
    return {"records": [{k: r[k] for k in ("id", "root", "name", "n")} for r in records()],
            "default": _rec_id(HISTORY)}


def watch_icon(q) -> str | None:
    """The deck's glyph for an operator kind, with its background made transparent. Path, or None.

    THE FILES ARE OPAQUE BLACK BEHIND THE ART. They are drawn for a black slide, so on a slide the
    square is invisible and nobody has had to care. On this page the pane is #0f1319 rather than
    #000, so the square showed as a dark tile around every icon; inverting it in CSS -- the first
    attempt -- turned that black background WHITE and made the tile worse.

    So the alpha is DERIVED FROM THE ART instead: a pixel's opacity is its own brightness, which
    takes the background to nothing, keeps the coloured strokes exactly as drawn, and keeps the
    anti-aliased edges soft rather than stair-stepped. Done here, once, and cached, rather than in
    the browser, because no CSS filter can turn a colour into transparency.

    The deck's own files are untouched: a slide still gets the black square it wants.
    """
    kind = (q.get("kind", [""])[0] or "").strip()
    if kind not in ICON_KINDS:
        return None
    src = os.path.join(ICON_DIR, kind + ".png")
    if not os.path.exists(src):
        return None
    out = os.path.join(EQ_CACHE, f"icon_{kind}.png")
    if os.path.exists(out) and os.path.getmtime(out) >= os.path.getmtime(src):
        return out
    try:
        import numpy as np
        from PIL import Image
        a = np.array(Image.open(src).convert("RGBA"))
        a[..., 3] = a[..., :3].max(axis=2)          # opacity = brightness: black -> nothing
        os.makedirs(EQ_CACHE, exist_ok=True)
        Image.fromarray(a).save(out)
        return out
    except Exception as e:                                       # noqa: BLE001
        print(f"[watch] icon {kind}: {type(e).__name__}: {e}", flush=True)
        return src                                   # the original is better than no icon at all


def watch_eq(q) -> str | None:
    """`/api/watch/eq?i=N&k=K` -- the K-th equation of step N's spec, typeset, as a PNG path.

    ADDRESSED BY POSITION, NOT BY CONTENT. The body is looked up here, from the spec the record
    already holds, so no LaTeX ever arrives from the browser -- which matters because rendering it
    means running `latex`. `spec_summary.equations_shown` walks the same dedup the HTML renderer
    walks, so the K-th image is the K-th equation that renderer emitted.
    """
    p = (_story_module().resolve(_nth(q, "png")) or {}).get("spec") or _nth(q, "spec")
    if not p or not os.path.exists(p):
        return None
    try:
        k = int(q.get("k", ["0"])[0])
        r = _story_module().resolve(_nth(q, "png")) or {}
        p = r.get("spec") or p
        _s, _F, _st = _story(p, r.get("why"), r.get("story"), r.get("png"))
        eqs = _story_module().equations(_F, _st)
        return eq_png(eqs[k]) if 0 <= k < len(eqs) else None
    except Exception as e:                                       # noqa: BLE001
        print(f"[watch] equation {q.get('k')} of {p}: {type(e).__name__}: {e}", flush=True)
        return None
