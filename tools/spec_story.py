"""The PLEXUS pane of the watcher as a STORY a person can read: a headline and a few sentences saying
what the scene is and what happens, the entities, then the GNN -- every step the simulation takes,
each with its kind's glyph, a plain-English name and one sentence -- listed in the order the story
tells it.

    python tools/spec_story.py <spec.yaml>            # write (or read back) the story, print it
    python tools/spec_story.py <spec.yaml> --facts    # only the fact sheet the agent is given

HALF SCRIPT, HALF AGENT. The script reads every FACT -- the sets and their counts, each operator's
registry title, its docstring's first sentence, what it acts on, its parameters, its equation, and
WHEN it runs (once at the start, every frame, or every substep and how many per frame) -- from the
spec and the live registry, so nothing in the pane can drift from either. An agent (`claude -p`, no
tools, everything inline, low effort) only WORDS them: a headline, a story of three to five
sentences, a plain-English name and a sentence for each step, and the order in which the steps are
best read beside the story. Its answer is checked (every step named, the order a permutation) and
anything missing falls back to the script's own wording, so a bad answer degrades the pane rather
than breaking it.

WHY AN AGENT. The script's own reading (`spec_summary.render_html`, replaced by this) grouped the
steps by the paper's operator KIND -- `lateral x6`, `structural x10`, `exchange x5` -- and named
each by its registry title. Every word of that was correct, and a reader who was not the developer
could not tell from it what the run IS: the kinds are the algebra's vocabulary, the titles were
written one operator at a time by different sessions, and nothing said what the scene was or why
these steps follow each other. That sentence needs the whole spec at once; a template cannot write
it, and a person should not have to.

ONCE PER SPEC, CACHED ON DISK. The answer is keyed on a hash of the fact sheet (and the prompt's
version), stored under log/spec_story/, so a spec is described once -- ~10-25 s -- and every later
view is a file read. The watcher asks for it in the background and shows the script's reading
until it lands.
"""
from __future__ import annotations

import hashlib
import html as htmlmod
import importlib.util
import json
import os
import re
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
CACHE = os.path.join(ROOT, "log", "spec_story")
PROMPT_VERSION = 11
MODEL = os.environ.get("PLEXUS_STORY_MODEL", "sonnet")
TIMEOUT_S = 420                  # a story took 288 s once (exp27_sph1, 15:56); 150 s timed 41 calls out


def _summary():
    m = sys.modules.get("plexus_spec_summary")
    if m is None:
        spec = importlib.util.spec_from_file_location("plexus_spec_summary", os.path.join(HERE, "spec_summary.py"))
        m = importlib.util.module_from_spec(spec)
        sys.modules["plexus_spec_summary"] = m
        spec.loader.exec_module(m)
    return m


# --------------------------------------------------------------------------- the facts (script)
def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, (list, tuple)) and len(v) <= 6:
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    return str(v)


def _github() -> str:
    """https://github.com/<owner>/<repo>, from the clone's own `origin` (ssh or https)."""
    g = globals().get("_GITHUB")
    if g is None:
        try:
            u = subprocess.run(["git", "-C", ROOT, "remote", "get-url", "origin"], capture_output=True,
                               text=True, timeout=5).stdout.strip()
            m = re.search(r"github\.com[:/](.+?)(?:\.git)?$", u)
            g = f"https://github.com/{m.group(1)}" if m else ""
        except Exception:                                        # noqa: BLE001
            g = ""
        globals()["_GITHUB"] = g
    return g


GITHUB_BRANCH = os.environ.get("PLEXUS_GITHUB_BRANCH", "main")


_PUSHED: dict = {}


def _pushed(rel: str) -> tuple[str, list[str]] | None:
    """(sha, lines) of `rel` in the last PUSHED commit of origin/<branch>, cached; None if absent there."""
    try:
        # THE PUSHED COMMIT IS RE-READ EVERY 60 s, not once per process: a watcher running for hours kept linking to the
        # commit it first saw (109ecf3c) after origin/main had moved on with corrected equations (2026-10-12)
        sha = _PUSHED.get("_sha")
        if sha is None or time.time() - _PUSHED.get("_t", 0.0) > 60.0:
            new = subprocess.run(["git", "-C", ROOT, "rev-parse", f"origin/{GITHUB_BRANCH}"], capture_output=True,
                                 text=True, timeout=10).stdout.strip()
            if new and new != sha:
                _PUSHED.clear()
            sha = new or sha
            _PUSHED["_sha"], _PUSHED["_t"] = sha, time.time()
        if rel not in _PUSHED:
            r = subprocess.run(["git", "-C", ROOT, "show", f"{sha}:{rel}"], capture_output=True, text=True, timeout=20)
            _PUSHED[rel] = r.stdout.splitlines() if r.returncode == 0 else None
        return (sha, _PUSHED[rel]) if sha and _PUSHED[rel] is not None else None
    except Exception:                                            # noqa: BLE001
        return None


def class_ref(cls) -> str:
    """`module:QualName` of a class, what a saved story keeps so the link is made when the page is drawn."""
    return f"{cls.__module__}:{cls.__qualname__}" if cls is not None else ""


def source_url(cls_or_ref, member: str | None = None) -> str:
    """A PERMALINK to a class on GitHub: the last pushed commit of origin/main, at the line where the class
    is defined IN THAT COMMIT -- so the page opens on the class, whatever the working tree has added above
    it since. A class not pushed yet links to the file in that commit, without a line.

    WITH `member` (an operator's `forward`): the line where that method is DEFINED, in whichever class of the MRO
    defines it -- possibly a base class in another file. A registered variant often only sets flags or a one-line
    hook (`Project` is EMIT + WRITES_BLOCK; the work is `_Project.forward`), so the class line pointed at code that
    computes nothing (Cedric, 2026-10-12: "class Project does not do much")."""
    import importlib
    import inspect
    cls = cls_or_ref
    if isinstance(cls_or_ref, str):
        if not cls_or_ref:
            return ""
        mod, _, qn = cls_or_ref.partition(":")
        try:
            cls = importlib.import_module(mod)
            for part in qn.split("."):
                cls = getattr(cls, part)
        except Exception:                                        # noqa: BLE001
            return ""
    owner = cls
    if member:
        owner = next((k for k in getattr(cls, "__mro__", ()) if member in vars(k)), None)
        if owner is None:
            return source_url(cls)
    try:
        rel = os.path.relpath(inspect.getsourcefile(owner), ROOT)
    except Exception:                                            # noqa: BLE001
        return ""
    if rel.startswith("..") or not _github():
        return ""
    got = _pushed(rel)
    if got is None:
        return f"{_github()}/blob/{GITHUB_BRANCH}/{rel}"
    sha, lines = got
    pat = re.compile(rf"^\s*class\s+{re.escape(owner.__name__)}\s*[(:]")
    n = next((i + 1 for i, ln in enumerate(lines) if pat.match(ln)), None)
    if member and n:                                     # the first `def member(` after the owner's class line
        mp = re.compile(rf"^\s+def\s+{re.escape(member)}\s*\(")
        n = next((i + 1 for i in range(n, len(lines)) if mp.match(lines[i])), n)
    return f"{_github()}/blob/{sha}/{rel}" + (f"#L{n}" if n else "")


def spec_url(path: str | None, set_name: str) -> str:
    """A permalink to a set's DECLARATION in its spec file (`  <name>:` under `sets:`), for a set no entity class
    describes (a `brain` container, a `retina` of signal lines) -- the spec is then the only code that says what it is."""
    if not path or not _github():
        return ""
    rel = os.path.relpath(os.path.realpath(path), ROOT)
    if rel.startswith(".."):
        return ""
    got = _pushed(rel)
    if got is None:
        return ""
    sha, lines = got
    i0 = next((i for i, ln in enumerate(lines) if re.match(r"^sets\s*:", ln)), None)
    if i0 is None:
        return ""
    pat = re.compile(rf"^\s+{re.escape(set_name)}\s*:")
    n = next((i + 1 for i in range(i0 + 1, len(lines)) if pat.match(lines[i])), None)
    return f"{_github()}/blob/{sha}/{rel}#L{n}" if n else ""


# --------------------------------------------------------------------------- the code each bullet is checked against
# THE RULE (Cedric, 2026-10-12: "each one must open exactly the code that bullet describes ... the right piece of code that
# needs to be checked"), one per kind of bullet:
#   GNN step     the `forward` that runs it (through the MRO); a variant that only overrides a hook used by that forward
#                (neuron_signal[shared].psi) gets the hook as a second link
#   learnable    the line of the reading step's `forward` where the learnable's value is read (`tau = lvl.get(self.tau_
#                block)`, `w_e = es.get(self.weight_block)`) -- where its value enters the model
#   entity       the state layout the run uses: the spec's inline `state:` block when it declares one (it outranks the
#                registry), else the entity's registered `*_schema` function
#   set          (no class, no inline state) its `sets:` entry in the spec
#   relation     its `sets:` entry in the spec (the line naming `edges_file`), and the tool that wrote that file
def _obj(ref):
    """`module:Qual.Name` -> the object, or None."""
    import importlib
    if not ref:
        return None
    if not isinstance(ref, str):
        return ref
    mod, _, qn = ref.partition(":")
    try:
        o = importlib.import_module(mod)
        for part in qn.split("."):
            o = getattr(o, part)
        return o
    except Exception:                                            # noqa: BLE001
        return None


def step_class(st_: dict):
    """The operator class of a saved step: its `cls` reference, else the registry's class for its `code`. A story saved while its operator was
    not yet registered keeps `cls: ""` for good (exp01's `spring_matrix`, 2026-10-30), and with no class a step has no EXPLAIN, so neither its row
    nor its equation could be clicked open."""
    c = _obj(st_.get("cls")) if st_.get("cls") else None
    if c is None and st_.get("code"):
        name, _, var = st_["code"].partition("[")
        try:
            import plexus.operators                              # noqa: F401
            c = _op_class(name, var.rstrip("]") or None)
        except Exception:                                        # noqa: BLE001
            c = None
    return c if isinstance(c, type) else None


def _src_of(obj):
    """(sha, rel, lines) of the pushed file that defines `obj` (a class or a function), or None."""
    import inspect
    try:
        rel = os.path.relpath(inspect.getsourcefile(obj), ROOT)
    except Exception:                                            # noqa: BLE001
        return None
    if rel.startswith("..") or not _github():
        return None
    got = _pushed(rel)
    return (got[0], rel, got[1]) if got else None


def _url(sha, rel, n):
    return f"{_github()}/blob/{sha}/{rel}" + (f"#L{n}" if n else "")


def _class_span(lines, name):
    """(first, last) 0-based line indices of `class name` and its body."""
    pat = re.compile(rf"^class\s+{re.escape(name)}\s*[(:]")
    a = next((i for i, ln in enumerate(lines) if pat.match(ln)), None)
    if a is None:
        return None
    b = next((i for i in range(a + 1, len(lines)) if re.match(r"^[^\s#)]", lines[i])), len(lines))
    return a, b


def _method(cls, member):
    """(sha, rel, lines, def_index, body_end) of `member` as the class's MRO resolves it, or None."""
    owner = next((k for k in getattr(cls, "__mro__", ()) if member in vars(k)), None)
    got = _src_of(owner) if owner is not None else None
    if not got:
        return None
    sha, rel, lines = got
    span = _class_span(lines, owner.__name__)
    if not span:
        return None
    mp = re.compile(rf"^(\s+)def\s+{re.escape(member)}\s*\(")
    d = next((i for i in range(span[0], span[1]) if mp.match(lines[i])), None)
    if d is None:
        return None
    ind = len(mp.match(lines[d]).group(1))
    e = next((i for i in range(d + 1, span[1]) if lines[i].strip() and not lines[i].lstrip().startswith("#")
              and len(lines[i]) - len(lines[i].lstrip()) <= ind), span[1])
    return sha, rel, lines, d, e


def explain_of(cls) -> dict | None:
    """What an operator does, in plain English, for a click on its equation: {text, own, source, where, url}.

    `text` is the class's own `EXPLAIN` string (the first one on its MRO, so an implementation variant such as the hashgrid
    contact_finder reads its base class's account); a class without one falls back to the first paragraph of its docstring
    (`source` says which). `where` is "file:line" of the class. EXPLAIN is a string, not a comment: the maths pane and the plexus
    pane both draw it from here, so the two cannot tell different things about one operator."""
    import inspect
    if cls is None:
        return None
    text, own, src = "", False, "docstring"
    for c in cls.__mro__:
        ex = c.__dict__.get("EXPLAIN")
        if isinstance(ex, str) and ex.strip():
            text, own, src = " ".join(ex.split()), c is cls, "EXPLAIN"
            break
    if not text:
        for c in cls.__mro__:
            if c.__module__.startswith("plexus.operators") and (c.__doc__ or "").strip():
                text = " ".join((inspect.getdoc(c) or "").split("\n\n")[0].split())
                break
    if not text:
        return None
    where = ""
    try:
        f = inspect.getsourcefile(cls)
        where = f"{os.path.relpath(f, ROOT)}:{inspect.getsourcelines(cls)[1]}"
    except (TypeError, OSError):
        pass
    try:
        url = source_url(cls)
    except Exception:                                            # noqa: BLE001
        url = ""
    return {"text": text, "own": own, "source": src, "where": where, "url": url}


def _explain_html(text: str, sym_url=None) -> str:
    """The explanation as HTML: `$...$` is LaTeX, typeset by the same typesetter as the legend's symbols (`sym_url`, LaTeX never
    comes from the browser) and set inline, shown as code when there is none or it does not typeset; `...` between backticks is a
    parameter name, set as code."""
    esc = htmlmod.escape
    out = []
    for k, part in enumerate(re.split(r"(\$[^$]+\$)", text)):
        if k % 2 == 1:
            tex = part[1:-1].strip()
            got = sym_url(tex) if sym_url else None
            if got:
                url, size = got
                dims = f' width="{size[0] // 2}" height="{size[1] // 2}"' if size and size[0] and size[1] else ""
                out.append(f'<img class="ps-tx" src="{esc(url)}" alt="{esc(tex)}"{dims}>')
            else:
                out.append(f"<code>{esc(tex)}</code>")
        else:
            out.append(re.sub(r"`([^`]+)`", lambda m: f"<code>{m.group(1)}</code>", esc(part).replace("&#x27;", "'")))
    return "".join(out)


def explain_block(st_: dict, hidden: bool = True, sym_url=None) -> str:
    """The hidden block a click opens under an equation or a step: `code (file:line)` then the explanation."""
    ex = explain_of(step_class(st_))
    if not ex:
        return ""
    esc = htmlmod.escape
    head = esc(st_.get("code", ""))
    loc = esc(ex["where"])
    if ex["url"] and loc:
        loc = f'<a class="ps-ln" href="{esc(ex["url"])}" target="_blank" onclick="event.stopPropagation()" title="the class on GitHub">{loc}</a>'
    note = "" if ex["source"] == "EXPLAIN" else ' <span class="ps-g">(the class docstring: no EXPLAIN written yet)</span>'
    return (f'<div class="ps-ex" style="{"display:none" if hidden else ""}"><span class="ps-exh">{head}'
            f'{" (" + loc + ")" if loc else ""}</span>{note}<br>{_explain_html(ex["text"], sym_url)}</div>')


def method_url(cls, member, token=None) -> tuple[str, str]:
    """(url, the line it opens) of `member`'s def line, or, with `token`, of the first line of its body holding it."""
    m = _method(cls, member)
    if not m:
        return "", ""
    sha, rel, lines, d, e = m
    n = d
    if token:
        hits = [i for i in range(d + 1, e) if token in lines[i] and not lines[i].lstrip().startswith("#")]
        n = next((i for i in hits if ".get(" in lines[i]), hits[0] if hits else d)
    return _url(sha, rel, n + 1), lines[n]


def token_url(cls, tokens) -> tuple[str, str]:
    """(url, line) of the first line holding one of `tokens` (preferring a `.get(` read) in `forward`'s body, else in any
    method the class resolves through its MRO (a helper `forward` calls), else ("", "")."""
    m = _method(cls, "forward")
    spans = []
    if m:
        spans.append(m)
    for k in getattr(cls, "__mro__", ()):
        got = _src_of(k) if k.__module__.startswith("plexus") else None
        if not got:
            continue
        sp = _class_span(got[2], k.__name__)
        if sp:
            spans.append((got[0], got[1], got[2], sp[0], sp[1]))
    def reads(ln, tok):
        """A line that READS the token: not a comment, a def, a dict key (`"theta_g": "..."`), an assignment TO it
        (`self.mesh_rate = torch.full(...)`), or the params lookup that sets it."""
        t = ln.strip()
        return (tok in ln and not t.startswith("#") and not re.match(r"def\s", t) and "params.get(" not in ln
                and "setattr(" not in ln
                and not re.search(re.escape(tok) + r"\s*:", ln) and not re.match(re.escape(tok) + r"\s*=[^=]", t))
    for tok in tokens:
        for sha, rel, lines, a, b in spans:
            hits = [i for i in range(a + 1, b) if reads(lines[i], tok)]
            if hits:
                n = next((i for i in hits if ".get(" in lines[i]), hits[0])
                return _url(sha, rel, n + 1), lines[n]
    return "", ""


def _hook(cls):
    """A method the variant defines itself and its inherited `forward` calls (`self.psi(`), or None."""
    import inspect
    if cls is None or "forward" in vars(cls):
        return None
    try:
        fwd = inspect.getsource(cls.forward)
    except Exception:                                            # noqa: BLE001
        return None
    own = [k for k, v in vars(cls).items() if callable(v) and not k.startswith("__")]
    return next((k for k in own if f"self.{k}(" in fwd), None)


def _attr_for_param(cls, param):
    """The instance attribute `__init__` sets from `params[param]` (`self.tau_block = params.get("tau")`)."""
    import inspect
    for k in getattr(cls, "__mro__", ()):
        if "__init__" not in vars(k):
            continue
        try:
            src = inspect.getsource(k.__init__)
        except Exception:                                        # noqa: BLE001
            continue
        m = re.search(rf"self\.(\w+)\s*=\s*[^\n]*params\.get\(\s*[\"']{re.escape(param)}[\"']", src)
        if m:
            att = m.group(1)
            # a block named by the param may be kept under a second name (`_b = params.get("bias")`; self.bias_block = _b)
            m2 = re.search(rf"self\.(\w*{re.escape(param)}\w*_block)\s*=", src)
            return m2.group(1) if m2 else att
        m = re.search(rf"(\w+)\s*=\s*params\.get\(\s*[\"']{re.escape(param)}[\"']", src)
        if m:
            m2 = re.search(rf"self\.(\w+)\s*=\s*{re.escape(m.group(1))}\b", src)
            if m2:
                return m2.group(1)
    return None


def _spec_lines(path):
    if not path:
        return None
    rel = os.path.relpath(os.path.realpath(path), ROOT)
    if rel.startswith(".."):
        return None
    got = _pushed(rel)
    return (got[0], rel, got[1]) if got else None


def _spec_set_line(lines, name):
    """0-based index of `  name:` under `sets:`, and the end of that set's block."""
    i0 = next((i for i, ln in enumerate(lines) if re.match(r"^sets\s*:", ln)), None)
    if i0 is None:
        return None, None
    pat = re.compile(rf"^(\s+){re.escape(name)}\s*:")
    a = next((i for i in range(i0 + 1, len(lines)) if pat.match(lines[i])), None)
    if a is None:
        return None, None
    ind = len(pat.match(lines[a]).group(1))
    b = next((i for i in range(a + 1, len(lines)) if lines[i].strip() and
              len(lines[i]) - len(lines[i].lstrip()) <= ind), len(lines))
    return a, b


_MAKER: dict = {}
_TOOLS_SAVEZ: list = []


def _maker(edges_file):
    """(url, line) of the `np.savez` that wrote `edges_file`, found in tools/*.py by the file name's leading word.
    Memoised per process (the answer depends only on the file name and the tools), and the scan reads only the tools that
    call np.savez, listed once."""
    if edges_file in _MAKER and _MAKER[edges_file][2] == _PUSHED.get("_sha"):
        return _MAKER[edges_file][:2]
    got_ = _maker_scan(edges_file)
    _MAKER[edges_file] = (*got_, _PUSHED.get("_sha"))
    return got_


def _maker_scan(edges_file):
    import glob as _g
    if not _TOOLS_SAVEZ:
        for f in sorted(_g.glob(os.path.join(ROOT, "tools", "*.py"))):
            try:
                b = open(f, errors="replace").read()
            except OSError:
                continue
            if "np.savez" in b:
                _TOOLS_SAVEZ.append((f, b))
    base = os.path.basename(edges_file or "")
    pre = (re.match(r"[A-Za-z_]+", base) or [None])[0]
    if not pre or len(pre) < 4:
        return "", ""
    suf = (re.search(r"(_[A-Za-z]+)$", os.path.splitext(base)[0]) or [None])[0]       # "_edges", "_win"
    best = None
    for f, body in _TOOLS_SAVEZ:
        # THE WORKING TREE PICKS THE CANDIDATES, git reads only them: a `git show` per tools/ file (~300) made the
        # first view of a pane take 100 s, and re-reading every file per relation 20 s
        if pre not in body and not (suf and f"{suf}.npz" in body):
            continue
        rel = os.path.relpath(f, ROOT)
        got = _pushed(rel)
        if not got:
            continue
        lines = got[1]
        sav = [i for i, ln in enumerate(lines) if "np.savez" in ln and not ln.lstrip().startswith("#")]
        if not sav:
            continue
        # the file NAME built in code: a string literal ending in .npz" that holds the leading word, or the trailing
        # word -- a docstring that merely mentions the file (`tissue/edges.npz`) does not count
        name_lines = [i for i, ln in enumerate(lines) if re.search(r'\.npz["\']', ln) and "load(" not in ln and
                      (pre in ln or (suf and f"{suf}.npz" in ln))]           # a file READ there is not written there
        if not name_lines:
            continue
        score = (2 if any(pre in lines[i] for i in name_lines) else 1)
        n = min(sav, key=lambda i: min(abs(i - j) for j in name_lines))
        if best is None or score > best[0]:
            best = (score, _url(got[0], rel, n + 1), lines[n])
    return (best[1], best[2]) if best else ("", "")


def bullet_links(s: dict, F: dict, expect: dict | None = None) -> dict:
    """{(kind, key): url} -- every link the plexus pane draws, by the rule above; `expect` (when given) is filled with
    {(kind, key): (the line the link opens, what that line must be)} for `check_links`."""
    import yaml
    out, exp = {}, (expect if expect is not None else {})
    raw_sets = {r["name"]: r for r in s.get("sets") or []}
    # THE MODEL SPEC IN THE REPO: a training config names it by repo path; a record's copy or a run folder's spec.yaml is
    # found under config/ by its name
    model_path = s.get("path")
    try:
        p0 = model_path if os.path.isabs(model_path) else os.path.join(ROOT, model_path)
        raw = yaml.safe_load(open(p0)) or {}
        m = (raw.get("model") or raw.get("spec")) if raw.get("learnable") else None
        if m:
            model_path = m if os.path.isabs(m) else os.path.join(ROOT, m)
    except Exception:                                            # noqa: BLE001
        pass
    if model_path and _spec_lines(model_path if os.path.isabs(model_path) else os.path.join(ROOT, model_path)) is None:
        # NOT ON GITHUB (outside the repo, or an ignored record copy under experiments/): its twin under config/
        import glob as _g
        stem = F.get("name") or ""
        hits = [h for h in _g.glob(os.path.join(ROOT, "config", "**", f"{stem}.yaml"), recursive=True)
                if "/training/" not in h] if stem else []
        if len(hits) > 1:                                 # the copy the record ran, byte for byte, else its body's twin
            try:
                mine = open(os.path.realpath(model_path if os.path.isabs(model_path)
                                             else os.path.join(ROOT, model_path))).read()
                same = [h for h in hits if open(h).read() == mine]
                hits = same or [h for h in hits if "/studio/" not in h] or hits
            except Exception:                                    # noqa: BLE001
                pass
        model_path = hits[0] if hits else model_path
    SP = _spec_lines(model_path if model_path and os.path.isabs(model_path) else
                     os.path.join(ROOT, model_path) if model_path else None)
    try:
        spec_raw = yaml.safe_load(open(model_path if os.path.isabs(model_path) else os.path.join(ROOT, model_path))) or {}
    except Exception:                                            # noqa: BLE001
        spec_raw = {}
    sets_raw = spec_raw.get("sets") or {}

    def spec_at(name, inner=None):
        """(url, line) of a set's `sets:` entry, or of `inner:` inside it."""
        if not SP:
            return "", ""
        sha, rel, lines = SP
        a, b = _spec_set_line(lines, name)
        if a is None:
            return "", ""
        n = a
        if inner:
            n = next((i for i in range(a, b) if re.match(rf"^\s+{re.escape(inner)}\s*:", lines[i])), a)
        return _url(sha, rel, n + 1), lines[n]

    # ENTITIES and SETS: the layout the run uses
    for r in F.get("sets") or []:
        nm = r["name"]
        if (sets_raw.get(nm) or {}).get("state"):
            u, ln = spec_at(nm, "state")
            out[("set", nm)], exp[("set", nm)] = u, (ln, r"^\s+state\s*:")
            continue
        c = _obj(r.get("cls")) or _entity_class((raw_sets.get(nm) or {}).get("entity") or nm)
        fn = getattr(c, "STATE_SCHEMA", None) if c is not None else None
        got = _src_of(fn) if callable(fn) and getattr(fn, "__name__", "") != "<lambda>" else None
        if got:
            sha, rel, lines = got
            n = next((i for i, ln in enumerate(lines) if re.match(rf"^def\s+{re.escape(fn.__name__)}\s*\(", ln)), None)
            if n is not None:
                out[("set", nm)], exp[("set", nm)] = _url(sha, rel, n + 1), (lines[n], r"^def\s+\w+\s*\(")
                continue
        u, ln = spec_at(nm)
        out[("set", nm)], exp[("set", nm)] = u, (ln, rf"^\s+{re.escape(nm)}\s*:")
    # RELATIONS: the spec line naming the edges, and the tool that wrote them
    rels = list(F.get("relations") or []) or [dict(name=r["name"]) for r in s.get("sets") or [] if r.get("relation")]
    for r in rels:
        nm = r["name"]
        u, ln = spec_at(nm)
        out[("relation", nm)], exp[("relation", nm)] = u, (ln, rf"^\s+{re.escape(nm)}\s*:")
        ef = r.get("edges_file") or (sets_raw.get(nm) or {}).get("edges_file")
        if ef:
            u2, ln2 = _maker(ef)
            out[("relation_maker", nm)], exp[("relation_maker", nm)] = u2, (ln2, r"np\.savez")
    # GNN STEPS: the forward that runs it, and the variant's own hook
    cls_of = {}
    for st in F.get("steps") or []:
        c = step_class(st)
        cls_of[st["id"]] = c
        u, ln = method_url(c, "forward") if c is not None else ("", "")
        out[("step", st["id"])], exp[("step", st["id"])] = u, (ln, r"^\s+def\s+forward\s*\(")
        hk = _hook(c)
        if hk:
            u2, ln2 = method_url(c, hk)
            out[("step_hook", st["id"])], exp[("step_hook", st["id"])] = u2, (ln2, rf"^\s+def\s+{hk}\s*\(")
    # LEARNABLES: where the reading step's forward reads the value
    T = F.get("training") or {}
    for e in T.get("learnables") or []:
        on, code = e.get("on") or "", e.get("code") or ""
        reader, param = None, None
        for st in F.get("steps") or []:
            if str(st.get("when", "")).startswith("once"):      # a seed sets the start value; the model reads it elsewhere
                continue
            if not on and code in [str(v) for v in (st.get("params") or {}).values()]:   # a field, named by a param
                reader, param = st, next(k for k, v in (st.get("params") or {}).items() if str(v) == code)
                break
            acts = [a.split("[")[0] for a in st.get("acts_on") or []]
            pv = st.get("params") or {}
            words = lambda v: re.findall(r"\w+", str(v))             # noqa: E731  ("one of junction_lr, junction_mr")
            if e.get("kind") and st.get("code", "").split("[")[0] == on:          # {param:, op:}: that operator's param
                reader, param = st, code
                break
            if on in words(pv.get("edge_set")) and code in ("w", pv.get("weight", "w")):  # an edge set's weight
                reader, param = st, "weight"
                break
            if on in acts and any(code in words(v) for v in pv.values()):           # a block named by a param
                reader, param = st, next(k for k, v in pv.items() if code in words(v))
                break
        c = cls_of.get(reader["id"]) if reader else None
        att = _attr_for_param(c, param) if c is not None and param else None
        toks = ([f"self.{att}"] if att else []) + ([f"self.{code}", f"self._{code}", f'"{code}"'] if e.get("kind") else [])
        if e.get("kind") and "_" in code:                 # a family of attributes read by name: getattr(self, f"W_{s}")
            toks.append('f"' + code[: code.rindex("_") + 1] + "{")
        u, ln = token_url(c, toks) if (c is not None and toks) else ("", "")
        if not u:
            # A BLOCK THE OPERATOR READS BY ITS OWN NAME (no param names it): the non-seed steps on that set, searched
            # for `"<block>"` -- e.g. a `tau` or `rest` block read inside the operator
            for st in F.get("steps") or []:
                if str(st.get("when", "")).startswith("once") or on not in [a.split("[")[0] for a in st.get("acts_on") or []]:
                    continue
                u, ln = token_url(cls_of.get(st["id"]), [f'"{code}"', f"'{code}'"])
                if u:
                    break
        out[("learnable", e["id"])] = u
        fam = re.escape('f"' + code[: code.rindex("_") + 1] + "{") if "_" in code else "$^"
        exp[("learnable", e["id"])] = (ln, (re.escape(code) if not att else rf"self\.{re.escape(att)}|{re.escape(code)}")
                                       + f"|{fam}")
    return out


def _op_class(name: str, variant: str | None):
    from plexus.models import registry as R
    try:
        return R.get_operator(name, variant=variant)
    except Exception:                                            # noqa: BLE001
        return R._OPERATOR_REGISTRY.get(name)


def _entity_class(name: str | None):
    from plexus.models import registry as R
    return R._ENTITY_REGISTRY.get(name) if name else None


def _flat(d, prefix: str = "") -> dict:
    """A spec flattened to `path -> value`; an operator list is keyed by each operator's identity
    (`operators[bm_bond[fem_membrane]@rubber_pt]`), so reordering operators is not a difference."""
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(_flat(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(d, list) and d and all(isinstance(x, dict) for x in d):
        seen = {}
        for i, o in enumerate(d):
            # A LIST ENTRY IS NAMED BY WHAT IT IS (an operator, a learnable parameter or block, a loss term),
            # so reordering entries is not a difference; anything else by its position (a training stage).
            if "op" in o and not ("param" in o or "block" in o):
                ident = o["op"] + (f"[{o['model']}]" if o.get("model") else "") + (f"@{o['at']}" if o.get("at") else "")
                drop = ("op", "model", "at")
            elif "param" in o:
                ident, drop = f"learnable {o['param']}@{o.get('op', '')}", ("param",)
            elif "block" in o:
                ident, drop = f"learnable {o['block']}@{o.get('of', '')}", ("block",)
            elif "field" in o:
                ident, drop = f"learnable field {o['field']}", ("field",)
            elif "term" in o:
                ident, drop = f"term {o['term']}", ("term",)
            else:
                ident, drop = f"#{i}", ()
            seen[ident] = seen.get(ident, 0) + 1
            ident += f"#{seen[ident]}" if seen[ident] > 1 else ""
            out[f"{prefix}[{ident}]"] = "__present__"
            out.update(_flat({k: v for k, v in o.items() if k not in drop}, f"{prefix}[{ident}]"))
    else:
        out[prefix] = _fmt(d) if isinstance(d, (list, tuple, float)) else d
    return out


_DIFF_SKIP = re.compile(r"^(general\.name|general\.title|descriptions|plotting)(\.|$)")


def _physics(path: str) -> dict:
    """A spec's content for comparison; a TRAINING spec also carries the model spec it trains, under
    `model_spec.`, so a change made inside the model is a difference of the training run too."""
    import yaml
    raw = yaml.safe_load(open(path)) or {}
    out = {k: v for k, v in _flat(raw).items() if not _DIFF_SKIP.match(k)}
    if raw.get("learnable") and (raw.get("model") or raw.get("spec")):
        try:
            mp = _summary()._model_path(os.path.realpath(path), raw)
            mraw = yaml.safe_load(open(mp)) or {}
            out.update({f"model_spec.{k}": v for k, v in _flat(mraw).items() if not _DIFF_SKIP.match(k)})
            out.pop("model", None); out.pop("spec", None)
        except Exception:                                        # noqa: BLE001
            pass
    return out


def spec_diff(prev: str | None, cur: str, max_lines: int = 14) -> list[str]:
    """What changed from the previous run's spec to this one, as `path: old -> new` lines -- an added or
    removed operator once, not each of its parameters. Rendering-only keys (`plotting`) are left out."""
    if not prev or not os.path.exists(prev):
        return []
    a, b = _physics(prev), _physics(cur)
    gone_ops = [k for k in a if a[k] == "__present__" and k not in b]
    new_ops = [k for k in b if b[k] == "__present__" and k not in a]
    under = lambda k, roots: any(k.startswith(r + ".") for r in roots)          # noqa: E731
    out = [f"operator added: {k}" for k in new_ops] + [f"operator removed: {k}" for k in gone_ops]
    for k in sorted(set(a) | set(b)):
        if k in new_ops or k in gone_ops or under(k, new_ops) or under(k, gone_ops):
            continue
        if a.get(k) != b.get(k):
            out.append(f"{k}: {a.get(k, '(none)')} -> {b.get(k, '(none)')}")
    return out[:max_lines] + ([f"... and {len(out) - max_lines} more"] if len(out) > max_lines else [])


def previous_spec(png: str | None) -> str | None:
    """The spec of the previous RUN in the step's record: the nearest earlier step whose spec differs
    from this one in anything but its name (a run's before/after shots carry the same spec)."""
    if not png:
        return None
    import glob
    rec = os.path.dirname(os.path.dirname(png))
    me = os.path.basename(png)
    cur = os.path.join(rec, "spec", me[:4] + ".yaml")
    if not os.path.exists(cur):
        return None
    try:
        mine = _physics(cur)
    except Exception:                                            # noqa: BLE001
        return None
    pngs = sorted(glob.glob(os.path.join(rec, "png", "[0-9][0-9][0-9][0-9].png")))
    for q in reversed([x for x in pngs if os.path.basename(x) < me]):
        sp = os.path.join(rec, "spec", os.path.basename(q)[:4] + ".yaml")
        if not os.path.exists(sp):
            continue
        try:
            other = _physics(sp)
        except Exception:                                        # noqa: BLE001
            continue
        if other and other != mine:
            return sp
    return None


def facts(s: dict, raw: dict | None = None, context: str = "", diff: list | None = None) -> dict:
    """The fact sheet, from `spec_summary.summarise(path)`: header, sets, and the STEPS -- one per
    operator, merged where the same operator does the same thing at the same moment (ten drains on
    ten faces are one step "x10"), in the order the run executes them."""
    h = s["header"]
    import yaml
    train_raw = None
    if raw is None:
        p0 = s["path"] if os.path.isabs(s["path"]) else os.path.join(ROOT, s["path"])
        raw = yaml.safe_load(open(p0)) or {}
        if s.get("training"):                                   # the sets are the MODEL's
            train_raw = raw
            raw = yaml.safe_load(open(_summary()._model_path(os.path.realpath(p0), raw))) or {}
    rs = raw.get("sets") or {}

    def _types(name):
        """Each type's material facts (what it is made of and its shape), from the spec itself."""
        out = []
        for tn, t in ((rs.get(name) or {}).get("types") or {}).items():
            t = t or {}
            keep = {k: _fmt(v) for k, v in t.items() if k in (
                "material", "density", "youngs", "bulk_modulus", "shape", "hollow", "aspect", "axis",
                "radius", "fraction", "count", "block", "p", "sign", "role")}
            out.append(f"{tn} {keep}" if keep else tn)
        return out
    def _roles(name):
        """Each role's total and its types, from the type table: 'afferent 41 (AF5_ipsi_L 15, ...)'."""
        by = {}
        for tn, t in ((rs.get(name) or {}).get("types") or {}).items():
            t = t or {}
            if t.get("role") and t.get("count") is not None:
                by.setdefault(str(t["role"]), []).append((tn, int(t["count"])))
        return [f"{role} {sum(c for _, c in ts)} ({', '.join(f'{tn} {c}' for tn, c in ts)})" for role, ts in by.items()]
    sets = [dict(name=r["name"], title=r.get("title") or "", n=r.get("seeded") or r.get("n"),
                 parent=r.get("parent") or "", per_parent=r.get("per_parent"),
                 buffer=(rs.get(r["name"]) or {}).get("buffer"),
                 start=_fmt((rs.get(r["name"]) or {}).get("start")) if (rs.get(r["name"]) or {}).get("start") else "",
                 types=_types(r["name"]), roles=_roles(r["name"]), state=r.get("state_labels") or [],
                 cls=class_ref(_entity_class(r.get("entity") or r["name"])))
            for r in s["sets"] if not r.get("relation")]
    # THE RELATIONS, as the spec states them: what they join, how many edges, and -- read from the edge file and the
    # joined sets' ordered type tables -- which TYPES the edges leave from and land on. Which population is a circuit's
    # input and which its output is for the agent to say from these facts; nothing here names them.
    relations = [dict(name=r["name"], pre=r["relation"][0], post=r["relation"][1], edges=r.get("n"),
                      edges_file=r.get("edges_file") or "", **_edge_types(rs, r))
                 for r in s["sets"] if r.get("relation")]
    fields = [dict(name=f["name"], grid=f.get("n_grid")) for f in s.get("fields") or []]
    steps, by_key = [], {}
    for o in s["operators"]:
        if o.get("probe"):
            continue
        when = ("once, at the start" if o["phase"] == "seed" else
                "every substep" if o.get("substep_dt") else
                "never (declared, not scheduled)" if o["phase"] == "unscheduled" else "every frame")
        targets = [x for x in (o.get("at"), o.get("from")) if isinstance(x, str)]
        # A SEED IS NEVER MERGED: each one builds a different body, and its geometry is the fact
        # that says what the body is (a 4 cm ball, a 4.1 cm shell around it).
        key = (when, o["name"], o.get("impl") or "", o["title"]) + ((tuple(targets),) if o["phase"] == "seed" else ())
        pv = {p["name"]: _fmt(p["value"]) for p in o["params"] if p["name"] not in ("_at",)}
        roles = {p["name"]: p.get("role", "") for p in o["params"] if p.get("role")}
        if key in by_key:
            st = by_key[key]
            st["count"] += 1
            st["acts_on"] += [t for t in targets if t not in st["acts_on"]]
            for k, v in pv.items():                               # the values that DIFFER, listed
                vals = st["_vals"].setdefault(k, [st["params"].get(k)])
                if v not in vals:
                    vals.append(v)
            continue
        _cls = _op_class(o["name"], o.get("impl"))
        # THE AUTHOR'S LEGEND AND UNITS FIRST (the curation of 2026-10-27, docs/CURATION_PLAN.md step 6b): a class may carry `SYMBOLS` ({"\\kappa": "tether stiffness ..."}),
        # the legend under its equation, and `PARAM_UNITS` ({"k": "acceleration", ...}, one vocabulary); the `claude -p` legend below is the fallback for a class without SYMBOLS.
        _sym = getattr(_cls, "SYMBOLS", None) if _cls is not None else None
        _units = getattr(_cls, "PARAM_UNITS", None) if _cls is not None else None
        st = dict(id=len(steps), when=when,
                  code=o["name"] + (f"[{o['impl']}]" if o.get("impl") and o.get("axis") == "model" else ""),
                  kind=o.get("kind") or "lateral", title=o["title"], gloss=o.get("gloss") or "",
                  acts_on=targets, count=1,
                  params=pv, roles=roles, _vals={}, cls=class_ref(_cls),
                  units={k: str(v) for k, v in _units.items()} if isinstance(_units, dict) else {},
                  symbols_author=[[str(k), str(v)] for k, v in _sym.items()] if isinstance(_sym, dict) and _sym else None,
                  equation=o.get("equation") or "",
                  substep_dt=o.get("substep_dt"))
        by_key[key] = st
        steps.append(st)
    for st in steps:
        for k, vals in st.pop("_vals").items():
            if len(vals) > 1:
                st["params"][k] = "one of " + ", ".join(str(v) for v in vals)
        for k, v in list(st["params"].items()):               # a list of box faces, said in words
            fs = re.findall(r"[+-][xyz]\b", str(v))
            if k in ("faces", "face") and fs:
                miss = [f for f in ("+x", "-x", "+y", "-y", "+z", "-z") if f not in str(v)]
                st["params"][k] = (f"{v} -- {len(set(fs))} of the 6 faces" +
                                   (f"; NOT {', '.join(miss)}" if miss else "") +
                                   ("; -y is the floor" if "-y" in miss else ""))
    sub = sorted({st["substep_dt"] for st in steps if st["substep_dt"]})
    per_frame = None
    if sub:
        dts = [float(d) for f, d in (h.get("dt_schedule") or [])] or [float(h["dt"])]
        n = sorted({max(1, round(d / float(sub[0]))) for d in dts})
        per_frame = f"{n[0]}" if len(n) == 1 else f"{n[0]}-{n[-1]}"
    units = h.get("units") or {}
    L = float(units.get("length_um", 0) or 0) * 1e-6           # metres per world unit, when declared
    return dict(
        headline_fixed=str(((raw.get("general") or {}).get("headline") or "")).strip(),            # `general.headline` (Cedric, 2026-10-28): the pane's headline, verbatim, instead of the agent's
        name=h["name"], title=h.get("title") or "", dim=h["dim"],
        box=(_fmt([w * L for w in h["world"]]) + " m") if L else _fmt(h["world"]) + " (no units declared)",
        simulated_time=(f"{h['seconds']:.4g} s" if h.get("seconds") else None),
        frames=h["n_frames"], substep=(f"{float(sub[0]):.3g} time units, {per_frame} per frame" if sub else None),
        units=(", ".join(x for x in (f"1 length unit = {units['length_um']:g} um" if units.get("length_um") else "",
                                     f"1 time unit = {units['time_s']:g} s" if units.get("time_s") else "") if x)
               or None),
        sets=sets, fields=fields, steps=steps, context=(context or "").strip()[:500], diff=list(diff or []),
        notation=_read_notation(),
        plotting={k: v for k, v in (s.get("plotting") or {}).items()
                  if k in ("renderer", "render_3d", "subject", "also_sets", "compartment_sets", "colors",
                           "also_colors", "camera", "background", "floor", "duration_s", "dot_size")},
        relations=relations,
        training=_training_facts(s, train_raw, sets, steps) if s.get("training") else None)


def _training_facts(s: dict, train_raw: dict | None, sets: list, steps: list) -> dict:
    """What a TRAINING spec adds to its model: the learnables (what each is, its representation, learning
    rate and prior, and the source of the operator or entity it belongs to), the data it is fitted to, the
    loss and the schedule -- read from `spec_summary.read_training` and the spec itself."""
    t = s["training"]
    raws = list((train_raw or {}).get("learnable") or [])
    by_set = {r["name"]: r for r in sets}
    out = []
    for i, e in enumerate(t["learnable"]):
        r = raws[i] if i < len(raws) else {}
        kind, cref, on = None, "", ""
        if r.get("param") and r.get("op"):
            st = next((x for x in steps if x["code"].split("[")[0] == r["op"]), None)
            kind = st["kind"] if st else None
            cref = st["cls"] if st else class_ref(_op_class(r["op"], None))
            on = r["op"]
        elif r.get("block") and r.get("of"):
            cref = (by_set.get(r["of"]) or {}).get("cls", "")
            on = r["of"]
        out.append(dict(id=f"L{i}", what=e["what"], how=e["how"], extra=e["extra"], role=e["role"],
                        kind=kind, cls=cref, on=on,
                        code=(r.get("param") or r.get("block") or r.get("field") or "")))
    return dict(name=t["name"], learnables=out, reference=t["reference"], observe=t["observe"],
                drive=t["drive"], loss=t["loss"], scheme=t["scheme"])


def _edge_types(rs: dict, r: dict) -> dict:
    """{pre_types, post_types}: how many of a relation's edges leave each type of its `pre` set and land on each type
    of its `post` set, from the edge file and the sets' ORDERED type tables (`type_layout: ordered`, each type a
    `count`, repeated per parent). Empty when the file or the layout does not say."""
    out = {"pre_types": {}, "post_types": {}}
    ef = r.get("edges_file")
    if not ef:
        return out
    try:
        import numpy as np
        from plexus.paths import graphs_data_path
        ei = np.load(graphs_data_path(ef))["edge_index"]
    except Exception:                                            # noqa: BLE001
        return out

    def hist(set_name, idx):
        sd = rs.get(set_name) or {}
        types = sd.get("types") or {}
        if sd.get("type_layout") != "ordered" or not types or not all("count" in (t or {}) for t in types.values()):
            return {}
        names, counts = list(types), [int(types[t]["count"]) for t in types]
        per = sum(counts)
        edges = np.cumsum(counts)
        pos = np.asarray(idx) % per if per else np.asarray(idx)
        which = np.searchsorted(edges, pos, side="right")
        h = {}
        for k in which:
            if 0 <= k < len(names):
                h[names[k]] = h.get(names[k], 0) + 1
        return h
    out["pre_types"] = hist(r["relation"][0], ei[0]) if r["relation"][0] else {}
    out["post_types"] = hist(r["relation"][1], ei[1]) if r["relation"][1] else {}
    return out


def _relations_text(F: dict) -> list:
    """The relations as lines of the fact sheet."""
    out = []
    for r in F.get("relations") or []:
        ln = f"- {r['name']}: relation {r['pre']} -> {r['post']}" + (f", {r['edges']:,} edges" if r.get("edges") else "")
        ln += f" (from {r['edges_file']})" if r.get("edges_file") else ""
        if r.get("pre_types"):
            ln += "; leaves from types " + ", ".join(f"{k} {v}" for k, v in r["pre_types"].items())
        if r.get("post_types"):
            ln += "; lands on types " + ", ".join(f"{k} {v}" for k, v in r["post_types"].items())
        out.append(ln)
    return out


def _read_notation() -> str:
    """THE REFERENCE PAPER'S FORMALISM, when the record carries one (Cedric, 2026-11-04, exp03: "make sure that the
    plexus pane and the math pane follow the formalism -- variable names, equations -- of the pdf"). A record folder
    (`PLEXUS_BUILDER`, the folder the watcher is pointed at) may hold `notation.md`: the paper's symbols, each with its
    meaning and the block or parameter that carries it in the spec, and the paper's equations verbatim. When it is
    there, it is part of the fact sheet -- so it is part of the cache key -- and the three prompts below are told to
    write in it. A record without the file is unchanged: no key, no prompt, no cache entry moves."""
    d = os.environ.get("PLEXUS_BUILDER") or ""
    f = os.path.join(d, "notation.md") if d else ""
    return open(f).read().strip() if f and os.path.isfile(f) else ""


def _notation_lines(F: dict) -> list:
    n = F.get("notation") or ""
    if not n:
        return []
    return ["", "NOTATION OF THE REFERENCE PAPER -- AUTHORITATIVE. This run reproduces that paper. Wherever a quantity "
            "below is in the notation table, use the PAPER's symbol for it (and name the spec's block or parameter that "
            "carries it), and wherever a step's law is one of the paper's equations, write THE PAPER'S equation, with "
            "the paper's letters, instead of the registry's. A letter of the registry that clashes with a paper letter "
            "(the paper's phi is the fibre angle) is renamed to the paper's meaning. Do not invent a symbol the paper "
            "does not use:", n, ""]


def _facts_text(F: dict) -> str:
    """The fact sheet as compact text for the prompt (the equations are left out: they cost tokens and
    the agent is asked for words, not maths)."""
    if F.get("training"):
        lines = [f"MODEL SPEC {F['name']}, {F['dim']}D" + (f" -- titled '{F['title']}'" if F["title"] else "")]
    else:
        lines = [f"SPEC {F['name']}" + (f" -- titled '{F['title']}'" if F["title"] else ""),
                 f"{F['dim']}D simulation box {F['box']} (the computed region, not a container); simulated time "
                 f"{F['simulated_time']}; {F['frames']} frames" + (f"; substep {F['substep']}" if F["substep"] else "")]
        if F.get("units"):
            lines.append(f"UNITS: every length and time in the params below is in world units -- {F['units']}. "
                         f"Convert before quoting a size or a duration.")
    if any(st["code"] == "gravity" for st in F["steps"]):
        lines.append("Gravity points along -y: y is UP, the -y face of the box is the floor, +y the ceiling.")
    if F.get("context"):
        lines += ["", "CONTEXT -- the developer's working note for this run. It may describe changes, fixes, "
                      "earlier runs and hopes; NONE of that goes in your text. Use it only to recognise what the "
                      "scene is meant to depict (e.g. that the bodies are a balloon and a bullet), and only where the "
                      "FACTS agree:", F["context"]]
    lines += _notation_lines(F)
    if F.get("diff"):
        lines += ["", "DIFF -- what changed from the PREVIOUS run's spec to this one (path: old -> new):"] + \
                 [f"  {d}" for d in F["diff"]]
    T = F.get("training")
    if T:
        lines += ["", f"THIS IS A TRAINING RUN ({T['name']}): the MODEL below (its SETS and STEPS) is fitted to data.",
                  f"data: {T['reference']}", f"driven by: {T['drive'] or '-'}", f"observed: {T['observe'] or '-'}",
                  f"loss: {T['loss']}", f"schedule: {T['scheme']}", "LEARNABLES (id, what, representation, settings, role):"]
        for e in T["learnables"]:
            lines.append(f"[{e['id']}] {e['what']} | {e['how']} | {e['extra'] or '-'} | {e['role']}")
        lines += ["", "THE MODEL:"]
    lines += ["", "SETS (what exists; a set of 1 holding many points is one body made of those points):"]
    for r in F["sets"]:
        _pb = next((q.get("buffer") for q in F["sets"] if q["name"] == r["parent"]), None) if r["parent"] else None
        if isinstance(r.get("per_parent"), dict):     # points per PARENT TYPE
            head = (f"- {r['name']}: {r['n'] if r['n'] is not None else '?'} points"
                    f" ({', '.join(f'{v} per {k} cell' for k, v in r['per_parent'].items())})")
        elif _pb:                                 # a parent with free slots: the allocation is the count a seed can wake
            head = (f"- {r['name']}: {int(r['per_parent']) * int(_pb):,} points allocated ({r['per_parent']} per "
                    f"{r['parent']} slot x {_pb} slots; the seeds below say how many are live at the start)")
        else:
            head = f"- {r['name']}: {r['n']}" + (f" ({r['per_parent']} per {r['parent']})" if r["parent"] else "")
        lines.append(head + (f" live at the start (up to {r['buffer']} slots)" if r.get("buffer") and not r["parent"] else "")
                     + (f"; at {r['start']}" if r["start"] else "")
                     + (f"; types {r['types']}" if r["types"] else "")
                     + (f"; BY ROLE (counted from the type table): {'; '.join(r['roles'])}" if r.get("roles") else "")
                     + (f"; state {r['state']}" if r["state"] else ""))
    for f in F["fields"]:
        lines.append(f"- {f['name']}: a background grid, {f['grid']} cells across")
    rel = _relations_text(F)
    if rel:
        lines += ["", "RELATIONS (what joins what; a step's `edge_set:` names one of these):"] + rel
    lines += ["", "STEPS (what the run does, in execution order; id, when, operator, acts on, x count):"]
    for st in F["steps"]:
        lines.append(f"[{st['id']}] {st['when']} | {st['code']} ({st['title']}) on {', '.join(st['acts_on']) or '-'}"
                     + (f" x{st['count']}" if st["count"] > 1 else ""))
        if st["gloss"]:
            lines.append(f"     does: {st['gloss'][:300]}")
        if st["params"]:
            ro, un = st.get("roles") or {}, st.get("units") or {}
            lines.append("     params: " + ", ".join(f"{k}={v}" + (f" [{(ro.get(k) or '').replace('_', ' ')}{', ' if ro.get(k) and un.get(k) else ''}{(un.get(k) or '').replace('_', ' ')}]" if ro.get(k) or un.get(k) else "")
                                                for k, v in list(st["params"].items())[:10]))
        if st.get("equation"):
            eq = re.sub(r"\s+", " ", st["equation"].strip().strip("$"))[:400]
            lines.append(f"     equation: {eq}")
    if F.get("plotting"):
        lines += ["", "RENDERING (the movie's `plotting:` block): " + json.dumps(F["plotting"], default=str)[:600]]
    return "\n".join(lines)


ROLE = """You write the PLEXUS pane of a simulation watcher. Its reader is a scientist watching the run,
NOT the developer: they know physics and biology, not this code base. You are given the facts of one
simulation spec. Output ONLY one JSON object, no prose around it, no code fences:

{"headline": "...", "story": "...", "change": "...", "entities": {"<set or grid name>": "..."},
 "order": [ids], "steps": {"<id>": {"name": "...", "what": "..."}}}

- headline: what the scene is, at most 12 words, plain words (e.g. "A water balloon shot by a bullet").
- story: 3 to 5 short sentences that walk the steps IN EXACTLY THE ORDER THEY ARE GIVEN (the order the run takes them), so the
  reader meets the same sequence in the story and then in the lists of the panes: first what the scene contains (sizes
  and counts from the facts, in units), then what each step does in turn; the bookkeeping steps at the end
  of the list (moving data to and from the grid, boundaries, removal) share one closing clause. End with
  how long is simulated.
- change: ONE short sentence (at most 20 words) naming the main difference from the previous run, read
  from DIFF only, in plain words ("The bullet now flies at 50 m/s instead of 250 m/s."). Empty string when
  there is no DIFF.
- steps: for EVERY step id given, a "name" (at most 6 words, starting with a LOWERCASE letter unless the
  first word is a proper name or an acronym, self-explanatory, no code words, no underscores, no
  abbreviations a scientist would not know) and a "what" (AT MOST 12 WORDS, one line, no full stop: what
  this step does to which part of the scene).
- symbols: for EVERY step that has an "equation", the list of its symbols, each as ["symbol", "meaning"]:
  the symbol written as it APPEARS in the equation, in LaTeX, since it is typeset (\\mu, t_0, F, C, \\det C, W_{ij},
  det C); the meaning in at most 12 words, plain English, with the value and unit when the step's params
  give it ("shear modulus of the rubber, 0.4 MPa"). Every letter and constant of the equation, once.
  Key: "symbols": {"<id>": [["μ", "..."], ...]}.
- rendering: ONE sentence (at most 25 words) saying how the movie draws the scene, from RENDERING: what
  appears as what (dots, surfaces), the colours in words, the viewpoint, the movie's length.
- entities: for EVERY set and grid named in the facts, a plain name of at most 4 words ("the water",
  "the rubber skin", "the water's grid"). For a CIRCUIT (a set of typed neurons with relations onto it and out of
  it) work out from the RELATIONS and the types' roles which types are its INPUT neurons (where the incoming
  relation lands), which its recurrent core, and which its OUTPUT neurons (what the readout relations leave from,
  and what each drives), with their counts -- and say so in the story and in the entities' names ("the 41 AF5
  input cells", "the 92 AMN cells driving the lateral rectus"). The facts give the data; the reading is yours.
  Every count is taken from the type table (BY ROLE) or the relations, never estimated.
- order: every step id exactly once, IN THE ORDER GIVEN. The panes list the steps in the order the run executes them (the
  seeds, then each frame's schedule); the order is not yours to choose, and the story follows it.
- Faces are written +x/-x/+y/-y/+z/-z; a face that is not listed is not affected.
- Say what the scene IS from its geometry and materials (a thin pre-stretched rubber shell around a ball of
  water is a water balloon). The box is the computed region, never a tank or container.
- Describe ONLY the simulated scene as the facts define it (only `change` compares with the previous
  run). Never mention versions, fixes, diagnostics, or what "should" happen. If a body in the CONTEXT is
  not in the SETS, it is not in the scene.
- Use only facts given. Do not invent mechanisms, numbers or results; a step acts only on what its facts
  name. No hedging, no marketing words."""


ROLE_TRAIN = (ROLE.replace(
    "You write the PLEXUS pane of a simulation watcher.",
    "You write the PLEXUS pane of a watcher, for a TRAINING run: a model (a graph network of states and steps) "
    "whose learnable parameters are fitted to data.")
    .replace('"order": [ids], "steps"', '"learnables": {"<L id>": {"name": "...", "what": "..."}}, "order": [ids], "steps"')
    + "\n- THIS IS A TRAINING RUN. headline: what is learned from what (e.g. \"learning a tissue's growth law "
      "from its recorded cell counts\"). story (3 to 5 sentences): what the data is, what the model is (its states and "
      "its loop, briefly, in the order of `order`), what is learnable, and how it is fitted (loss, schedule length). "
      "learnables: for EVERY L id, a \"name\" (at most 6 words, lowercase start) and a \"what\" (at most 12 words, one "
      "line) saying what that parameter means in the model. change: from DIFF, as above (a `model_spec.` path is a "
      "change inside the model being trained). The SETS and STEPS are the MODEL, not a simulated scene.")


def _key(F: dict) -> str:
    blob = json.dumps({"v": PROMPT_VERSION, "model": MODEL, "f": _facts_text(F)}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def _norm(st: dict) -> dict:
    """JSON turns the step ids (dict keys) into strings; the facts use ints."""
    st["steps"] = {int(k): v for k, v in (st.get("steps") or {}).items()}
    st["symbols"] = {int(k): v for k, v in (st.get("symbols") or {}).items()}
    st["order"] = [int(i) for i in st.get("order") or []]
    return st


def cached(F: dict) -> dict | None:
    p = os.path.join(CACHE, _key(F) + ".json")
    if os.path.exists(p):
        try:
            return _norm(json.load(open(p)))
        except Exception:                                        # noqa: BLE001
            return None
    return None


def _parse(text: str, F: dict) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object in the answer")
    j = json.loads(m.group(0))
    ids = [st["id"] for st in F["steps"]]
    steps = {int(k): v for k, v in (j.get("steps") or {}).items() if str(k).lstrip("-").isdigit()}
    order = [int(i) for i in (j.get("order") or []) if isinstance(i, (int, str)) and str(i).isdigit()]
    order = list(ids)                                           # the run's own order, whatever the agent wrote: it is not the agent's to choose
    ent = {str(k): str(v).strip() for k, v in (j.get("entities") or {}).items() if str(v).strip()}
    lrn = {str(k): dict(name=str((v or {}).get("name") or "").strip(), what=str((v or {}).get("what") or "").strip())
           for k, v in (j.get("learnables") or {}).items() if isinstance(v, dict)}
    sym = {}
    for k, v in (j.get("symbols") or {}).items():
        if str(k).lstrip("-").isdigit() and isinstance(v, list):
            sym[int(k)] = [[str(a[0]), str(a[1])] for a in v if isinstance(a, (list, tuple)) and len(a) >= 2]
    return dict(headline=(F.get("headline_fixed") or str(j.get("headline") or "").strip()), story=str(j.get("story") or "").strip(),
                change=str(j.get("change") or "").strip() if F.get("diff") else "",
                entities=ent, learnables=lrn, symbols=sym, rendering=str(j.get("rendering") or "").strip(),
                order=order, steps={i: dict(name=str((steps.get(i) or {}).get("name") or "").strip(),
                                            what=str((steps.get(i) or {}).get("what") or "").strip())
                                    for i in ids})


def missing_symbols(F: dict, st: dict | None) -> list:
    """The ids of the steps that have an equation and no symbol legend in `st` -- the pane's rule is that
    every equation comes with every one of its variables and constants defined."""
    if st is None:
        return []
    sym = st.get("symbols") or {}
    return [x["id"] for x in F["steps"] if (x.get("equation") or "").strip()
            and not (sym.get(x["id"]) or sym.get(str(x["id"])))]


ROLE_SYMBOLS = """You complete the PLEXUS pane of a simulation watcher: the legend under each equation. Its reader is a
scientist, not the developer of this code. You are given the steps of one run that have an equation, with the
step's parameters and what each parameter is. Output ONLY one JSON object, no prose around it, no code fences:

{"symbols": {"<id>": [["symbol", "meaning"], ...], ...}}

- For EVERY step id given, list EVERY letter, index and constant that appears in its equation, once each: the
  symbol written as it APPEARS in the equation, in LaTeX, since it is typeset (\\mu, t_0, F, C, \\det C, W_{ij}, det C;
  \\frac, \\sum and other LaTeX commands are not symbols); the meaning in at most 12 words of plain English,
  with the value and unit when the step's params give it ("shear modulus of the rubber, 0.4 MPa"; "time step,
  1/60 s"). Indices (i, j, e) and the sum's range are symbols too. A symbol that names a block of the entity
  ("the voltage of neuron i") says so.
- A symbol whose value is 0 for every type and parameter of this run says so ("0 in this run: the term is
  absent"); a factor that is 1 because nothing is named says so.
- Use only the facts given; do not invent values. No hedging."""


def _doc_of(code: str, n: int = 700) -> str:
    """What the operator's docstring says about its symbols: the registered class's docstring and, for a `model:`
    variant, the base operator's as well (that is where g_i, Omega_i, W_e are defined for every neuron_signal model).
    Prompt-time only; never stored in the facts."""
    try:
        import plexus.operators                                  # noqa: F401  self-registers
        name, _, var = code.partition("[")
        cls = _op_class(name, var.rstrip("]") or None)
        if cls is None:
            return ""
        docs = []
        for c in (cls,) + tuple(b for b in cls.__mro__[1:3] if (b.__doc__ or "").strip()):
            d = re.sub(r"[ \t]+", " ", (c.__doc__ or "").strip())
            if d and d not in docs:
                docs.append(d[:n])
        return "\n       ".join(docs)
    except Exception:                                            # noqa: BLE001
        return ""


def _sets_text(F: dict) -> list:
    """The SETS with their types' material facts and parameter rows -- what decides which terms of a general law
    are alive in this run (a type row `p = [a, b, g, s, w, h]`: a decay, b offset, g gain, s self-coupling, w width,
    h threshold, for the neuron operators)."""
    out = ["", "SETS (what exists; a type's `p` is its parameter row, a 0 in it switches that term off):"]
    for r in F.get("sets") or []:
        head = f"- {r['name']}: {r.get('n')}" + (f" ({r.get('per_parent')} per {r['parent']})" if r.get("parent") else "")
        out.append(head + (f"; types {r['types']}" if r.get("types") else "")
                   + (f"; BY ROLE: {'; '.join(r['roles'])}" if r.get("roles") else "")
                   + (f"; state {r['state']}" if r.get("state") else ""))
    rel = _relations_text(F)
    if rel:
        out += ["RELATIONS (what joins what):"] + rel
    return out


def _alive_text(F: dict, st: dict) -> str:
    """What this run switches OFF in a step's general law, read by the script, not guessed by the agent: for an
    operator whose module names the type row's slots (`P_NAMES`, the neural family: p = [a, b, g, s, w, h]),
    each slot's value over the acted-on set's types -- a slot that is 0 for every type kills every term it
    multiplies; a step parameter that is 0 (`noise`) likewise; a `field` not named leaves its factor at 1."""
    import ast
    try:
        import plexus.operators                                  # noqa: F401
        name, _, var = st["code"].partition("[")
        cls = _op_class(name, var.rstrip("]") or None)
        names = getattr(sys.modules.get(getattr(cls, "__module__", "")), "P_NAMES", None) if cls else None
    except Exception:                                            # noqa: BLE001
        names = None
    out = []
    if names:
        for sname in st.get("acts_on") or []:
            r = next((q for q in F.get("sets") or [] if q["name"] == sname), None)
            rows = []
            for t in (r or {}).get("types") or []:
                try:
                    d = ast.literal_eval(t[t.index("{"):]) if "{" in t else {}
                    rows.append([float(x) for x in ast.literal_eval(str(d.get("p")))] if d.get("p") else None)
                except Exception:                                # noqa: BLE001
                    rows.append(None)
            rows = [x for x in rows if x and len(x) == len(names)]
            if rows:
                parts = []
                for k, nm in enumerate(names):
                    vals = sorted({x[k] for x in rows})
                    parts.append(f"{nm} = {vals[0]:g} for every type" + (" -- every term it multiplies is ABSENT in this run"
                                                                         if vals[0] == 0 else "")
                                 if len(vals) == 1 else f"{nm} varies by type ({', '.join(f'{v:g}' for v in vals[:6])})")
                out.append(f"type rows p = [{', '.join(names)}] of {sname}: " + "; ".join(parts))
    blocks = {b for sname in (st.get("acts_on") or []) for q in (F.get("sets") or []) if q["name"] == sname
              for b in (q.get("state") or [])}
    for k, v in (st.get("params") or {}).items():
        try:
            if float(v) == 0.0:
                out.append(f"param {k} = 0 -- the term it sets is ABSENT in this run")
        except (TypeError, ValueError):
            if isinstance(v, str) and v in blocks:
                out.append(f"param {k} names the state block '{v}' of the acted-on set: a PER-ELEMENT value (indexed by "
                           f"the element, learnable in a training run), which overrides any type-row constant it stands for")
    if names and "field" not in (st.get("params") or {}):
        out.append("no field named: the modulation factor Omega_i is 1 (not a term)")
    return "; ".join(out)


def _symbols_text(F: dict, ids: list) -> str:
    lines = [f"RUN {F['name']}" + (f" -- titled '{F['title']}'" if F.get("title") else "")]
    if F.get("units"):
        lines.append(f"UNITS: every length and time in the params below is in world units -- {F['units']}.")
    if F.get("training"):
        lines.append("A TRAINING RUN: the steps below are the model being fitted.")
    lines += _notation_lines(F)
    lines += _sets_text(F)
    lines += ["", "STEPS WITH AN EQUATION (id, operator, what it does, params [what each is], equation):"]
    by = {x["id"]: x for x in F["steps"]}
    for i in ids:
        st = by[i]
        lines.append(f"[{i}] {st['code']} ({st['title']}) on {', '.join(st['acts_on']) or '-'}")
        if st.get("gloss"):
            lines.append(f"     does: {st['gloss'][:300]}")
        if st.get("params"):
            ro = st.get("roles") or {}
            lines.append("     params: " + ", ".join(f"{k}={v}" + (f" [{ro[k].replace('_', ' ')}]" if ro.get(k) else "")
                                                for k, v in list(st["params"].items())[:12]))
        lines.append(f"     equation: {re.sub(chr(92) + 's+', ' ', (st.get('equation') or '').strip().strip('$'))[:500]}")
        alive = _alive_text(F, st)
        if alive:
            lines.append(f"     IN THIS RUN: {alive}")
        doc = _doc_of(st["code"])
        if doc:
            lines.append(f"     the operator's own words (its symbols are defined here): {doc}")
    return "\n".join(lines)


def complete_symbols(F: dict, st: dict, tries: int = 2) -> dict:
    """Ask for the legends the story lacks (a focused second call), merge them into `st`, up to `tries`
    times for what is still missing. Returns `st`; what could not be written stays missing and is said so
    in the pane."""
    from shutil import which
    st.setdefault("symbols", {})
    for x in F["steps"]:                                                       # the author's SYMBOLS, when the class carries them, are the legend; nothing is asked for those
        if x.get("symbols_author") and (x.get("equation") or "").strip() and not st["symbols"].get(x["id"]):
            st["symbols"][x["id"]] = [list(r) for r in x["symbols_author"]]
    for _ in range(max(1, tries)):
        ids = missing_symbols(F, st)
        if not ids:
            break
        cmd = [which("claude") or "claude", "-p", _symbols_text(F, ids), "--append-system-prompt", ROLE_SYMBOLS,
               "--model", MODEL, "--effort", "low", "--allowedTools", ""]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S, cwd=ROOT)
        if r.returncode != 0:
            raise RuntimeError(f"claude -p (symbols) failed ({r.returncode}): {(r.stderr or r.stdout)[-300:]}")
        m = re.search(r"\{.*\}", r.stdout, re.S)
        if not m:
            continue
        try:
            j = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        for k, v in (j.get("symbols") or {}).items():
            if str(k).lstrip("-").isdigit() and isinstance(v, list):
                rows = [[str(a[0]), str(a[1])] for a in v if isinstance(a, (list, tuple)) and len(a) >= 2]
                if rows and int(k) in ids:
                    st["symbols"][int(k)] = rows
        st["symbols_written"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return st


ROLE_MATH = """You write the MATH pane of a simulation watcher: every equation of one run at once, in ONE notation, each
located in the scene by one word, then one legend of every variable and every constant. Its reader is a scientist,
not the developer of this code. You are given the run's steps in the order its story tells them, each with its
name, what it acts on, its parameters (with what each is), its equation (LaTeX) and the legend already written for
it. Output ONLY one JSON object, no prose around it, no code fences:

{"equations": {"<id>": "<LaTeX>"}, "where": {"<id>": "<one word>"},
 "variables": [["symbol", "meaning"], ...], "constants": [["symbol", "meaning, value and unit"], ...]}

- equations: for EVERY step id given, its equation rewritten in ONE notation shared by all of them: the same
  quantity carries the same letter in every equation (if one step writes the neuron state x_i and another v_i,
  choose one and use it everywhere; the same for a weight, a position, a time constant, an angle). Rename
  letters ONLY to make them agree; never change the mathematics, the operators, the indices' meaning or the
  structure of an equation, and keep "+=" (\\mathrel{+}=) where it is: it says the step ADDS this term to the
  rate of change. Valid amsmath LaTeX, no $$ delimiters, no \\text with a bare underscore, no line breaks.
- THE GRAPH'S EDGES ARE `\\mathrm{edges}`, NEVER E (Cedric, 2026-11-06). A neighbour list, a contact graph or any other set of
  edges is written \\mathrm{edges} = \\{(i,j) : ...\\} in every equation and its legend row ("\\mathrm{edges}: the pairs of
  dots closer than the cutoff"), whatever letter the step's own equation gives it; E stays free for an energy.
- SUMS OVER EDGES ARE WRITTEN AS SUMS OVER NAMED POPULATIONS. An operator's registered equation may sum over the
  edges of a relation, "\\sum_{e : post(e)=i} w_e send(x_{pre(e)})" (edge e from its sender pre(e) to its receiver
  post(e), weight w_e). In this pane write it as the reader reads it: a sum over the SENDING population, named --
  "\\sum_{k \\in \\mathcal{I}} W^{in}_{ik}\\, r_k" for the input map (the retina channels k onto an input neuron i),
  "\\sum_{j \\in \\mathcal{N}} W_{ij}\\, \\tanh(x_j)" for the recurrent synapses, "\\sum_{i \\in \\mathcal{O}_m} W^{out}_{mi}\\,
  \\tanh(x_i)" for a readout (the output neurons of muscle m) -- and DEFINE those populations in the legend, with
  their counts and types from the RELATIONS and the type table: \\mathcal{I} the input neurons (the types the
  incoming relation lands on, e.g. "the 41 AF5 cells"), \\mathcal{N} the whole circuit, \\mathcal{O}_m the output
  neurons read by muscle m (the types the readout relation leaves from, e.g. "the 92 AMN cells for LR, the 35 AIN
  cells for MR"), and W^{in}, W, W^{out} the weight from the sender to the receiver along that relation, with its
  edge count. No pre(e) / post(e) in this pane. State for which i an equation holds when it is not every neuron
  ("for i in \\mathcal{I}" under the input map, "for each muscle m" under a readout).
- PLACEHOLDERS ARE REPLACED BY THIS RUN'S VALUES. A registered equation may write a parameter by its NAME -- send(·),
  act(·), gain, bias -- standing for whatever the step's params set. Write the actual thing: send: tanh -> \\tanh,
  activation: softplus -> \\mathrm{softplus}, activation: identity -> nothing, gain: 10 -> the symbol g_{in} with
  its value in the legend, bias: bias -> the per-element b_i. "send" and "act" are never left in an equation.
- NO SEED STEPS: a step run "once, at the start" (a seed) gets no equation in this pane and its symbols no legend row
  unless another equation uses them -- a seed's value is a learnable's starting value or a driven block's, not a law.
- AS THIS RUN COMPUTES IT. The registry's equation is the operator's general law; the run's parameters and type
  rows (SETS, params) say which of its terms are alive here. DROP a term whose coefficient is 0 for every type
  and every parameter of this run (a self-coupling s = 0, a noise of 0, a field that is not named so its factor
  is 1), and say so in the legend ("s_i: self-coupling, 0 in this run -- no such term"). A seed whose range is
  one value is that constant, not a formula. Keep a letter wherever its value varies by type, by element or is
  learnable, WITH ITS INDEX when it varies per element (a parameter that names a state block is per element:
  a_i = 1/tau_i, not a); never substitute a number into an equation. The pane must show the model this run
  actually runs.
- where: for EVERY step id, ONE short label locating the equation in the scene -- the body or population it
  changes (plant, eye, membrane, water, grid, cell, tissue ...) or, for a circuit, the population or the link
  it acts on, worked out from the RELATIONS and the types' roles: the input population ("retina→AF5" for the
  projection onto the input cells), the recurrent core ("INTG" or "circuit"), the readout ("AMN→LR"). A word, or
  two names joined by an arrow; the same label for the same part.
- variables: every symbol that CHANGES in time or differs per element (a state, an index, a sum over
  neighbours), ONCE across all equations, as ["symbol", "meaning"]: the symbol in LaTeX EXACTLY as it is written
  in your equations (x_i, W^{syn}_{ij}, \\tau_i, \\dot{u}, \\det C) -- it is typeset -- the meaning in at most 12
  words, naming the entity it belongs to.
- constants: every fixed parameter and number, ONCE, as ["symbol", "meaning, value and unit"], the value and
  unit from the params when given ("shear modulus of the rubber, 0.4 MPa"; "time step, 1/60 s"). In a TRAINING
  RUN a quantity named LEARNABLE is NOT a constant -- it is fitted and differs per element -- so it goes under
  variables, with "learnable" in its meaning (a synapse weight W_ij, a time constant tau_i, a bias b_i); only what
  the spec pins stays a constant.
- Use only the facts given; do not invent values or mechanisms. No hedging."""


def missing_math(F: dict, st: dict | None) -> bool:
    """True when the story has no MATH pane yet, or its equations do not cover every equation step."""
    if st is None:
        return False
    m = st.get("math") or {}
    eqs = m.get("equations") or {}
    ids = [x["id"] for x in F["steps"] if (x.get("equation") or "").strip() and not str(x.get("when", "")).startswith("once")]
    return bool(ids) and any(not (eqs.get(i) or eqs.get(str(i))) for i in ids)


def _math_text(F: dict, st: dict) -> str:
    lines = [f"RUN {F['name']}" + (f" -- titled '{F['title']}'" if F.get("title") else "")]
    if st.get("headline"):
        lines.append(f"THE SCENE: {st['headline']}. {st.get('story', '')[:600]}")
    if F.get("units"):
        lines.append(f"UNITS: every length and time in the params below is in world units -- {F['units']}.")
    if F.get("training"):
        T = F["training"]
        lines.append("A TRAINING RUN: the steps below are the model being fitted. LEARNABLE (fitted to the data, so these are "
                     "per-element VARIABLES of the model, not constants): "
                     + "; ".join(f"{e['code']} of {e['on']} ({e['what']})" for e in T.get("learnables") or [])
                     + f". Data: {T.get('reference')}; loss: {T.get('loss')}.")
    ent = st.get("entities") or {}
    if ent:
        lines.append("ENTITIES: " + "; ".join(f"{k} = {v}" for k, v in ent.items()))
    lines += _notation_lines(F)
    lines += _sets_text(F)
    lines += ["", "STEPS WITH AN EQUATION, in the story's order (id, name, operator, acts on, params [what each is], "
              "equation, legend):"]
    names = st.get("steps") or {}
    sym = st.get("symbols") or {}
    for x in ordered_steps(F, st):
        if not (x.get("equation") or "").strip():
            continue
        i = x["id"]
        nm = (names.get(i) or names.get(str(i)) or {}).get("name") or x["title"]
        lines.append(f"[{i}] {nm} | {x['code']} on {', '.join(x['acts_on']) or '-'}")
        if x.get("params"):
            ro = x.get("roles") or {}
            lines.append("     params: " + ", ".join(f"{k}={v}" + (f" [{ro[k].replace('_', ' ')}]" if ro.get(k) else "")
                                                for k, v in list(x["params"].items())[:12]))
        lines.append(f"     equation: {re.sub(chr(92) + 's+', ' ', (x.get('equation') or '').strip().strip('$'))[:500]}")
        leg = sym.get(i) or sym.get(str(i)) or []
        if leg:
            lines.append("     legend: " + "; ".join(f"{a} = {b}" for a, b in leg[:20]))
        alive = _alive_text(F, x)
        if alive:
            lines.append(f"     IN THIS RUN: {alive}")
        doc = _doc_of(x["code"], 500)
        if doc:
            lines.append(f"     the operator's own words: {doc}")
    return "\n".join(lines)


def complete_math(F: dict, st: dict, tries: int = 2) -> dict:
    """Write the MATH pane's content (`st['math']`): every equation in one notation, a locating word each, and the
    consolidated legend. Up to `tries` calls until every equation step is covered."""
    from shutil import which
    ids = [x["id"] for x in F["steps"] if (x.get("equation") or "").strip() and not str(x.get("when", "")).startswith("once")]
    if not ids:
        return st
    for _ in range(max(1, tries)):
        # MEDIUM effort: at low, one notation across nine equations came back with the same letter listed as a
        # variable and as a constant, and a learnable weight called fixed (exp24, 2026-10-12)
        cmd = [which("claude") or "claude", "-p", _math_text(F, st), "--append-system-prompt", ROLE_MATH,
               "--model", MODEL, "--effort", "medium", "--allowedTools", ""]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S, cwd=ROOT)
        if r.returncode != 0:
            raise RuntimeError(f"claude -p (math) failed ({r.returncode}): {(r.stderr or r.stdout)[-300:]}")
        m = re.search(r"\{.*\}", r.stdout, re.S)
        if not m:
            continue
        try:
            j = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        eqs = {int(k): str(v).strip().strip("$").strip() for k, v in (j.get("equations") or {}).items()
               if str(k).lstrip("-").isdigit() and str(v).strip()}
        _arrow = lambda v: re.sub(r"\s*(\\to|\\rightarrow|->|→)\s*", "→", str(v).strip())          # noqa: E731
        where = {int(k): re.sub(r"[^A-Za-z0-9_/→>+-]", "", _arrow(v).split()[0])[:24]
                 if str(v).strip() else "" for k, v in (j.get("where") or {}).items() if str(k).lstrip("-").isdigit()}
        rows = lambda key: [[str(a[0]), str(a[1])] for a in (j.get(key) or [])     # noqa: E731
                            if isinstance(a, (list, tuple)) and len(a) >= 2]
        cand = dict(equations=eqs, where=where, variables=rows("variables"), constants=rows("constants"),
                    written=time.strftime("%Y-%m-%d %H:%M:%S"), model=MODEL)
        if all(i in eqs for i in ids):
            st["math"] = cand
            if not math_legend_orphans(cand):                # every legend symbol occurs in some equation
                break
    return st


def math_legend_orphans(m: dict) -> list:
    """The legend rows whose symbol occurs in none of the pane's equations (a `d_i` under equations that write `d_m`):
    the agent's letters drifted between the equations and the legend, and the per-step legends would miss them."""
    alltex = " ".join(str(v) for v in (m.get("equations") or {}).values())
    fake = {"math": {"variables": list(m.get("variables") or []) + list(m.get("constants") or []), "constants": []}}
    hit = {a for a, _ in math_symbols_for(fake, alltex)}
    return [a for a, _ in fake["math"]["variables"] if a not in hit]


def _eq_steps(F: dict) -> list:
    """The steps the MATH pane writes an equation for: with an equation, and not a seed."""
    return [x for x in F["steps"] if (x.get("equation") or "").strip() and not str(x.get("when", "")).startswith("once")]


def _sig(x: dict) -> str:
    """A step's identity across the stories of one record: the operator, what it acts on, its registered equation, its parameters and their roles. The agent's
    legends quote the parameters' values ("viscous drag coefficient, 150 per time") and its equations drop the terms those values switch off, so two steps
    with the same signature have the same math and the same legend."""
    return json.dumps([x["code"], list(x.get("acts_on") or []), re.sub(r"\s+", " ", (x.get("equation") or "").strip()), x.get("params") or {}, x.get("roles") or {}],
                      sort_keys=True, default=str)


def _context(F: dict) -> str:
    """What else the agent reads when it writes the math, besides the steps: the units, the sets and relations, the learnables, the experiment's notation."""
    return json.dumps([F.get("units"), _sets_text(F), (F.get("training") or {}).get("learnables"), _notation_lines(F)], sort_keys=True, default=str)


def _siblings(save_to: str):
    """(facts, story) of the other saved stories of the same record (`<record>/story/NNNN.json`), the nearest step number first."""
    d, me = os.path.dirname(save_to), os.path.basename(save_to)
    num = lambda f: int(f[:-5])                                  # noqa: E731
    mine = num(me) if re.fullmatch(r"\d+\.json", me) else 0
    for f in sorted((f for f in os.listdir(d) if re.fullmatch(r"\d+\.json", f) and f != me), key=lambda f: abs(num(f) - mine)) if os.path.isdir(d) else []:
        got = load_saved(os.path.join(d, f))
        if got is not None:
            yield f, got[0], got[1]


def borrow(F: dict, st: dict, save_to: str | None) -> list:
    """FILL WHAT THE AGENT WOULD WRITE FROM A STORY ALREADY WRITTEN FOR THIS RECORD, no agent called (Cedric, 2026-11-07). A step of the same experiment whose
    equation steps have the same signatures (`_sig`) and the same context (`_context`) was written from the same facts, so its MATH pane is this one's: the
    equations, locating words and legend are copied under this step's ids. The per-step legends (`symbols`) are borrowed one step at a time, from the nearest
    story that has a step of the same signature. Returns what was copied, e.g. ["math from 0004.json", "symbols of steps [3, 5]"]; the caller saves."""
    if not save_to:
        return []
    got = []
    need = _eq_steps(F)
    lack = missing_symbols(F, st)
    if lack or missing_math(F, st):
        ctx = _context(F)
        for name, Fd, Sd in _siblings(save_to):
            if lack:
                by = {_sig(x): x["id"] for x in _eq_steps(Fd)}
                taken = []
                for x in F["steps"]:
                    if x["id"] in lack and _sig(x) in by:
                        rows = (Sd.get("symbols") or {}).get(by[_sig(x)])
                        if rows:
                            st.setdefault("symbols", {})[x["id"]] = [list(r) for r in rows]
                            taken.append(x["id"])
                if taken:
                    got.append(f"symbols of steps {taken} from {name}")
                    lack = [i for i in lack if i not in taken]
            if missing_math(F, st) and need and not missing_math(Fd, Sd) and _context(Fd) == ctx:
                pool = {}
                for x in _eq_steps(Fd):
                    pool.setdefault(_sig(x), []).append(x["id"])
                pick = {}
                for x in need:
                    ids = pool.get(_sig(x))
                    if not ids:
                        break
                    pick[x["id"]] = ids.pop(0)
                if len(pick) == len(need):
                    m = Sd["math"]
                    eq, wh = m.get("equations") or {}, m.get("where") or {}
                    g = lambda d, k: d.get(k) if k in d else d.get(str(k))          # noqa: E731
                    st["math"] = dict(m, equations={i: g(eq, d) for i, d in pick.items()}, where={i: (g(wh, d) or "") for i, d in pick.items()}, borrowed_from=name)
                    got.append(f"math from {name}")
            if not lack and not missing_math(F, st):
                break
    return got


_MATHING: dict = {}


def ensure_math(F: dict, st: dict, save_to: str | None) -> None:
    """A SAVED story without its MATH pane: write it in the background, once per step file, and re-save."""
    if not save_to or not missing_math(F, st):
        return
    if borrow(F, st, save_to):                                    # the same math is on disk for this record: copy it, no agent
        save(save_to, F, _norm(json.loads(json.dumps(st))))
        if not missing_math(F, st):
            return
    th = _MATHING.get(save_to)
    if (th is not None and th.is_alive()) or _recently_failed(("math", save_to)):
        return

    def work():
        try:
            st2 = complete_math(F, st)
            save(save_to, F, _norm(json.loads(json.dumps(st2))))
            p = os.path.join(CACHE, _key(F) + ".json")
            if os.path.exists(p):
                json.dump(st2, open(p + ".tmp", "w"), indent=1, default=str)
                os.replace(p + ".tmp", p)
        except Exception as e:                                   # noqa: BLE001
            _FAILED_AT[("math", save_to)] = time.time()
            print(f"[spec_story] math of {save_to}: {type(e).__name__}: {e}", flush=True)
    th = threading.Thread(target=work, daemon=True)
    _MATHING[save_to] = th
    th.start()


def math_pending(save_to: str | None) -> bool:
    th = _MATHING.get(save_to) if save_to else None
    return th is not None and th.is_alive()


def math_equations(F: dict, st: dict | None) -> list:
    """The MATH pane's equations in the story's order: (step id, LaTeX in the one notation, the original LaTeX,
    the locating word), one per equation step."""
    m = (st or {}).get("math") or {}
    eqs, where = m.get("equations") or {}, m.get("where") or {}
    out = []
    for x in ordered_steps(F, st):
        # NO SEEDS (Cedric, 2026-10-12): a seed writes a block once, at frame 0, and its "(0)" read as time 0 when it is
        # a learnable's starting value (overwritten by the trainer) or a driven block's (overwritten by the drive)
        if str(x.get("when", "")).startswith("once"):
            continue
        body = (x.get("equation") or "").strip()
        if body.startswith("$$") and body.endswith("$$"):
            body = body[2:-2].strip()
        if not body:
            continue
        i = x["id"]
        out.append((i, (eqs.get(i) or eqs.get(str(i)) or body), body, (where.get(i) or where.get(str(i)) or "")))
    return out


def _img(src: str, size=None) -> str:
    """An equation image; with `size` = (w, h) in pixels of a 2x render, shown at half so it is crisp."""
    attrs = f' width="{size[0] // 2}" height="{size[1] // 2}"' if size and size[0] and size[1] else ""
    return f'<img src="{src}" alt="equation"{attrs}>'


_UNI_TEX = {"∞": r"\infty", "·": r"\cdot", "≤": r"\le", "≥": r"\ge", "→": r"\to", "∑": r"\sum", "∂": r"\partial",
            "∇": r"\nabla", "×": r"\times", "°": r"^\circ", "′": "'", "−": "-", "₀": "_0", "₁": "_1", "₂": "_2",
            "₃": "_3", "ᵢ": "_i", "ⱼ": "_j", "ₖ": "_k", "ₙ": "_n", "ü": r"\ddot{u}", "ẍ": r"\ddot{x}", "ẋ": r"\dot{x}",
            "ẏ": r"\dot{y}", "ȧ": r"\dot{a}", "ṁ": r"\dot{m}", "ṙ": r"\dot{r}", "ṡ": r"\dot{s}", "ṫ": r"\dot{t}",
            "ṽ": r"\tilde{v}", "â": r"\hat{a}", "ê": r"\hat{e}", "î": r"\hat{i}"}


def tex_of_symbol(sym: str) -> str | None:
    """A legend symbol as LaTeX for the typesetter: the agent writes LaTeX (W^{syn}_{ij}, \\dot u) or plain Unicode
    (μ, t₀, u̇, ∞); both become one maths string. None for a symbol that needs no typesetting (a bare word, a
    single ASCII letter) -- the text column keeps it."""
    t = (sym or "").strip().strip("$").strip()
    if not t:
        return None
    t = re.sub("([A-Za-z])\u0307", r"\\dot{\1}", t)          # combining dot, diaeresis, hat after a letter
    t = re.sub("([A-Za-z])\u0308", r"\\ddot{\1}", t)
    t = re.sub("([A-Za-z])\u0302", r"\\hat{\1}", t)
    for k, v in _UNI_TEX.items():
        t = t.replace(k, v + " ")
    for name, ch in _GREEK.items():                          # μ -> \mu, Δ -> \Delta (defined further down)
        if len(ch) == 1:
            t = t.replace(ch, "\\" + name + " ")
    t = re.sub(r"[ ]+", " ", t).strip()
    if "\\" not in t and "_" not in t and "^" not in t and re.fullmatch(r"[A-Za-z]|[A-Za-z][A-Za-z ,/()-]{2,}", t):
        return None                                           # a word or a bare letter: text is enough
    t = re.sub(r"_([A-Za-z0-9]{2,})(?![}\w])", r"_{\1}", t)    # W_ij -> W_{ij}, tau_min -> tau_{min}
    t = re.sub(r"\^([A-Za-z0-9]{2,})(?![}\w])", r"^{\1}", t)
    return t


def _sym_cell(sym: str, sym_url) -> str:
    """The symbol column: typeset when a renderer offers `sym_url(tex) -> (url, (w, h)) | None`, else text."""
    esc = htmlmod.escape
    tex = tex_of_symbol(sym) if sym_url else None
    got = sym_url(tex) if tex else None
    if got and got[0]:
        url, size = got
        attrs = f' width="{size[0] // 2}" height="{size[1] // 2}"' if size and size[0] and size[1] else ""
        return f'<span class="ps-sy"><img class="ps-syi" src="{url}" alt="{esc(sym)}"{attrs}></span>'
    return f'<span class="ps-sy">{esc(_greek(sym))}</span>'


def render_math_html(F: dict, st: dict | None, meq_url, pending: bool = False, meq_size=None, sym_url=None) -> str:
    """The MATH pane: every equation of the run at once, in the story's order and one notation, a locating word to
    the right of each; then the variables and the constants, each once."""
    esc = htmlmod.escape
    rows = math_equations(F, st)
    if not rows:
        return '<div class="ps-g">this run has no equation step</div>'
    m = (st or {}).get("math") or {}
    L = ['<div class="ps-mp">']
    L.append('<div class="ps-h">maths</div>')
    L.append('<table class="ps-a ps-mq">')
    by = {x["id"]: x for x in F["steps"]}
    for k, (i, tex, orig, word) in enumerate(rows):
        st_ = by.get(i) or {}
        url = method_url(step_class(st_), "forward")[0] if step_class(st_) is not None else ""
        label = (f'<a class="ps-ln" href="{esc(url)}" target="_blank" title="{esc(st_.get("code", ""))} -- source on '
                 f'GitHub">{esc(word)}</a>' if url else esc(word))
        exb = explain_block(st_, sym_url=sym_url)
        if exb:                                       # a click on the equation opens what the operator does, under it
            label = label.replace("<a ", '<a onclick="event.stopPropagation()" ', 1) if label.startswith("<a ") else label
            L.append(f'<tr class="ps-mrow" style="cursor:pointer" title="click: what this equation does" '
                     f'onclick="var n=this.nextElementSibling;n.style.display=n.style.display===\'none\'?\'\':\'none\'">'
                     f'<td class="ps-mi">{_img(meq_url(k), meq_size(k) if meq_size else None)}</td>'
                     f'<td class="ps-mw">{label}</td></tr>'
                     f'<tr style="display:none"><td colspan="2">{exb.replace("display:none", "")}</td></tr>')
        else:
            L.append(f'<tr><td class="ps-mi">{_img(meq_url(k), meq_size(k) if meq_size else None)}</td>'
                     f'<td class="ps-mw">{label}</td></tr>')
    L.append('</table>')
    eqs_all = m.get("equations") or {}
    kept_tex = " ".join(t for _i, t, _o, _w in rows)
    seed_tex = " ".join(str(eqs_all.get(x["id"]) or eqs_all.get(str(x["id"])) or "") for x in F["steps"]
                        if str(x.get("when", "")).startswith("once"))
    only_seed = ({a for a, _ in math_symbols_for(st, seed_tex)} - {a for a, _ in math_symbols_for(st, kept_tex)}
                 if seed_tex.strip() else set())
    for key, title, sub in (("variables", "variables", "what changes in time or per element"),
                            ("constants", "constants", "the fixed parameters, with their values")):
        rs = [row for row in (m.get(key) or []) if row[0] not in only_seed]
        if rs:
            L.append(f'<div class="ps-h">{title} <span class="ps-g" style="font-size:85%;font-weight:400">{sub}</span></div>')
            L.append('<table class="ps-e ps-ml">')
            for a, b in rs:
                L.append(f'<tr><td>{_sym_cell(a, sym_url)}</td><td class="ps-sym">{esc(b)}</td></tr>')
            L.append('</table>')
    L.append('</div>')
    return "\n".join(L)


_COMPLETING: dict = {}


def ensure_symbols(F: dict, st: dict, save_to: str | None) -> None:
    """A SAVED story whose equations lack their legends: complete them in the background, once per step file,
    and re-save (the pane re-reads the file by its mtime). Also refreshes the agent's cache for these facts."""
    if not save_to or not missing_symbols(F, st):
        return
    if borrow(F, st, save_to):                                    # the legends of the same steps are on disk for this record: copy them, no agent
        save(save_to, F, _norm(json.loads(json.dumps(st))))
        if not missing_symbols(F, st):
            return
    th = _COMPLETING.get(save_to)
    if (th is not None and th.is_alive()) or _recently_failed(("symbols", save_to)):
        return

    def work():
        try:
            st2 = complete_symbols(F, st)
            save(save_to, F, _norm(json.loads(json.dumps(st2))))
            p = os.path.join(CACHE, _key(F) + ".json")
            if os.path.exists(p):
                json.dump(st2, open(p + ".tmp", "w"), indent=1, default=str)
                os.replace(p + ".tmp", p)
        except Exception as e:                                   # noqa: BLE001
            _FAILED_AT[("symbols", save_to)] = time.time()
            print(f"[spec_story] symbols of {save_to}: {type(e).__name__}: {e}", flush=True)
    th = threading.Thread(target=work, daemon=True)
    _COMPLETING[save_to] = th
    th.start()


def generate(F: dict, save_to: str | None = None) -> dict:
    """Ask the agent, check the answer, store it. Blocking; ~10-25 s. The legends and the math the record already holds (`borrow`, `save_to`'s neighbours) are copied, not asked for."""
    os.makedirs(CACHE, exist_ok=True)
    from shutil import which
    t0 = time.time()
    cmd = [which("claude") or "claude", "-p", _facts_text(F), "--append-system-prompt",
           ROLE_TRAIN if F.get("training") else ROLE,
           "--model", MODEL, "--effort", "low", "--allowedTools", ""]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S, cwd=ROOT)
    if r.returncode != 0:
        raise RuntimeError(f"claude -p failed ({r.returncode}): {(r.stderr or r.stdout)[-300:]}")
    out = _parse(r.stdout, F)
    out.update(model=MODEL, seconds=round(time.time() - t0, 1), version=PROMPT_VERSION,
               written=time.strftime("%Y-%m-%d %H:%M:%S"))
    p = os.path.join(CACHE, _key(F) + ".json")
    # THE STORY IS KEPT THE MOMENT IT EXISTS (2026-10-18): the legends and the math below are two more agent calls of up to
    # TIMEOUT_S each; a timeout in either raised, the story that had just been written was thrown away, and the next poll of the
    # pane asked for it all again -- 41 timeouts in a session. They are best effort now and `ensure_symbols` / `ensure_math` finish
    # whatever is missing later, at most once per RETRY_AFTER_S.
    json.dump(out, open(p + ".tmp", "w"), indent=1)
    os.replace(p + ".tmp", p)
    borrow(F, out, save_to)
    for fill, missing in ((complete_symbols, missing_symbols), (complete_math, missing_math)):
        if missing(F, out):
            try:
                fill(F, out)
            except Exception as e:                       # noqa: BLE001
                print(f"[spec_story] {F['name']}: {fill.__name__}: {type(e).__name__}: {e}", flush=True)
    json.dump(out, open(p + ".tmp", "w"), indent=1)
    os.replace(p + ".tmp", p)
    return out


_RUNNING: dict = {}
_ERRORS: dict = {}
_FAILED_AT: dict = {}                  # key -> when the agent last failed for it: no new call before RETRY_AFTER_S
RETRY_AFTER_S = 4 * 3600


def _recently_failed(key) -> bool:
    return key in _FAILED_AT and time.time() - _FAILED_AT[key] < RETRY_AFTER_S


def fallback_story(F: dict) -> dict:
    """A story WITHOUT the agent, from the facts alone, for a pane that has none yet: what the run holds (its sets and types) and the
    steps it takes in order, each by its registry title. Never saved; the agent's story replaces it when it lands."""
    sets = F.get("sets") or []
    lead = next((x for x in sets if x.get("types")), sets[0] if sets else None)
    parts = []
    if lead:
        ty = []
        for t in lead.get("types") or []:
            m = re.search(r"'count': '(\d+)'", t)
            ty.append(f"{t.split()[0]} ({m.group(1)})" if m else t.split()[0])
        parts.append(f"The box holds {lead['n']:,} {lead['name']}s" + (f" of {len(ty)} types: {', '.join(ty)}" if ty else "") + ".")
        sub = next((x for x in sets if x.get("parent") == lead["name"]), None)
        if sub:
            parts.append(f"Each is made of {sub.get('per_parent') or '?'} material points ({sub['n']:,} in all).")
    if F.get("simulated_time"):
        parts.append(f"It is run for {F.get('frames')} frames, {F['simulated_time']} of simulated time.")
    steps = [st for st in F.get("steps") or []]
    if steps:
        parts.append("Each step is listed below in the order the run takes it.")
    ent = {}
    for x in sets:
        ent[x["name"]] = f"the {x['n']:,} {x['name']}s" if x.get("n") else f"the {x['name']}"
    for f in F.get("fields") or []:
        ent[f["name"]] = f"the {f['name']} field"
    return {"headline": (f"a model of {lead['n']:,} {lead['name']}s" if lead else "a model"), "story": " ".join(parts),
            "change": "", "entities": ent, "learnables": {}, "symbols": {}, "rendering": "",
            "order": [int(st["id"]) for st in steps],
            "steps": {int(st["id"]): {"name": st["title"], "what": ""} for st in steps}, "math": {}, "fallback": True}


def save(path: str, F: dict, st: dict) -> None:
    """The step's story, SAVED BESIDE ITS PICTURE (`<record>/story/NNNN.json`): the facts it was written
    from and the words, so the pane renders it as it was written -- no registry walk, no agent -- every
    time the step is shown again, whatever has changed in the code since."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump({"facts": F, "story": st}, open(path + ".tmp", "w"), indent=1, default=str)
    os.replace(path + ".tmp", path)


def load_saved(path: str | None):
    """(facts, story) from a step's saved story file, or None."""
    if not path or not os.path.exists(path):
        return None
    try:
        j = json.load(open(path))
        return j["facts"], _norm(j["story"])
    except Exception:                                            # noqa: BLE001
        return None


def ensure(F: dict, save_to: str | None = None) -> dict | None:
    """The story if it is on disk; else start writing it in the background (once) and return None.
    `save_to`: the step's story file, written as soon as the story exists."""
    st = cached(F)
    if st is not None:
        if save_to and not os.path.exists(save_to):
            save(save_to, F, st)
        return st
    k = _key(F)
    if k in _RUNNING and _RUNNING[k].is_alive():
        return None
    if _recently_failed(k):                                      # not again, not on every poll of the page
        return None

    def work():
        try:
            st2 = generate(F, save_to)
            if save_to:
                save(save_to, F, _norm(json.loads(json.dumps(st2))))
            _ERRORS.pop(k, None)
            _FAILED_AT.pop(k, None)
        except Exception as e:                                   # noqa: BLE001
            _ERRORS[k] = f"{type(e).__name__}: {e}"
            _FAILED_AT[k] = time.time()
            print(f"[spec_story] {F['name']}: {_ERRORS[k]}", flush=True)
    th = threading.Thread(target=work, daemon=True)
    _RUNNING[k] = th
    th.start()
    return None


def error(F: dict) -> str | None:
    return _ERRORS.get(_key(F))


def pending(F: dict) -> bool:
    k = _key(F)
    return cached(F) is None and k in _RUNNING and _RUNNING[k].is_alive()


# --------------------------------------------------------------------------- the pane (HTML)
def _fallback_name(st: dict) -> str:
    return _summary().lower_first(st["title"])


def _short(text: str, n: int = 160) -> str:
    text = re.sub(r"`([^`]*)`", r"\1", text or "")
    return text if len(text) <= n else text[: text.rfind(" ", 0, n)] + "..."


_GREEK = {"alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "Delta": "Δ", "epsilon": "ε", "eta": "η",
          "theta": "θ", "kappa": "κ", "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ", "pi": "π", "rho": "ρ",
          "sigma": "σ", "tau": "τ", "phi": "φ", "Phi": "Φ", "psi": "ψ", "omega": "ω", "Omega": "Ω",
          "nabla": "∇", "partial": "∂"}


def _greek(sym: str) -> str:
    """A symbol as it reads in the equation: Greek names to letters, a `hat` to a circumflex."""
    t = re.sub(r"\\?\b(" + "|".join(_GREEK) + r")(?=\b|_)", lambda m: _GREEK[m.group(1)], sym)
    return re.sub(r"\b([A-Za-z])(?: |_)?hat(?=\b|_)", lambda m: m.group(1) + "\u0302", t)


def ordered_steps(F: dict, story: dict | None) -> list[dict]:
    """The steps in the order the run EXECUTES them -- the schedule's, seeds first (`spec_summary.schedule_order`; the facts list them so)
    -- for every pane that lists them: the plexus pane, the math pane and the story's equations. The agent's own `order` (a narrative,
    read beside its story) is no longer used: it put a cell cycle before the contacts that precede it in every frame, and two panes
    that sort their steps by a free choice cannot agree (Cedric, 2026-11-01: "the two should be very consistent")."""
    return list(F["steps"])


def step_equation(st: dict, story: dict | None) -> str:
    """The equation the panes draw for a step: THE MATH PANE'S (one notation, as this run computes it, sums over
    named populations) once it is written, else the operator's registered one. Cedric, 2026-10-12: "the math is the
    reference, the plexus should consume the math tab" -- so the two tabs cannot show two equations for one step."""
    m = (story or {}).get("math") or {}
    eqs = m.get("equations") or {}
    own = eqs.get(st["id"]) or eqs.get(str(st["id"]))
    if own:
        return str(own).strip().strip("$").strip()
    body = (st["equation"] or "").strip()
    if body.startswith("$$") and body.endswith("$$"):
        body = body[2:-2].strip()
    return body


def equations(F: dict, story: dict | None) -> list[str]:
    """The LaTeX bodies the pane draws, in the order it draws them (one per step that has one)."""
    return [b for b in (step_equation(st, story) for st in ordered_steps(F, story)) if b]


def _norm_tex(t: str) -> str:
    t = re.sub(r"\\[,;!: ]", "", (t or "").strip().strip("$"))
    t = t.replace(" ", "")
    t = re.sub(r"(\\math[a-z]+)\{([A-Za-z])\}", r"\1\2", t)          # \mathcal{O} and \mathcal O alike
    return re.sub(r"([_^])\{([A-Za-z0-9])\}", r"\1\2", t)


def math_symbols_for(story: dict | None, tex: str) -> list:
    """The math pane's legend rows whose symbol occurs in `tex` -- the per-step legend, taken from the ONE legend so
    the two tabs agree. A bare letter must stand alone (u, not the u of \\mu); anything else is a substring."""
    m = (story or {}).get("math") or {}
    rows = list(m.get("variables") or []) + list(m.get("constants") or [])
    eq, out = _norm_tex(tex), []
    for sym, meaning in rows:
        for part in [x for x in re.split(r",(?![^{]*\})", _norm_tex(sym)) if x]:
            if re.fullmatch(r"[A-Za-z]", part):
                hit = re.search(r"(?<![A-Za-z\\\\])" + re.escape(part) + r"(?![A-Za-z_^])", eq) is not None
            else:
                hit = part in eq
            if hit:
                out.append([sym, meaning])
                break
    return out


WHEN_SHORT = {"once, at the start": "once", "every frame": "each frame", "every substep": "each substep",
              "never (declared, not scheduled)": "unscheduled"}


def _rgb_hex(c) -> str | None:
    """A colour as #rrggbb, from [r, g, b] in 0-1 or 0-255, or a '#hex' / name string."""
    if isinstance(c, str):
        return c if c.startswith("#") else None
    try:
        v = [float(x) for x in c][:3]
        if max(v) <= 1.0:
            v = [x * 255 for x in v]
        return "#" + "".join(f"{int(round(max(0, min(255, x)))):02x}" for x in v)
    except Exception:                                            # noqa: BLE001
        return None


# what each experiment's own trajectory renderer draws -- one line, read by `rendering_facts` when a step's why
# names the tool (a studio movie replaces the spec's live movie, so the `plotting:` block alone would mislead)
STUDIO_LOOKS = {
    "exp22_studio": "surfaces, gradient background, floor with shadows, perspective camera",
    "exp23_studio": "each cell its own surface, coloured point by point from its yolk (orange animal half, cream "
                    "vegetal half), matte, teal-grey background, seen from above the animal pole; a cross-section "
                    "inset bottom-left (a vertical slice through the centre: cell walls, the yolk gradient) and a "
                    "black cell-count plot top-right",
}


def rendering_facts(plotting: dict, why: str = "") -> list[tuple[str, str, list]]:
    """How the movie draws the scene, read from the spec's `plotting:` block (and the step's note, when the
    movie is a studio render of the trajectory): (label, text, [colour swatches]) rows."""
    P = plotting or {}
    rows = []
    mode = P.get("render_3d", "dots")
    what = {"dots": "points, one dot per material point", "surface": "one skin built at the start and carried",
            "contour": "surfaces re-contoured every frame", "compartments": "one surface per set"}.get(mode, mode)
    rows.append(("renderer", f"{P.get('renderer', 'default')}, {what}", []))
    shown = [x for x in [P.get("subject")] + list(P.get("also_sets") or []) + list(P.get("compartment_sets") or []) if x]
    if shown:
        rows.append(("shows", ", ".join(dict.fromkeys(shown)), []))
    cols = []
    for k, v in (P.get("colors") or {}).items():
        h = _rgb_hex(v)
        if h:
            cols.append((k, h))
    for k, v in (P.get("also_colors") or {}).items():
        h = _rgb_hex(v)
        if h:
            cols.append((k, h))
    if cols:
        rows.append(("colours", ", ".join(k for k, _ in cols), cols))
    cb = P.get("contour_bodies")
    if cb:
        cb = cb if isinstance(cb, dict) else {"block": str(cb)}
        txt = "each cell its own surface" + (", touching cells sharing one face" if cb.get("mode", "partition") == "partition" else "")
        sw = []
        by = cb.get("color_by")
        if str(P.get("cell_colours", "")).lower() in ("type", "id"):     # the movie colours each cell by its TYPE or its id (the `colours` row above): no second legend for a colour map the studio does not use (Cedric, 2026-11-01)
            by = None
        if by:
            txt += f"; each cell coloured by its {by.get('block', 'chem')} through the {by.get('cmap', 'Blues')} colour map"
            try:
                import matplotlib.pyplot as _plt
                cm = _plt.get_cmap(str(by.get("cmap", "Blues")))
                lo = float(by.get("floor", 0.25))
                sw = [(f"low {by.get('block', 'chem')}", "#%02x%02x%02x" % tuple(int(255 * c) for c in cm(lo)[:3])),
                      (f"high {by.get('block', 'chem')}", "#%02x%02x%02x" % tuple(int(255 * c) for c in cm(1.0)[:3]))]
            except Exception:                                    # noqa: BLE001
                sw = []
        elif cb.get("colors"):
            sw = [(f"cell {i}", h) for i, h in enumerate(_rgb_hex(c) for c in cb["colors"]) if h][:4]
        rows.append(("cells", txt, sw))
    cam = P.get("camera") or {}
    if cam:
        rows.append(("camera", ", ".join(f"{k} {v}" for k, v in cam.items()), []))
    bg = [f"background {P['background']}"] if P.get("background") else []
    if P.get("floor"):
        bg.append(f"floor {P['floor']}")
    if bg:
        rows.append(("scene", ", ".join(bg), []))
    if P.get("duration_s"):
        rows.append(("movie", f"{P['duration_s']} s long", []))
    m = re.search(r"tools/(\w+_studio)\.py", why or "")
    if m or "studio render" in (why or ""):
        tool = m.group(1) if m else "exp22_studio"
        look = STUDIO_LOOKS.get(tool, "")
        rows.append(("studio", f"this step's movie is a render of the recorded trajectory (tools/{tool}.py)"
                               + (f": {look}" if look else ""), []))
    return rows


def render_html(s: dict, F: dict, story: dict | None, icon_url, eq_url, note: str = "", why: str = "",
                eq_size=None, sym_url=None) -> str:
    """Headline + story, the entities, then the GNN: one row per step, each with its kind's glyph, in the
    story's order."""
    esc = lambda x: htmlmod.escape(str(x))                     # noqa: E731
    L = ['<div class="ps">']
    if story and story.get("headline"):
        L.append(f'<div class="ps-hl">{esc(story["headline"])}</div>')
        L.append(f'<div class="ps-story">{esc(story["story"])}</div>')
        if story.get("change"):
            L.append(f'<div class="ps-chg">{esc(story["change"])}</div>')
    else:
        L.append(f'<div class="ps-hl">{esc(F["title"] or F["name"])}</div>')
        L.append(f'<div class="ps-story ps-g">{esc(note or "a plain-English description of this run is being written...")}</div>')
    ent = (story or {}).get("entities") or {}
    if ent:                                                      # the agent's plain names, the code name kept in grey
        s = dict(s, sets=[dict(r, title=ent.get(r["name"]) or r.get("title") or "") for r in s["sets"]],
                 fields=[dict(f, title=ent.get(f["name"]) or f.get("title") or "") for f in s.get("fields") or []])
    BL = bullet_links(s, F)
    link = lambda x: (BL.get(("step", x["id"])) if "when" in x else BL.get(("learnable", x["id"])) if "how" in x  # noqa: E731
                      else BL.get(("set", x.get("name"))) or BL.get(("relation", x.get("name"))) or "")
    srcs = {k[1]: v for k, v in BL.items() if k[0] in ("set", "relation") and v}
    kids = {}
    for r in F.get("sets") or []:
        if r.get("parent"):
            kids.setdefault(r["parent"], []).append(r["name"])
    for par, ks in kids.items():                          # a one-body parent holding ONE set is drawn in its place
        if len(ks) == 1 and srcs.get(ks[0]) and not srcs.get(par):
            srcs[par] = srcs[ks[0]]
    T = F.get("training")
    if T:
        # A TRAINING RUN: the learnables (what is fitted) where a simulation has its entities, then the data and
        # the loss in grey, then the MODEL -- its entities and its GNN steps, exactly as a simulation shows them.
        L.append('<div class="ps-h">learnables <span class="ps-g" style="font-size:85%;font-weight:400">'
                 'what the training fits</span></div><table class="ps-a">')
        lnames = (story or {}).get("learnables") or {}
        for e in T["learnables"]:
            nm = _summary().lower_first((lnames.get(e["id"]) or {}).get("name") or e["role"])
            what = (lnames.get(e["id"]) or {}).get("what") or e["role"]
            nm_html = (f'<a class="ps-ln" href="{esc(link(e))}" target="_blank" title="source on GitHub">{esc(nm)}</a>'
                       if link(e) else esc(nm))
            icon = (f'<img src="{icon_url(e["kind"])}" alt="{esc(e["kind"])}" title="{esc(e["kind"])}">'
                    if e.get("kind") else "")
            meta = " &middot; ".join(esc(x) for x in (e["code"], e["on"], e["how"], e["extra"]) if x)
            L.append(f'<tr><td class="ps-i">{icon}</td><td class="ps-t"><span class="ps-nm">{nm_html}</span>'
                     f'<div class="ps-w">{esc(what)}</div><div class="ps-m">{meta}</div></td></tr>')
        L.append("</table>")
        for lab, v in (("data", T["reference"]), ("driven by", T["drive"]), ("observed", T["observe"]),
                       ("loss", T["loss"]), ("schedule", T["scheme"])):
            if v:
                L.append(f'<div class="ps-m" style="margin:0 0 1px 34px">{lab}: {esc(v)}</div>')
        L.append('<div class="ps-h">model <span class="ps-g" style="font-size:85%;font-weight:400">'
                 'the graph network being trained</span></div>')
    def _maker_html(r):                                   # a relation's edges: the tool that wrote them
        u = BL.get(("relation_maker", r["name"]))
        return (f' &middot; <a class="ps-ln" href="{esc(u)}" target="_blank" title="the code that wrote its edge file">'
                f'edges written by</a>' if u else "")
    L.append(_summary().render_entities_html(s, href=lambda r: srcs.get(r["name"]), extra=_maker_html).replace(
        '<div class="ps-h">states</div>', '<div class="ps-h" style="font-size:13px">states</div>' if T else
        '<div class="ps-h">states</div>'))
    sub = (f' <span class="ps-g" style="font-size:85%;font-weight:400">substep {esc(F["substep"])}</span>'
           if F["substep"] else "")
    L.append(f'<div class="ps-h"{" style=\"font-size:13px\"" if T else ""}>dynamics{sub}</div>')
    L.append('<table class="ps-a">')
    k_eq = 0
    names = (story or {}).get("steps") or {}
    for st in ordered_steps(F, story):
        nm = _summary().lower_first((names.get(st["id"]) or {}).get("name") or _fallback_name(st))
        # NO DOCSTRING TEXT ON THE PANE (Cedric, 2026-10-14): before the plain-English names are written the line was
        # the operator docstring's first sentence -- uppercase lead-ins, quotes with his name, cut mid-sentence. The
        # pane shows the short name alone until the written description arrives.
        what = (names.get(st["id"]) or {}).get("what") or ""
        cnt = f' <span class="ps-g">&times;{st["count"]}</span>' if st["count"] > 1 else ""
        meta = f'{WHEN_SHORT.get(st["when"], st["when"])} &middot; {esc(st["code"])}'
        _hk = BL.get(("step_hook", st["id"]))
        if _hk:                                            # the variant's own piece: its hook, beside the base forward
            meta += f' &middot; <a class="ps-ln" href="{esc(_hk)}" target="_blank" title="the variant\'s own code">variant</a>'
        if st["acts_on"]:
            meta += f' &middot; {esc(", ".join(st["acts_on"]))}'
        nm_html = (f'<a class="ps-ln" href="{esc(link(st))}" target="_blank" title="source on GitHub">{esc(nm)}</a>'
                   if link(st) else esc(nm))
        exb = explain_block(st, sym_url=sym_url)
        nm_html = nm_html.replace("<a ", '<a onclick="event.stopPropagation()" ', 1) if nm_html.startswith("<a ") else nm_html
        tr_open = ('<tr class="ps-mrow" style="cursor:pointer" title="click: what this step does" '
                   'onclick="var n=this.querySelector(\'.ps-ex\');if(n)n.style.display=n.style.display===\'none\'?\'\':\'none\'">'
                   if exb else '<tr>')
        L.append(tr_open +
                 f'<td class="ps-i"><img src="{icon_url(st["kind"])}" alt="{esc(st["kind"])}" '
                 f'title="{esc(st["kind"])}"></td><td class="ps-t"><span class="ps-nm">{nm_html}</span>{cnt}'
                 f'<div class="ps-w">{esc(what)}</div><div class="ps-m">{meta}</div>{exb}')
        # No equation and no symbol legend under a step in THIS pane (Cedric, 2026-10-22: "only plain concise English"):
        # the equations and every symbol live in the math pane, `render_math_html`.
        L.append('</td></tr>')
    L.append("</table>")
    rows = rendering_facts(s.get("plotting") or {}, why)
    if rows:
        L.append('<div class="ps-h">rendering</div>')
        studio = [r for r in rows if r[0] == "studio"]
        if studio:                    # this step's movie IS the trajectory render: say that first, the spec's live
            rows = studio + [("live movie" if r[0] == "renderer" else r[0], r[1], r[2])     # movie after it
                             for r in rows if r[0] != "studio"]
        elif story and story.get("rendering"):
            L.append(f'<div class="ps-w" style="white-space:normal;margin:0 0 3px 34px">{esc(story["rendering"])}</div>')
        for lab, txt, sw in rows:
            chips = "".join(f'<span class="ps-sw" style="background:{esc(h)}" title="{esc(k)} {esc(h)}"></span>'
                            for k, h in sw)
            L.append(f'<div class="ps-m" style="margin:0 0 1px 34px">{esc(lab)}: {chips}{esc(txt)}</div>')
    if story and story.get("seconds"):
        L.append(f'<div class="ps-m">words: {esc(story.get("model"))}, {esc(story.get("seconds"))} s, '
                 f'{esc(story.get("written"))}; facts: the spec and the operator registry</div>')
    L.append("</div>")
    return "\n".join(L)


def _has_spec(path: str | None) -> bool:
    if not path or not os.path.exists(path):
        return False
    return any(ln.strip() and not ln.lstrip().startswith("#") for ln in open(path, errors="replace"))


def _spec_from_why(why: str | None) -> str | None:
    """A run's config named in a step's note (`run    <dir>` or `spec   training/<group>/<name>`)."""
    if not why or not os.path.exists(why):
        return None
    for ln in open(why, errors="replace"):
        head, _, rest = ln.partition(" ")
        tok = rest.strip().split(" ")[0] if rest.strip() else ""
        if head == "run" and tok:
            c = os.path.join(tok, "config.yaml")
            if os.path.exists(c):
                return c
    return None


def resolve(png: str | None) -> dict | None:
    """What a step's pane tells: its own spec; else (a measurement, a figure) the run its note names; else
    the nearest earlier step that has a spec. Returns {spec, why, story, png, lead} or None."""
    if not png:
        return None
    import glob
    rec, n = os.path.dirname(os.path.dirname(png)), os.path.basename(png)[:4]
    side = lambda k, m, ext: os.path.join(rec, k, m + ext)                      # noqa: E731
    me = dict(spec=side("spec", n, ".yaml"), why=side("why", n, ".txt"), story=side("story", n, ".json"), png=png, lead="")
    if _has_spec(me["spec"]):
        return me
    alt = _spec_from_why(me["why"])
    if alt:
        return dict(me, spec=alt, png=None,
                    lead=f"This step is not a run of this record; it shows the run its note names "
                         f"({os.path.basename(os.path.dirname(alt))}):")
    pngs = sorted(glob.glob(os.path.join(rec, "png", "[0-9][0-9][0-9][0-9].png")))
    for q in reversed([x for x in pngs if os.path.basename(x)[:4] < n]):
        m = os.path.basename(q)[:4]
        if _has_spec(side("spec", m, ".yaml")):
            return dict(spec=side("spec", m, ".yaml"), why=side("why", m, ".txt"), story=side("story", m, ".json"),
                        png=q, lead=f"This step is a measurement; the run it follows is step {m}:")
    return None


def _walk(records: list[str]):
    """(record id, resolved step) for every step of each record that tells a spec, newest last."""
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from plexus.gui import watch as W
    for rid in records:
        for png in W.shots(os.path.join(ROOT, rid)):
            r = resolve(png)
            if r is not None and os.path.exists(r["story"]):
                yield rid, r


def refresh_equations(records: list[str]) -> None:
    """Re-read each saved story's facts from its spec and the LIVE registry, and update the EQUATIONS (and titles)
    of its steps in place -- the words are kept; a step whose equation changed loses its legend and the story
    its MATH pane, so the next repair writes them again. No agent is called."""
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from plexus.gui import watch as W
    n_seen = n_changed = 0
    seen = set()
    for rid, r in _walk(records):
        if r["story"] in seen:
            continue
        seen.add(r["story"])
        got = load_saved(r["story"])
        if got is None:
            continue
        F, st = got
        try:
            F2 = facts(_summary().summarise(r["spec"]), context=W._why_context(r["why"]),
                       diff=spec_diff(previous_spec(r["png"]), r["spec"]))
        except Exception as e:                                    # noqa: BLE001
            print(f"SKIP  {rid}/{os.path.basename(r['story'])[:4]}  facts: {type(e).__name__}: {str(e)[:120]}", flush=True)
            continue
        n_seen += 1
        if [x["code"] for x in F2["steps"]] != [x["code"] for x in F["steps"]]:
            print(f"DIFF  {rid}/{os.path.basename(r['story'])[:4]}  the steps changed; rewrite it with --force", flush=True)
            continue
        changed = []
        if F.get("sets") != F2.get("sets") or F.get("relations") != F2.get("relations"):
            F["sets"], F["relations"] = F2["sets"], F2.get("relations") or []
            changed.append("sets")
        for a, b in zip(F["steps"], F2["steps"]):
            if (a.get("equation") or "") != (b.get("equation") or ""):
                a["equation"] = b.get("equation") or ""
                a["title"], a["gloss"] = b.get("title", a.get("title")), b.get("gloss", a.get("gloss"))
                changed.append(a["id"])
        if changed:
            for i in changed:
                if i != "sets":
                    (st.get("symbols") or {}).pop(i, None)
            st.pop("math", None)
            save(r["story"], F, st)
            n_changed += 1
            print(f"ok    {rid}/{os.path.basename(r['story'])[:4]}  equations of steps {changed} updated", flush=True)
    print(f"[spec_story] refresh: {n_seen} stories read, {n_changed} updated", flush=True)


def write_math(records: list[str], jobs: int, force: bool) -> None:
    """The MATH pane of every saved story that lacks one (or every one, with `force`), `jobs` agents at once."""
    from concurrent.futures import ThreadPoolExecutor
    t0 = time.time()
    todo, seen = [], set()
    for rid, r in _walk(records):
        if r["story"] in seen:
            continue
        seen.add(r["story"])
        got = load_saved(r["story"])
        if got is None:
            continue
        F, st = got
        if force or missing_math(F, st):
            todo.append((rid, r["story"], F, st))
    print(f"[spec_story] math: {len(todo)} stories to write over {len(records)} record(s); {jobs} agents at a time",
          flush=True)

    def one(item):
        rid, p, F, st = item
        try:
            if force:
                st.pop("math", None)
            borrow(F, st, p)
            st2 = complete_math(F, st) if missing_math(F, st) else st
            save(p, F, _norm(json.loads(json.dumps(st2))))
            c = os.path.join(CACHE, _key(F) + ".json")
            if os.path.exists(c):
                json.dump(st2, open(c + ".tmp", "w"), indent=1, default=str); os.replace(c + ".tmp", c)
            return f"{'ok   ' if not missing_math(F, st2) else 'FAIL '} {rid}/{os.path.basename(p)[:4]}"
        except Exception as e:                                    # noqa: BLE001
            return f"FAIL  {rid}/{os.path.basename(p)[:4]}  {type(e).__name__}: {str(e)[:160]}"
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        for line in ex.map(one, todo):
            print(line, flush=True)
    print(f"[spec_story] math done in {time.time() - t0:.0f} s", flush=True)


def check_links(records: list[str], verbose: bool = False) -> dict:
    """Every link the plexus pane draws, for every saved step of each record, VERIFIED against the line it opens, by the
    rule of `bullet_links` (a step on its `def forward(`, a hook on its `def`, a learnable on the forward line that reads
    it, an entity on its spec `state:` or its `*_schema` def, a set or relation on its spec entry, a maker on its
    `np.savez`). A bullet with no link fails. Prints the rate per record and the failures; returns the counts."""
    tot, ok, seen, fails, per = 0, 0, set(), {}, {}
    for rid, r in _walk(records):
        if r["spec"] in seen:
            continue
        seen.add(r["spec"])
        got = load_saved(r["story"])
        if got is None:
            continue
        F, _st = got
        try:
            s = _summary().summarise(r["spec"])
        except Exception:                                        # noqa: BLE001
            continue
        exp = {}
        BL = bullet_links(s, F, expect=exp)
        for key, url in BL.items():
            ln, rx = exp.get(key, ("", r"^$"))
            good = bool(url) and bool(ln) and re.search(rx, ln) is not None
            tot += 1; ok += good
            a = per.setdefault(rid, [0, 0]); a[0] += 1; a[1] += good
            if not good:
                code = next((x["code"] for x in F.get("steps", []) if x["id"] == key[1]), key[1])
                fails.setdefault(f"{key[0]} {code}", []).append(
                    f"{rid}/{os.path.basename(r['story'])[:4]} -> {url or 'NO LINK'}  [{(ln or '').strip()[:70]}]")
    for rid, (n, k) in sorted(per.items()):
        print(f"  {rid:45s} {k:4d}/{n:<4d} links on the right line", flush=True)
    print(f"[spec_story] links: {ok}/{tot} on the right line ({100 * ok / max(tot, 1):.0f} %) over {len(seen)} distinct specs",
          flush=True)
    for what, where in sorted(fails.items(), key=lambda kv: -len(kv[1])):
        print(f"  FAIL x{len(where):<3d} {what}: {where[0]}" + ("" if not verbose else "\n      " + "\n      ".join(where[1:6])))
    return {"total": tot, "ok": ok, "fails": fails}


def repair_symbols(records: list[str], jobs: int) -> None:
    """Every SAVED story of each record (`<record>/story/NNNN.json`) whose equation steps lack their legends:
    ask for them and re-save, `jobs` agents at once. The scan is cheap; only the stories that lack something
    cost a call."""
    from concurrent.futures import ThreadPoolExecutor
    import glob
    t0 = time.time()
    todo = []
    for rid in records:
        for p in sorted(glob.glob(os.path.join(ROOT, rid, "story", "[0-9][0-9][0-9][0-9].json"))):
            got = load_saved(p)
            if got is None:
                continue
            F, st = got
            if missing_symbols(F, st):
                todo.append((rid, p, F, st))
    print(f"[spec_story] {len(todo)} saved stories lack a legend, over {len(records)} record(s); "
          f"{jobs} agents at a time", flush=True)

    def one(item):
        rid, p, F, st = item
        try:
            st2 = complete_symbols(F, st)
            save(p, F, _norm(json.loads(json.dumps(st2))))
            c = os.path.join(CACHE, _key(F) + ".json")
            if os.path.exists(c):
                json.dump(st2, open(c + ".tmp", "w"), indent=1, default=str); os.replace(c + ".tmp", c)
            left = missing_symbols(F, st2)
            return f"{'ok   ' if not left else 'PART '} {rid}/{os.path.basename(p)[:4]}" + \
                   (f"  still missing {left}" if left else "")
        except Exception as e:                                    # noqa: BLE001
            return f"FAIL  {rid}/{os.path.basename(p)[:4]}  {type(e).__name__}: {str(e)[:160]}"
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        for line in ex.map(one, todo):
            print(line, flush=True)
    print(f"[spec_story] repair done in {time.time() - t0:.0f} s", flush=True)


def batch(records: list[str], last: int | None, jobs: int, force: bool) -> None:
    """Write and save the story of every step (or the last `last`) of each record, `jobs` agents at once.
    The facts of every step are read first; steps with the same facts (the before/after shots of one run)
    share one agent call."""
    from concurrent.futures import ThreadPoolExecutor
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from plexus.gui import watch as W
    t0 = time.time()
    steps, groups = [], {}
    for rid in records:
        fs = W.shots(os.path.join(ROOT, rid))
        picked, seen_out = [], set()
        for png in reversed(fs):                                  # newest first, so --last counts steps that tell
            r = resolve(png)
            if r is None or r["story"] in seen_out:
                continue
            seen_out.add(r["story"]); picked.append(r)
            if last and len(picked) >= last:
                break
        for r in reversed(picked):
            spec, why, out, png = r["spec"], r["why"], r["story"], r["png"]
            tag = f"{rid}/{os.path.basename(out)[:4]}"
            if os.path.exists(out) and not force:
                continue
            try:
                F = facts(_summary().summarise(spec), context=W._why_context(why),
                          diff=spec_diff(previous_spec(png), spec))
            except Exception as e:                                # noqa: BLE001
                print(f"SKIP  {tag}  facts: {type(e).__name__}: {str(e)[:160]}", flush=True)
                continue
            steps.append((tag, out, F))
            groups.setdefault(_key(F), F)
    print(f"[spec_story] {len(steps)} steps, {len(groups)} distinct stories, {len(records)} record(s), "
          f"{jobs} agents at a time (facts read in {time.time() - t0:.0f} s)", flush=True)

    def one(F):
        try:
            return _key(F), cached(F) or _norm(json.loads(json.dumps(generate(F)))), None
        except Exception as e:                                    # noqa: BLE001
            return _key(F), None, f"{type(e).__name__}: {str(e)[:160]}"
    done = {}
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        for k, st, err in ex.map(one, list(groups.values())):
            done[k] = (st, err)
    for tag, out, F in steps:
        st, err = done.get(_key(F), (None, "not run"))
        if st is None:
            print(f"FAIL  {tag}  {err}", flush=True)
            continue
        save(out, F, st)
        print(f"ok    {tag}  {st['headline']}", flush=True)
    print(f"[spec_story] done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", nargs="?")
    ap.add_argument("--batch", nargs="*", default=None, help="record ids (e.g. experiments/exp22_water_balloon); "
                    "with no id, every record the watcher lists")
    ap.add_argument("--last", type=int, default=None, help="only the last N steps of each record")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--repair-symbols", action="store_true",
                    help="with --batch: complete the equation legends of every saved story that lacks them")
    ap.add_argument("--refresh-equations", action="store_true",
                    help="with --batch: re-read every saved story's equations from the live registry (no agent)")
    ap.add_argument("--check-links", action="store_true",
                    help="with --batch: verify every plexus-pane link lands on the line that computes / declares it")
    ap.add_argument("--math", action="store_true",
                    help="with --batch: write the MATH pane (one notation, one legend) of every saved story lacking it")
    ap.add_argument("--facts", action="store_true")
    ap.add_argument("--context", default="", help="the step's why, as the watcher passes it")
    a = ap.parse_args()
    if a.batch is not None:
        if not a.batch:
            sys.path.insert(0, os.path.join(ROOT, "src"))
            from plexus.gui import watch as _W
            a.batch = [r["id"] for r in _W.records()]
        if a.check_links:
            check_links(a.batch); sys.exit(0)
        if a.refresh_equations:
            refresh_equations(a.batch); sys.exit(0)
        if a.repair_symbols:
            repair_symbols(a.batch, a.jobs); sys.exit(0)
        if a.math:
            write_math(a.batch, a.jobs, a.force); sys.exit(0)
        batch(a.batch, a.last, a.jobs, a.force); sys.exit(0)
    F = facts(_summary().summarise(a.spec), context=a.context)
    if a.facts:
        print(_facts_text(F)); sys.exit(0)
    st = cached(F) or generate(F)
    print(st["headline"]); print(st["story"]); print()
    by = {x["id"]: x for x in F["steps"]}
    for x in ordered_steps(F, st):
        i = x["id"]
        print(f"- [{x['kind']}] {st['steps'][i]['name']}{' x' + str(x['count']) if x['count'] > 1 else ''}"
              f" -- {st['steps'][i]['what']}  ({WHEN_SHORT.get(x['when'], x['when'])}; {x['code']})")
    print(f"\n({st.get('model')}, {st.get('seconds')} s)")
