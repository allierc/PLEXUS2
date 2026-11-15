"""`python -m plexus.gui [--tab bio|material] [--port N] [--no-browser]`.

Starts the server (the watcher at `/watch`, the scene API at `/api/*`) and, unless --no-browser, opens the watcher.
"""

from __future__ import annotations

import argparse
import sys
import threading
import webbrowser

from plexus.gui.server import serve


# main: Command line of the GUI server -> None.
def main(argv=None):
    ap = argparse.ArgumentParser(prog="plexus.gui", description="the Plexus watcher and scene server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8799, help="default 8799, so the URL can be bookmarked")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--tab", default="material", choices=("bio", "material"),
                    help="the tab the server's scene starts on")
    args = ap.parse_args(argv)

    # the server's tab, from the flag: without it the first scene request looks like a tab switch and resets the scene
    from plexus.gui import bio as _bio
    _bio.STATE["tab"] = args.tab
    httpd = serve(args.host, args.port)
    url = f"http://{args.host}:{args.port}/watch"

    print(f"  Plexus watcher  ->  {url}")
    print("  (Ctrl-C to stop)")

    if not args.no_browser:
        threading.Timer(0.6, lambda: _try_open(url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped.")
        httpd.shutdown()


# _try_open: Open `url` in a browser, silently doing nothing when headless -> None.
def _try_open(url):
    try:
        webbrowser.open(url)
    except Exception:  # noqa: BLE001 -- headless / no browser; the URL was printed
        pass


if __name__ == "__main__":
    main(sys.argv[1:])
