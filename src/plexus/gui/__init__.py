"""Plexus GUI server: the watcher and the scene API.

    python -m plexus.gui [--tab bio|material] [--port N]   # serves /watch and /api/*

`/watch` is the read-only build history and live views (`watch.py`); `/api/scene/*` and `/api/tab/<tab>/build` are the scene routes that build sessions drive (`tools/gui_drive.py`). Backend is stdlib-only (`http.server`).
"""

from plexus.gui.server import serve

__all__ = ["serve", "launch"]


# launch: Start the server (blocking) -> None.
def launch(host="127.0.0.1", port=8765, open_browser=True):
    from plexus.gui.__main__ import main
    argv = ["--host", host, "--port", str(port)]
    if not open_browser:
        argv.append("--no-browser")
    main(argv)
