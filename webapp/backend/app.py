"""Flask app factory + route registration."""
from __future__ import annotations

import pathlib

from flask import Flask, send_from_directory
from flask_cors import CORS

from . import api

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
STATIC_ROOT = REPO_ROOT / "webapp" / "backend" / "static"


def create_app() -> Flask:
    # static_folder=None disables Flask's built-in /static/<file> route — our SPA catch-all
    # below serves both index.html and the Vite-built assets/ directory.
    app = Flask(__name__, static_folder=None)
    CORS(app, resources={r"/api/*": {"origins": "*"}, r"/camera/*": {"origins": "*"}})
    app.register_blueprint(api.bp)

    # SPA shell — serve index.html for any route not claimed by /api or /camera, so
    # client-side routes (/cameras, /motors, …) reload correctly.
    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def spa(path: str):
        index = STATIC_ROOT / "index.html"
        if not index.is_file():
            return (
                "<h1>Frontend not built</h1>"
                "<p>Run <code>make webapp-build</code> (or <code>make webapp-dev</code>"
                " for a hot-reload dev server).</p>"
            ), 503
        # If the request points to a real file under static/, send that file.
        target = STATIC_ROOT / path
        if path and target.is_file():
            return send_from_directory(STATIC_ROOT, path)
        return send_from_directory(STATIC_ROOT, "index.html")

    return app
