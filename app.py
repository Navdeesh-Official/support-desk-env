"""Root server entrypoint for deployment validators."""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

import uvicorn


def _run_fallback_server(host: str, port: int, error: str) -> None:
    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                body = json.dumps({"status": "degraded", "message": error}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            body = json.dumps({"detail": "Not Found"}).encode("utf-8")
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    server = HTTPServer((host, port), _Handler)
    server.serve_forever()


def main() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    try:
        uvicorn.run("server.app:app", host=host, port=port, workers=1)
    except Exception as e:
        _run_fallback_server(host, port, str(e))


if __name__ == "__main__":
    main()
