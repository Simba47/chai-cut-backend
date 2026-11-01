"""The transcriber service on this machine (for checks before Modal): the same /submit and /result.

  python -m transcriber.local_server            # http://127.0.0.1:8790
  then the worker:  REFERENCE_TRANSCRIBER=shadow TRANSCRIBER_URL=http://127.0.0.1:8790
One transcription at a time; the rest wait their turn.
"""
import json
import queue
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from transcriber.pipeline import Transcriber, load_audio

PORT = 8790
results: dict[str, dict] = {}
todo: "queue.Queue[tuple[str, str, str | None]]" = queue.Queue()


def worker(t: Transcriber):
    while True:
        call_id, url, lang = todo.get()
        try:
            results[call_id] = {"done": True, "out": t.transcribe(load_audio(url), lang)}
        except Exception as e:  # reported to the caller as a failed transcript
            results[call_id] = {"done": True, "error": str(e)[:500]}
        print(f"[transcriber] {call_id}: {'failed' if 'error' in results[call_id] else 'done'}", flush=True)


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: dict):
        data = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        if urlparse(self.path).path != "/submit":
            return self._send(404, {"error": "not found"})
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if not isinstance(body.get("audio_url"), str):
            return self._send(400, {"error": "audio_url is required"})
        call_id = "local-" + uuid.uuid4().hex[:12]
        results[call_id] = {"done": False}
        todo.put((call_id, body["audio_url"], body.get("language")))
        self._send(200, {"call_id": call_id})

    def do_GET(self):
        u = urlparse(self.path)
        if u.path != "/result":
            return self._send(404, {"error": "not found"})
        r = results.get((parse_qs(u.query).get("call_id") or [""])[0])
        if r is None:
            return self._send(404, {"error": "unknown ticket"})
        if not r["done"]:
            return self._send(202, {"status": "running"})
        return self._send(500, {"error": r["error"]}) if "error" in r else self._send(200, r["out"])


if __name__ == "__main__":
    print("[transcriber] loading models…", flush=True)
    threading.Thread(target=worker, args=(Transcriber(),), daemon=True).start()
    print(f"[transcriber] ready on http://127.0.0.1:{PORT}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
