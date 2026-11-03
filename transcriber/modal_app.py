"""The transcriber service on Modal (serverless GPU: billed per second while working, nothing when idle).

  pip install modal && modal setup                       # once, on the deploying machine
  modal secret create shortcut-transcriber HF_TOKEN=... TRANSCRIBER_TOKEN=...   # a person runs this
  cd backend && modal deploy transcriber/modal_app.py

Endpoints (what the worker's lib/referenceTranscript.ts calls; Authorization: Bearer TRANSCRIBER_TOKEN):
  POST /submit {"audio_url", "language"?}  -> {"call_id"}     at once
  GET  /result?call_id=...                 -> 202 still working | 200 the transcript | 500 {"error"}
The worker's TRANSCRIBER_URL is the web endpoint's URL that `modal deploy` prints.
"""
import os

import modal

app = modal.App("shortcut-transcriber")
secret = modal.Secret.from_name("shortcut-transcriber")


def _download_models():
    # Into the image at build time, so a cold start does not download ~6 GB
    from huggingface_hub import snapshot_download
    from transcriber.pipeline import TEXT_MODEL, TEXT_REVISION, TEXT_SKIP, TIME_MODELS
    snapshot_download(TEXT_MODEL, revision=TEXT_REVISION, ignore_patterns=TEXT_SKIP)
    for name, revision in TIME_MODELS.values():
        snapshot_download(name, revision=revision, ignore_patterns=["*.msgpack", "*.h5"])


gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install_from_requirements("transcriber/requirements.txt")
    .add_local_python_source("transcriber", copy=True)
    .run_function(_download_models, secrets=[secret])
)
web_image = modal.Image.debian_slim(python_version="3.12").pip_install("fastapi[standard]")


# cpu=4: cutting at pauses, audio decoding and the timing fit run on the CPU (the default slice starved them).
# A10G when no L4 is free (3 Nov: an L4 queue held a cold start for 7+ min); both 24 GB, bf16.
@app.cls(gpu=["L4", "A10G"], cpu=4.0, image=gpu_image, secrets=[secret], timeout=60 * 60, scaledown_window=120, max_containers=20)
class Worker:
    @modal.enter()
    def load(self):
        import time
        t0 = time.time()
        print(f"container up at {time.strftime('%H:%M:%S')}", flush=True)
        from transcriber.pipeline import Transcriber
        self.t = Transcriber("cuda")
        print(f"models loaded in {time.time() - t0:.0f} s", flush=True)

    @modal.method()
    def run(self, audio_url: str, language: str | None = None) -> dict:
        from transcriber.pipeline import load_audio
        return self.t.transcribe(load_audio(audio_url), language)


@app.function(image=web_image, secrets=[secret])
@modal.asgi_app()
def web():
    from fastapi import FastAPI, Header, HTTPException
    from fastapi.responses import JSONResponse

    api = FastAPI()

    def check(authorization: str | None):
        token = os.environ.get("TRANSCRIBER_TOKEN")
        if not token or authorization != f"Bearer {token}":
            raise HTTPException(status_code=401, detail="unauthorised")

    @api.post("/submit")
    def submit(body: dict, authorization: str | None = Header(default=None)):
        check(authorization)
        url = body.get("audio_url")
        if not isinstance(url, str) or not url.startswith(("https://", "http://")):
            raise HTTPException(status_code=400, detail="audio_url is required")
        call = Worker().run.spawn(url, body.get("language"))
        return {"call_id": call.object_id}

    @api.get("/result")
    def result(call_id: str, authorization: str | None = Header(default=None)):
        check(authorization)
        try:
            out = modal.FunctionCall.from_id(call_id).get(timeout=0)
        except TimeoutError:
            return JSONResponse({"status": "running"}, status_code=202)
        except Exception as e:  # the transcription failed, or the ticket is unknown or expired
            return JSONResponse({"error": str(e)[:500]}, status_code=500)
        return out

    return api
