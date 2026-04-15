#!/usr/bin/env python3
"""Web UI for cloud speech-to-text transcription using Sarvam AI Batch API."""

import json
import os
import re
import shutil
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import transcribe as tr

from flask import Flask, Response, abort, jsonify, render_template, request

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB

SUPPORTED_FORMATS = {
    ".mp3", ".wav", ".m4a", ".webm", ".ogg", ".flac", ".aac", ".wma",
    ".aiff", ".opus", ".amr", ".mp4",
}
UPLOAD_BASE = PROJECT_ROOT / "uploads_temp"
OUTPUT_BASE = PROJECT_ROOT / "web_output"
JOB_TTL = 2 * 60 * 60
MAX_FILES_PER_JOB = 20

# Clean stale uploads on startup (output preserved for TTL-based cleanup daemon)
if UPLOAD_BASE.exists():
    shutil.rmtree(UPLOAD_BASE, ignore_errors=True)
UPLOAD_BASE.mkdir(exist_ok=True)
OUTPUT_BASE.mkdir(exist_ok=True)

JOBS: dict = {}
JOBS_LOCK = threading.Lock()
MAX_CONCURRENT_JOBS = 3  # Sarvam does the compute; we just poll — safe to run a few in parallel
WORKER_SEMAPHORE = threading.Semaphore(MAX_CONCURRENT_JOBS)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
def api_upload():
    # Accept "files" (multi) or "file" (single, backward compat)
    files = request.files.getlist("files")
    if not files or (len(files) == 1 and not files[0].filename):
        single = request.files.get("file")
        if single:
            files = [single]
        else:
            return jsonify({"error": "No files provided"}), 400

    sarvam_api_key  = (request.form.get("sarvam_api_key", "") or "").strip()
    language        = (request.form.get("language", "auto") or "auto").strip()
    with_diarization = request.form.get("with_diarization", "false").lower() == "true"
    num_speakers    = int(request.form.get("num_speakers", "2") or "2")

    if not sarvam_api_key:
        return jsonify({
            "error": "Sarvam API key is required. Add it in Settings → Transcription API."
        }), 400

    if len(files) > MAX_FILES_PER_JOB:
        return jsonify({
            "error": f"Too many files. Maximum {MAX_FILES_PER_JOB} files per batch."
        }), 400

    # Reject if this API key already has an active (queued/running) job
    with JOBS_LOCK:
        active_for_key = sum(
            1 for j in JOBS.values()
            if j["sarvam_api_key"] == sarvam_api_key and j["status"] in ("queued", "running")
        )
    if active_for_key >= 2:
        return jsonify({
            "error": "You already have 2 jobs in progress. Wait for one to finish before submitting another."
        }), 429

    # Guard against disk exhaustion (need headroom for uploads + outputs)
    disk = shutil.disk_usage(PROJECT_ROOT)
    if disk.free < 500 * 1024 * 1024:  # 500 MB minimum free
        return jsonify({
            "error": "Server disk space is low. Please try again later."
        }), 503

    # Validate extensions
    for f in files:
        ext = Path(f.filename).suffix.lower()
        if ext not in SUPPORTED_FORMATS:
            return jsonify({
                "error": f"Unsupported format '{ext}' in '{f.filename}'. "
                         f"Supported: {', '.join(sorted(SUPPORTED_FORMATS))}"
            }), 400

    job_id     = str(uuid.uuid4())
    upload_dir = UPLOAD_BASE / job_id
    output_dir = OUTPUT_BASE / job_id
    upload_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    saved_names = []
    for f in files:
        safe_name = Path(f.filename).name
        f.save(str(upload_dir / safe_name))
        saved_names.append(safe_name)

    job = {
        "job_id":          job_id,
        "status":          "queued",
        # Multi-file fields
        "filenames":       saved_names,
        "file_count":      len(saved_names),
        # Single-file compat (used by transcript/status endpoints)
        "filename":        saved_names[0] if len(saved_names) == 1 else f"{len(saved_names)} files",
        "language":        language,
        "sarvam_api_key":  sarvam_api_key,
        "with_diarization": with_diarization,
        "num_speakers":    max(2, min(8, num_speakers)),
        "upload_dir":      str(upload_dir),
        "output_dir":      str(output_dir),
        "log_lines":       [],
        "log_event":       threading.Event(),
        # Results
        "transcript":      None,          # combined text (all files joined)
        "transcripts":     {},            # filename -> text
        "diarization":     {},            # filename -> list of speaker turns
        "error":           None,
        "created_at":      time.time(),
        "finished_at":     None,
    }

    with JOBS_LOCK:
        JOBS[job_id] = job

    threading.Thread(target=_run_job, args=(job_id,), daemon=True).start()

    return jsonify({
        "job_id":     job_id,
        "filenames":  saved_names,
        "file_count": len(saved_names),
        # backward compat
        "filename":   job["filename"],
    }), 202


@app.route("/api/jobs/<job_id>/stream")
def api_stream(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        return jsonify({"error": "Job not found"}), 404

    def generate():
        cursor = 0
        ev     = job["log_event"]
        while True:
            lines = job["log_lines"]
            while cursor < len(lines):
                yield f"data: {lines[cursor]}\n\n"
                cursor += 1

            if job["status"] in ("done", "error"):
                yield _status_event(job)
                return

            ev.clear()
            if cursor < len(job["log_lines"]):
                continue
            if not ev.wait(timeout=25):
                yield ": heartbeat\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/jobs/<job_id>/status")
def api_status(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        return jsonify({"error": "Job not found"}), 404
    return jsonify({
        "job_id":          job["job_id"],
        "status":          job["status"],
        # multi-file fields
        "filenames":       job.get("filenames", [job.get("filename", "")]),
        "file_count":      job.get("file_count", 1),
        # backward compat
        "filename":        job.get("filename", ""),
        "has_transcript":  bool(job.get("transcript") or job.get("transcripts")),
        "has_diarization": bool(job.get("diarization")),
        "error":           job["error"],
        "log_lines":       list(job["log_lines"]),
    })


@app.route("/api/jobs/<job_id>/transcript")
def api_transcript(job_id):
    job = _get_job_or_404(job_id)
    filename = request.args.get("file")

    # Per-file transcript
    if filename and filename in job.get("transcripts", {}):
        text = job["transcripts"][filename]
        stem = _safe_filename(Path(filename).stem)
    elif job.get("transcript"):
        text = job["transcript"]
        stem = _safe_filename(
            Path(job["filenames"][0]).stem if job.get("filenames") else "transcript"
        )
    else:
        return jsonify({"error": "Transcript not ready"}), 404

    download = request.args.get("download") == "1"
    return Response(
        text,
        mimetype="text/plain",
        headers={"Content-Disposition":
                 f'attachment; filename="{stem}.txt"'} if download else {},
    )


@app.route("/api/jobs/<job_id>/diarization")
def api_diarization(job_id):
    job = _get_job_or_404(job_id)
    filename = request.args.get("file")

    dia = job.get("diarization", {})
    if not dia:
        return jsonify({"error": "No diarization data available"}), 404

    if filename and filename in dia:
        return jsonify({"diarization": dia[filename]})

    return jsonify({"diarization": dia})


# =============================================================================
# AI Analysis
# =============================================================================

ANALYSIS_PROMPT = """You are a sales call analysis expert. Analyze this transcript and return a JSON object with exactly these three keys:

1. "summary": An array of 3-5 concise bullet point strings summarizing the key points of the call.

2. "speakers": An array of up to 15 key exchanges, each with "speaker" (string: "Sales Rep" or "Prospect") and "text" (string: a concise summary of what they said in that turn, not verbatim). Focus on the most important moments. If you can't clearly distinguish speakers, use "Speaker 1" and "Speaker 2".

3. "keywords": An object with these category keys, each containing an array of strings:
   - "competitors": Any competitor names or products mentioned
   - "pricing": Pricing-related terms, amounts, or discussions
   - "pain_points": Customer pain points, challenges, or problems mentioned
   - "products": Product or feature names discussed
   - "topics": Other significant topics or themes

Return ONLY valid JSON, no markdown fences, no explanation.

TRANSCRIPT:
"""


@app.route("/api/models", methods=["POST"])
def api_models():
    data     = request.get_json(silent=True) or {}
    provider = data.get("provider", "").strip().lower()
    api_key  = data.get("api_key", "").strip()

    if not provider or not api_key:
        return jsonify({"error": "Provider and API key are required"}), 400

    try:
        if provider == "claude":
            models = _list_models_claude(api_key)
        elif provider == "openai":
            models = _list_models_openai(api_key)
        elif provider == "gemini":
            models = _list_models_gemini(api_key)
        else:
            return jsonify({"error": f"Unknown provider: {provider}"}), 400
        return jsonify({"models": models})
    except Exception as exc:
        return jsonify({"error": f"Failed to fetch models: {exc}"}), 400


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    data     = request.get_json(silent=True) or {}
    job_id   = data.get("job_id",   "").strip()
    provider = data.get("provider", "").strip().lower()
    api_key  = data.get("api_key",  "").strip()
    model    = data.get("model",    "").strip()
    filename = data.get("filename", "").strip()  # optional: analyze a specific file

    if not all([job_id, provider, api_key, model]):
        return jsonify({"error": "job_id, provider, api_key, and model are required"}), 400

    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        return jsonify({"error": "Job not found"}), 404

    # Pick the right transcript to analyze
    if filename and filename in job.get("transcripts", {}):
        transcript_text = job["transcripts"][filename]
    elif job.get("transcript"):
        transcript_text = job["transcript"]
    else:
        return jsonify({"error": "Transcript not ready"}), 400

    prompt = ANALYSIS_PROMPT + transcript_text

    try:
        if provider == "claude":
            raw = _call_claude(api_key, model, prompt)
        elif provider == "openai":
            raw = _call_openai(api_key, model, prompt)
        elif provider == "gemini":
            raw = _call_gemini(api_key, model, prompt)
        else:
            return jsonify({"error": f"Unknown provider: {provider}"}), 400

        result       = _extract_json(raw)
        job["analysis"] = result
        return jsonify({"analysis": result})

    except json.JSONDecodeError as exc:
        return jsonify({"error": f"AI returned invalid JSON: {exc}", "raw": raw[:500]}), 502
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": f"Analysis failed: {exc}"}), 502


def _list_models_claude(api_key):
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    resp   = client.models.list(limit=100)
    models = [{"id": m.id, "name": m.display_name or m.id} for m in resp.data]
    models.sort(key=lambda x: x["name"])
    return models

def _call_claude(api_key, model, prompt):
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    msg    = client.messages.create(
        model=model, max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text

def _list_models_openai(api_key):
    import openai
    client = openai.OpenAI(api_key=api_key)
    resp   = client.models.list()
    models = []
    for m in resp.data:
        mid = m.id
        if mid.startswith("gpt-") or mid.startswith("o") or mid.startswith("chatgpt-"):
            models.append({"id": mid, "name": mid})
    models.sort(key=lambda x: x["name"])
    return models

def _call_openai(api_key, model, prompt):
    import openai
    client = openai.OpenAI(api_key=api_key)
    resp   = client.chat.completions.create(
        model=model, max_tokens=16384,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content

def _list_models_gemini(api_key):
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    models = []
    for m in genai.list_models():
        if "generateContent" in (m.supported_generation_methods or []):
            display = m.display_name or m.name.replace("models/", "")
            models.append({"id": m.name, "name": display})
    models.sort(key=lambda x: x["name"])
    return models

def _call_gemini(api_key, model, prompt):
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    m    = genai.GenerativeModel(model)
    resp = m.generate_content(prompt)
    return resp.text

def _extract_json(text):
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip()
    cleaned = re.sub(r"```\s*$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end   = text.rfind("}")
    if start != -1 and end > start:
        return json.loads(text[start : end + 1])
    raise json.JSONDecodeError("No JSON object found in response", text, 0)


# =============================================================================
# Worker thread
# =============================================================================

def _run_job(job_id: str):
    job = JOBS[job_id]
    ev  = job["log_event"]

    def emit(msg: str):
        ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        job["log_lines"].append(line)
        ev.set()

    # Try to grab a worker slot immediately; if full, tell the user they're queued
    acquired = WORKER_SEMAPHORE.acquire(blocking=False)
    if not acquired:
        with JOBS_LOCK:
            running = sum(1 for j in JOBS.values() if j["status"] == "running")
        emit(f"⏳ Server is processing {running} job(s). You're in the queue — hang tight…")
        WORKER_SEMAPHORE.acquire()  # block until a slot opens
        emit("🟢 Your turn! Starting transcription…")

    try:
        job["status"] = "running"

        input_dir  = Path(job["upload_dir"])
        output_dir = Path(job["output_dir"])

        audio_files = sorted(
            f for f in input_dir.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_FORMATS
        )

        if not audio_files:
            raise RuntimeError("No valid audio files found in upload.")

        results = tr.process_batch(
            audio_files,
            output_dir,
            job["language"],
            api_key=job["sarvam_api_key"],
            with_diarization=job["with_diarization"],
            num_speakers=job["num_speakers"],
            log_fn=emit,
        )

        # Store per-file results
        for r in results:
            fname = r["filename"]
            if r["status"] == "success" and r.get("transcript"):
                job["transcripts"][fname] = r["transcript"]
                if r.get("diarization"):
                    job["diarization"][fname] = r["diarization"]

        # Combined transcript (all files joined with separator)
        if job["transcripts"]:
            if len(job["transcripts"]) == 1:
                job["transcript"] = next(iter(job["transcripts"].values()))
            else:
                parts = []
                for fname, text in job["transcripts"].items():
                    parts.append(f"=== {fname} ===\n{text}")
                job["transcript"] = "\n\n".join(parts)

        failed_count = sum(1 for r in results if r["status"] != "success")

        if failed_count == len(results):
            raise RuntimeError("All files failed to transcribe.")
        elif failed_count > 0:
            emit(f"⚠ {failed_count}/{len(results)} file(s) failed to transcribe.")

        emit("✅ Transcription complete!")
        job["finished_at"] = time.time()
        job["status"]      = "done"

    except Exception as exc:
        emit(f"❌ Error: {exc}")
        job["error"]       = str(exc)
        job["finished_at"] = time.time()
        job["status"]      = "error"

    finally:
        WORKER_SEMAPHORE.release()
        ev.set()
        shutil.rmtree(job["upload_dir"], ignore_errors=True)


# =============================================================================
# Cleanup daemon
# =============================================================================

def _cleanup_daemon():
    while True:
        time.sleep(600)
        now = time.time()
        with JOBS_LOCK:
            expired = [
                jid for jid, j in JOBS.items()
                if j["finished_at"] and (now - j["finished_at"]) > JOB_TTL
            ]
            for jid in expired:
                j = JOBS.pop(jid)
                shutil.rmtree(j["output_dir"], ignore_errors=True)

threading.Thread(target=_cleanup_daemon, daemon=True).start()


# =============================================================================
# Helpers
# =============================================================================

def _safe_filename(name: str) -> str:
    return name.replace('"', "'").replace("\\", "_")

def _get_job_or_404(job_id: str) -> dict:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        abort(404, description="Job not found")
    return job

def _status_event(job: dict) -> str:
    payload = json.dumps({
        "status":          job["status"],
        "error":           job["error"],
        "has_diarization": bool(job.get("diarization")),
        "filenames":       job.get("filenames", []),
        "file_count":      job.get("file_count", 1),
    })
    return f"event: status\ndata: {payload}\n\n"


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    _host = os.environ.get("FLASK_HOST", "127.0.0.1")
    _port = int(os.environ.get("PORT", 5001))
    print("\n🎙️  Sales Transcription Cloud — Web UI")
    print("=" * 42)
    print(f"  Open in your browser: http://localhost:{_port}")
    print("  Press Ctrl+C to stop\n")
    app.run(host=_host, port=_port, debug=False, threaded=True)
