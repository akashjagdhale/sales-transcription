#!/usr/bin/env python3
"""
Cloud speech-to-text transcription pipeline using Sarvam AI Batch API.
Optimised for Indian languages + English code-switching.
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

# ── Constants ─────────────────────────────────────────────────────────────────
SARVAM_STATUS_URL  = "https://api.sarvam.ai/speech-to-text/job/v1/{job_id}/status"
SARVAM_MODEL       = "saaras:v3"
SARVAM_MODE        = "codemix"      # transcribe | translate | verbatim | translit | codemix
MAX_FILE_SIZE_BYTES = 200 * 1024 * 1024  # 200 MB practical limit
MAX_BATCH_SIZE     = 20
POLL_INTERVAL_SECS = 5
MAX_POLL_SECS      = 3600  # 1 hour hard timeout

SUPPORTED_FORMATS = {
    ".mp3", ".wav", ".m4a", ".webm", ".ogg", ".flac", ".aac", ".wma",
    ".aiff", ".opus", ".amr", ".mp4",
}


# ── Logging ───────────────────────────────────────────────────────────────────

def _default_log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ── Prerequisites ─────────────────────────────────────────────────────────────

def check_prerequisites(api_key: str = None) -> str:
    """Return the Sarvam API key or raise if missing."""
    key = (api_key or os.environ.get("SARVAM_API_KEY", "")).strip()
    if not key:
        raise RuntimeError(
            "Sarvam API key not set. "
            "Add it in Settings → Transcription API, "
            "or set the SARVAM_API_KEY environment variable."
        )
    return key


# ── Core transcription ────────────────────────────────────────────────────────

def process_batch(
    audio_files: list,
    output_dir: Path,
    language_code: str,
    api_key: str,
    with_diarization: bool = False,
    num_speakers: int = 2,
    log_fn=None,
) -> list:
    """
    Full batch pipeline via Sarvam AI:
      create_job → upload_files → start → poll → get_results → download → parse

    Returns a list of per-file result dicts:
      { filename, status, transcript, diarization, transcription_time_seconds, error }
    """
    from sarvamai import SarvamAI

    _log = log_fn or _default_log
    start_time = time.time()

    # ── Validate ──────────────────────────────────────────────────────────────
    if not audio_files:
        raise RuntimeError("No audio files provided.")
    if len(audio_files) > MAX_BATCH_SIZE:
        raise RuntimeError(
            f"Too many files ({len(audio_files)}). Sarvam batch limit is {MAX_BATCH_SIZE}."
        )
    for f in audio_files:
        f = Path(f)
        if not f.exists():
            raise RuntimeError(f"File not found: {f}")
        size_mb = f.stat().st_size / (1024 * 1024)
        if f.stat().st_size > MAX_FILE_SIZE_BYTES:
            raise RuntimeError(
                f"{f.name} is {size_mb:.1f} MB — limit is {MAX_FILE_SIZE_BYTES // (1024*1024)} MB."
            )

    n = len(audio_files)
    _log(f"Starting batch transcription of {n} file(s)…")

    # ── Create job ────────────────────────────────────────────────────────────
    client = SarvamAI(api_subscription_key=api_key)

    create_kwargs = dict(
        model=SARVAM_MODEL,
        mode=SARVAM_MODE,
        with_diarization=with_diarization,
    )
    if language_code and language_code != "auto":
        create_kwargs["language_code"] = language_code
    if with_diarization:
        create_kwargs["num_speakers"] = num_speakers

    _log("Creating batch job with Sarvam AI…")
    job = client.speech_to_text_job.create_job(**create_kwargs)
    _log(f"Batch job created (ID: {job.job_id})")

    # ── Upload files ──────────────────────────────────────────────────────────
    _log(f"Uploading {n} file(s) to Sarvam…")
    job.upload_files(file_paths=[str(f) for f in audio_files])
    _log("Upload complete. Starting transcription…")

    # ── Start processing ──────────────────────────────────────────────────────
    job.start()
    _log("Sarvam batch job started. Waiting for results…")

    # ── Poll status (custom loop so we can emit progress) ─────────────────────
    poll_start = time.time()
    last_completed = -1
    consecutive_errors = 0

    while True:
        if time.time() - poll_start > MAX_POLL_SECS:
            raise RuntimeError(
                f"Batch job timed out after {MAX_POLL_SECS // 60} minutes."
            )

        try:
            resp = requests.get(
                SARVAM_STATUS_URL.format(job_id=job.job_id),
                headers={"api-subscription-key": api_key},
                timeout=(10, 30),
            )
            consecutive_errors = 0

            if resp.status_code == 200:
                status = resp.json()
                job_state = status.get("job_state", "")
                done_count = status.get("successful_files_count", 0) or 0
                fail_count = status.get("failed_files_count", 0) or 0
                total      = status.get("total_files", n) or n

                if done_count + fail_count != last_completed:
                    last_completed = done_count + fail_count
                    if total > 1:
                        _log(
                            f"Processing: {done_count + fail_count}/{total} files done"
                            f" ({done_count} succeeded, {fail_count} failed)…"
                        )

                if job_state in ("Completed", "Failed"):
                    _log(f"Batch job {job_state.lower()}.")
                    if job_state == "Failed":
                        err_msg = status.get("error_message") or status.get("error", "")
                        if err_msg:
                            _log(f"Sarvam error: {err_msg}")
                    break

        except requests.exceptions.RequestException as exc:
            consecutive_errors += 1
            _log(f"⚠ Poll error ({consecutive_errors}): {exc}")
            if consecutive_errors >= 5:
                raise RuntimeError(
                    f"Lost contact with Sarvam AI after 5 consecutive poll failures: {exc}"
                )

        time.sleep(POLL_INTERVAL_SECS)

    # ── Get results + download outputs ────────────────────────────────────────
    file_results = job.get_file_results()
    successful   = file_results.get("successful", [])
    failed       = file_results.get("failed", [])

    if successful:
        _log(f"Downloading {len(successful)} transcript(s)…")
        job.download_outputs(output_dir=str(output_dir))
        _log("Download complete.")

    # ── Parse outputs ─────────────────────────────────────────────────────────
    results = []
    elapsed_total = time.time() - start_time

    # Build lookup: original filename → file info
    success_names = {f.get("file_name", ""): f for f in successful}
    failed_names  = {f.get("file_name", ""): f for f in failed}

    for audio_file in audio_files:
        audio_file = Path(audio_file)
        fname = audio_file.name

        if fname in failed_names:
            err = failed_names[fname].get("error_message", "Unknown error")
            _log(f"✗ {fname}: {err}")
            results.append({
                "filename":                   fname,
                "original_format":            audio_file.suffix.lower(),
                "transcription_time_seconds": None,
                "status":                     f"failed: {str(err)[:100]}",
                "transcript":                 None,
                "diarization":                None,
                "error":                      err,
                "timestamp_processed":        datetime.now().isoformat(),
            })
            continue

        # Look for downloaded JSON output file
        # Sarvam downloads as {name}.json (e.g. recording.wav.json)
        transcript_text  = None
        diarization_data = None

        for candidate_name in (f"{audio_file.name}.json", f"{audio_file.stem}.json"):
            candidate = output_dir / candidate_name
            if candidate.exists():
                try:
                    data = json.loads(candidate.read_text("utf-8"))
                    transcript_text = data.get("transcript", "").strip()

                    raw_dia = data.get("diarized_transcript")
                    if raw_dia and isinstance(raw_dia, dict):
                        entries = raw_dia.get("entries", [])
                        diarization_data = [
                            {
                                "speaker_id": e.get("speaker_id", ""),
                                "transcript": e.get("transcript", ""),
                                "start":      e.get("start_time_seconds", 0),
                                "end":        e.get("end_time_seconds", 0),
                            }
                            for e in entries
                        ]
                except (json.JSONDecodeError, OSError):
                    pass
                break

        # Also check plain .txt in case format differs
        if transcript_text is None:
            txt_candidate = output_dir / f"{audio_file.stem}.txt"
            if txt_candidate.exists():
                transcript_text = txt_candidate.read_text("utf-8").strip()

        # Write a .txt alongside for backward compatibility (app.py reads this)
        if transcript_text:
            txt_path = output_dir / f"{audio_file.stem}.txt"
            if not txt_path.exists() or txt_path.read_text("utf-8").strip() != transcript_text:
                txt_path.write_text(transcript_text, encoding="utf-8")
            _log(f"✓ {fname}")
        else:
            _log(f"⚠ {fname}: transcript was empty or not found in output")

        results.append({
            "filename":                   fname,
            "original_format":            audio_file.suffix.lower(),
            "transcription_time_seconds": round(elapsed_total, 1),
            "status":                     "success" if transcript_text else "failed: empty transcript",
            "transcript":                 transcript_text,
            "diarization":                diarization_data,
            "error":                      None if transcript_text else "Empty transcript returned",
            "timestamp_processed":        datetime.now().isoformat(),
        })

    elapsed = time.time() - start_time
    mins, secs = int(elapsed // 60), int(elapsed % 60)
    _log(f"Batch complete in {mins} min {secs} sec.")

    return results


def process_file(
    audio_file: Path,
    output_dir: Path,
    language_code: str,
    log_fn=None,
    api_key: str = None,
    with_diarization: bool = False,
    num_speakers: int = 2,
) -> dict:
    """Transcribe a single audio file via Sarvam AI Batch API."""
    api_key = check_prerequisites(api_key)
    results = process_batch(
        [audio_file],
        output_dir,
        language_code,
        api_key,
        with_diarization=with_diarization,
        num_speakers=num_speakers,
        log_fn=log_fn or _default_log,
    )
    return results[0]


# ── CLI batch mode ────────────────────────────────────────────────────────────

def _write_metadata(output_dir: Path, row: dict):
    csv_path    = output_dir / "metadata.csv"
    file_exists = csv_path.exists()
    fieldnames  = [
        "filename", "original_format",
        "transcription_time_seconds", "status", "timestamp_processed",
    ]
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def main():
    parser = argparse.ArgumentParser(
        description="Cloud speech-to-text transcription using Sarvam AI Batch API"
    )
    parser.add_argument("--input",       required=True, type=Path,
                        help="Directory containing audio recordings")
    parser.add_argument("--output",      required=True, type=Path,
                        help="Directory where transcripts will be saved")
    parser.add_argument("--language",    default="auto",
                        help="Language code e.g. hi-IN, ta-IN, en-IN (default: auto)")
    parser.add_argument("--diarization", action="store_true",
                        help="Enable speaker diarization (up to 8 speakers)")
    parser.add_argument("--num-speakers", type=int, default=2, dest="num_speakers",
                        help="Number of speakers for diarization (default: 2, max: 8)")
    args = parser.parse_args()

    if not args.input.is_dir():
        print(f"ERROR: Input directory does not exist: {args.input}", file=sys.stderr)
        sys.exit(1)

    args.output.mkdir(parents=True, exist_ok=True)

    try:
        api_key = check_prerequisites()
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    all_files = sorted(
        f for f in args.input.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_FORMATS
    )

    # Skip already-transcribed files
    audio_files = []
    skipped = 0
    for f in all_files:
        if (args.output / f"{f.stem}.txt").exists():
            _default_log(f"Already transcribed, skipping: {f.name}")
            skipped += 1
        else:
            audio_files.append(f)

    if not audio_files:
        _default_log(f"No new audio files to process (skipped {skipped})")
        sys.exit(0)

    _default_log(f"Found {len(audio_files)} file(s) to process (skipped {skipped})")

    processed = failed = 0
    failed_files = []
    total_start  = time.time()

    # Process in batches of MAX_BATCH_SIZE
    for i in range(0, len(audio_files), MAX_BATCH_SIZE):
        batch = audio_files[i : i + MAX_BATCH_SIZE]
        batch_num = i // MAX_BATCH_SIZE + 1
        total_batches = (len(audio_files) + MAX_BATCH_SIZE - 1) // MAX_BATCH_SIZE
        _default_log(f"Submitting batch {batch_num}/{total_batches} ({len(batch)} files)…")

        try:
            results = process_batch(
                batch, args.output, args.language, api_key,
                with_diarization=args.diarization,
                num_speakers=args.num_speakers,
            )
            for r in results:
                _write_metadata(args.output, r)
                if r["status"] == "success":
                    processed += 1
                else:
                    failed += 1
                    failed_files.append(r["filename"])
        except Exception as e:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{ts}] ERROR: Batch {batch_num} failed — {e}", file=sys.stderr, flush=True)
            failed += len(batch)
            failed_files.extend(f.name for f in batch)
            for f in batch:
                _write_metadata(args.output, {
                    "filename":                   f.name,
                    "original_format":            f.suffix.lower(),
                    "transcription_time_seconds": "",
                    "status":                     f"failed: {str(e)[:100]}",
                    "timestamp_processed":        datetime.now().isoformat(),
                })

    total_elapsed = time.time() - total_start
    total_mins    = int(total_elapsed // 60)
    total_secs    = int(total_elapsed % 60)

    print("\n" + "=" * 50)
    print("  TRANSCRIPTION SUMMARY")
    print("=" * 50)
    print(f"  Files processed : {processed}")
    print(f"  Files skipped   : {skipped}")
    print(f"  Files failed    : {failed}")
    print(f"  Total time      : {total_mins} min {total_secs} sec")
    if failed_files:
        print(f"  Failed files    : {', '.join(failed_files)}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
