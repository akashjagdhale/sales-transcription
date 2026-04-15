# Sales Transcription

Cloud-powered speech-to-text transcription for sales calls, meetings, and interviews. Optimised for Indian languages and English code-switching using the [Sarvam AI](https://www.sarvam.ai/) Batch API.

---

## Features

- **Drag-and-drop web UI** — upload up to 20 files at once, watch real-time progress
- **Indian language support** — Hindi, Tamil, Telugu, Kannada, and more mixed with English
- **Large file support** — up to 200 MB and ~1 hour of audio per file
- **Native speaker diarization** — up to 8 speakers identified at the audio-signal level
- **AI-powered analysis** — optional post-transcription summary, keyword detection, and speaker labelling via Anthropic, OpenAI, or Gemini
- **CLI batch mode** — script large folders without the UI
- **Auto-deploys** — GitHub Actions pushes to a Docker container on every merge to `main`

---

## Supported Formats

`.mp3` `.wav` `.m4a` `.webm` `.ogg` `.flac` `.aac` `.wma` `.aiff` `.opus` `.amr` `.mp4`

---

## Getting Started

### Prerequisites

- Python 3.10+
- A [Sarvam AI](https://www.sarvam.ai/) account and API key
- (Optional) An API key from Anthropic, OpenAI, or Google for AI analysis

### Run the web UI

```bash
./start.sh
```

Opens at `http://localhost:5001`. Enter your Sarvam API key in **Settings → Transcription API**, then drag and drop your audio files.

### Run the CLI

```bash
# Transcribe all audio files in a folder
python3 transcribe.py --input /path/to/audio --output /path/to/transcripts

# With speaker diarization (3 speakers)
python3 transcribe.py --input /path/to/audio --output /path/to/transcripts --diarization --num-speakers 3
```

Already-transcribed files are skipped automatically on re-runs.

---

## AI Analysis (Optional)

After transcription, open the **Analysis** panel in the UI and provide an API key for one of the supported providers:

| Provider | What you need |
|---|---|
| Anthropic | API key from [console.anthropic.com](https://console.anthropic.com) |
| OpenAI | API key from [platform.openai.com](https://platform.openai.com) |
| Gemini | API key from [aistudio.google.com](https://aistudio.google.com) |

Analysis returns a **call summary**, **keyword detection** (competitors, pricing, pain points, products, topics), and a **speaker view** with labelled turns.

API keys are sent per-request and never stored server-side.

---

## Speaker Diarization

Enable in **Settings → Diarization**. Set the number of speakers (2–8). Results appear in the **Speaker View** tab with timestamps and per-speaker transcript segments.

Native diarization runs at the audio-signal level via Sarvam — no AI analysis step needed.

---

## Deployment (Docker)

```bash
# Build and run
docker compose up -d --build
```

Set these environment variables (or add them to a `.env` file):

```
SARVAM_API_KEY=your_key_here   # optional — can also be set per-request in the UI
PORT=5010
TRAEFIK_HOST=your-domain.com   # only needed if deploying behind Traefik
```

GitHub Actions auto-deploys to a configured VPS on every push to `main` — set the `SSH_HOST`, `SSH_USER`, `SSH_KEY`, and `SSH_PORT` secrets in your repo and adjust the deploy path inside [`.github/workflows/deploy.yml`](.github/workflows/deploy.yml).

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/upload` | `POST` | Upload 1–20 files, returns `job_id` |
| `/api/jobs/<id>/stream` | `GET` | SSE log stream — real-time progress |
| `/api/jobs/<id>/status` | `GET` | Job status and metadata |
| `/api/jobs/<id>/transcript` | `GET` | Combined transcript (`?file=name` for one file, `?download=1` to save) |
| `/api/jobs/<id>/diarization` | `GET` | Speaker diarization data (`?file=name` for one file) |
| `/api/models` | `POST` | List available models for a given AI provider |
| `/api/analyze` | `POST` | Run AI analysis on a completed transcript |

---

## Project Structure

```
transcribe.py       Core transcription pipeline (Sarvam AI Batch API)
app.py              Flask web server — upload, SSE streaming, AI analysis
templates/          index.html — drag-and-drop UI (Tailwind CSS)
start.sh            Runs the web UI via uv (auto-installs dependencies)
Dockerfile          Container definition
docker-compose.yml  Compose config for deployment
```

---

## License

MIT
