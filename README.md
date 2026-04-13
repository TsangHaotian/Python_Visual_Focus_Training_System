# Python Visual Focus Training System

An interactive **web-based focus training game** powered by real-time webcam analysis.

The system uses MediaPipe face landmarks to estimate focus level (`0-3`) and dynamically adjusts game difficulty. It also displays an explainable AI strategy panel and generates an end-of-session summary.

## Features

- Real-time webcam stream with face landmark overlay (MJPEG in browser)
- Focus level estimation on a unified `0-3` scale
- Adaptive game difficulty (speed + obstacle height) based on focus trends
- Explainable AI strategy report (short/mid/long window statistics)
- Live focus curve chart (last 2 minutes)
- Session duration control and restart support
- Mercy mechanism for long unfocused periods with end-of-game summary

## Tech Stack

- Python + Flask
- OpenCV
- MediaPipe
- NumPy
- Vanilla JavaScript + HTML Canvas

## Project Structure

- `web_app.py`: Flask server, camera loop, focus state API
- `ai_strategy.py`: adaptive strategy logic and game summary generation
- `templates/index.html`: main web UI
- `static/js/app.js`: game logic, chart drawing, API polling
- `static/css/style.css`: page styling
- `rule_model/models/face_landmarker.task`: required MediaPipe model file

## Requirements

- Python 3.9+ (recommended)
- A working webcam
- Model file at: `rule_model/models/face_landmarker.task`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

From project root:

```bash
python web_app.py
```

Then open:

- [http://127.0.0.1:5000](http://127.0.0.1:5000)

## Optional CLI Arguments

You can tune performance and quality:

```bash
python web_app.py --host 127.0.0.1 --port 5000 --cam-id 0 --capture-width 960 --stream-width 720 --jpeg-quality 75
```

Common options:

- `--model`: model path (`.task`)
- `--cam-id`: camera index
- `--capture-width`: input width before inference
- `--detect-width`: width for detection workload
- `--stream-width`: MJPEG output width
- `--jpeg-quality`: stream quality (`40-95`)
- `--infer-interval`: minimum interval between inferences (seconds)

## API Endpoints

- `GET /`: main page
- `GET /video_feed`: webcam MJPEG stream
- `GET /api/state`: current focus state + AI strategy/report
- `POST /api/game_summary`: generate session summary

## Gameplay Controls

- `Space`: micro-adjust the eagle (manual upward impulse)
- `R`: restart the game
- Set target session minutes and click **Apply & Restart**

## Notes

- If the model fails to load, the page will show an error preview.
- If no face is detected, focus is treated as `0`.
- For production/public repos, do **not** hardcode API keys in source code.
