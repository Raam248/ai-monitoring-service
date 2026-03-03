# AI Meeting Monitor - Backend

Production-ready multimodal hate speech and emotion detection service.

## Features

- **Speech-to-Text**: Local Whisper ASR (no API costs)
- **Hate Speech Detection**: toxic-bert + violence keyword boosting
- **Emotion Detection**: FER facial expression recognition (optional)
- **Multimodal Fusion**: Combines audio + video for smarter alerts
- **Configurable**: Environment variables for all settings
- **Production-ready**: Proper logging, error handling, health checks

## Requirements

- Python 3.10+
- ffmpeg (must be in PATH)
- ~4GB RAM for models

## Installation

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Install ffmpeg (Windows)
```powershell
winget install ffmpeg
```

## Running the Server

```bash
# Development
python main.py

# Production
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Configuration (Environment Variables)

```bash
# Feature flags
ENABLE_VIDEO=true       # Set false if no webcam needed
ENABLE_AUDIO=true

# Model settings
WHISPER_MODEL=base      # tiny, base, small, medium, large

# Detection thresholds
HATE_THRESHOLD=0.5
DANGER_THRESHOLD=0.7

# Processing
MIN_AUDIO_SIZE=5000
AUDIO_BUFFER_CHUNKS=3
FUSION_WINDOW_SECONDS=30
```

## API Endpoints

### WebSocket
- `ws://host:8000/ws/monitor` - Combined audio + video

### REST
- `GET /health` - Health check with model status
- `GET /test?text=...` - Test hate detection on text

## Response Format

```json
{
  "type": "audio_result",
  "transcription": "detected speech",
  "detection": {
    "label": "toxic",
    "score": 0.85,
    "level": "danger",
    "keywords": ["kill"]
  },
  "fusion": {
    "level": "critical",
    "reason": "Dangerous speech + angry expression"
  },
  "timestamp": "2024-01-01T12:00:00"
}
```

## Alert Levels

- `safe` - No threats
- `warning` - Score >= 0.5
- `danger` - Score >= 0.7 or violence keywords
- `critical` - Multimodal fusion (audio + video combined)

## Multimodal Fusion Logic

- **CRITICAL**: Dangerous speech + angry face
- **CRITICAL**: 2+ danger events in 30 seconds
- **DANGER**: Warning speech + angry face
- **DANGER**: 3+ warnings in 30 seconds
- **WARNING**: Strong angry expression (>70%)
