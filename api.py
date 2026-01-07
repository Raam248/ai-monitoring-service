from fastapi import FastAPI, WebSocket
import whisper
import tempfile
import os
import datetime
import subprocess
from transformers import pipeline

app = FastAPI()

# ===============================
# CONFIG
# ===============================

HATE_THRESHOLD = 0.5
MIN_AUDIO_SIZE = 10000

# ===============================
# GLOBAL STATE (IMPORTANT)
# ===============================

is_processing = False   # 🔒 prevents backlog

# ===============================
# Load models ONCE at startup
# ===============================

model = whisper.load_model("base")
print("✅ Whisper model loaded")

hate_classifier = pipeline(
    "text-classification",
    model="unitary/toxic-bert",
    top_k=None
)
print("✅ Hate speech model loaded")

# ===============================
# Helper: Convert WEBM → WAV
# ===============================

def convert_webm_to_wav(webm_path, wav_path):
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i", webm_path,
            "-ar", "16000",
            "-ac", "1",
            wav_path
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    return result.returncode == 0 and os.path.exists(wav_path)

# ===============================
# WebSocket: Audio streaming
# ===============================

@app.websocket("/ws/audio")
async def audio_stream(ws: WebSocket):
    global is_processing

    await ws.accept()
    print("🔌 Audio WebSocket connected")

    try:
        while True:
            audio_bytes = await ws.receive_bytes()

            if len(audio_bytes) < MIN_AUDIO_SIZE:
                continue

            # ⛔ Skip if still processing previous chunk
            if is_processing:
                continue

            is_processing = True

            ts = datetime.datetime.now().strftime("%H:%M:%S")
            size_kb = len(audio_bytes) / 1024
            print(f"[{ts}] Received audio chunk: {size_kb:.2f} KB")

            # Save WEBM
            with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
                tmp.write(audio_bytes)
                webm_path = tmp.name

            wav_path = webm_path.replace(".webm", ".wav")

            success = convert_webm_to_wav(webm_path, wav_path)

            if not success:
                os.remove(webm_path)
                is_processing = False
                continue

            # Transcription (BLOCKING)
            result = model.transcribe(wav_path, fp16=False)
            text = result["text"].strip()

            # Cleanup
            os.remove(webm_path)
            os.remove(wav_path)

            if not text:
                is_processing = False
                continue

            print("🗣️ Transcription:", text)

            # ===============================
            # Hate Speech Detection
            # ===============================

            hate_result = hate_classifier(text)
            predictions = hate_result[0]

            max_pred = max(predictions, key=lambda x: x["score"])
            label = max_pred["label"]
            score = max_pred["score"]

            if score >= HATE_THRESHOLD:
                print(f"🚨 Hate speech detected → {label} ({score:.2f})")
            else:
                print("✅ Clean speech (no hate detected)")

            # 🔓 Allow next chunk
            is_processing = False

    except Exception as e:
        print("❌ WebSocket closed:", e)
        is_processing = False


# ===============================
# TEST ENDPOINT
# ===============================

@app.get("/test-hate")
def test_hate():
    text = "I hate you and your stupid ideas"

    result = hate_classifier(text)
    predictions = result[0]
    max_pred = max(predictions, key=lambda x: x["score"])

    return {
        "input_text": text,
        "top_label": max_pred["label"],
        "confidence": round(max_pred["score"], 3),
        "all_scores": predictions
    }
