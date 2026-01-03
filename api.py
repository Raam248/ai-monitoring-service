from fastapi import FastAPI, WebSocket
import whisper
import tempfile
import os
import datetime
from transformers import pipeline

app = FastAPI()

# ===============================
# Load models ONCE at startup
# ===============================

# Whisper model
model = whisper.load_model("base")
print("✅ Whisper model loaded")

# Hate speech classifier
hate_classifier = pipeline(
    "text-classification",
    model="unitary/toxic-bert",
    top_k=None
)
print("✅ Hate speech model loaded")


# ===============================
# WebSocket: Audio streaming
# ===============================

@app.websocket("/ws/audio")
async def audio_stream(ws: WebSocket):
    await ws.accept()
    print("🔌 Audio WebSocket connected")

    try:
        while True:
            # receive raw audio bytes
            audio_bytes = await ws.receive_bytes()

            size_kb = len(audio_bytes) / 1024
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            print(f"[{ts}] Received audio chunk: {size_kb:.2f} KB")

            # save chunk to temporary wav file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            # transcribe using Whisper
            result = model.transcribe(tmp_path, fp16=False)
            text = result["text"].strip()

            # cleanup temp file
            os.remove(tmp_path)

            if text:
                print("🗣️ Transcription:", text)

                # hate speech detection
                hate_result = hate_classifier(text)

                print("🚨 Hate detection:", hate_result)


    except Exception as e:
        print("❌ WebSocket closed:", e)


# ===============================
# TEST ENDPOINT: Hate detection
# ===============================

@app.get("/test-hate")
def test_hate():
    text = "I hate you and your stupid ideas"

    result = hate_classifier(text)

    return {
        "input_text": text,
        "model_output": result
    }
