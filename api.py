from fastapi import FastAPI, WebSocket
import whisper
import tempfile
import os
import datetime

app = FastAPI()

# Load Whisper model ONCE at startup
model = whisper.load_model("base")
print("✅ Whisper model loaded")

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

    except Exception as e:
        print("❌ WebSocket closed:", e)
