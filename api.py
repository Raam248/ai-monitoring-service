from fastapi import FastAPI, WebSocket
import datetime

app = FastAPI()

@app.websocket("/ws/audio")
async def audio_stream(ws: WebSocket):
    await ws.accept()
    print("🔌 Audio WebSocket connected")

    try:
        while True:
            data = await ws.receive_bytes()
            size_kb = len(data) / 1024
            ts = datetime.datetime.now().strftime("%H:%M:%S")

            print(f"[{ts}] Received audio chunk: {size_kb:.2f} KB")
    except Exception as e:
        print("❌ WebSocket closed:", e)
