import asyncio
import websockets

async def send_audio():
    uri = "ws://127.0.0.1:8000/ws/audio"

    async with websockets.connect(uri) as ws:
        print("Connected to server")

        with open("test.wav", "rb") as f:
            audio = f.read()

        await ws.send(audio)
        print("Sent full audio file")

        await asyncio.sleep(1)

asyncio.run(send_audio())
