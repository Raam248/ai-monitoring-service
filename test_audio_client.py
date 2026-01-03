import asyncio
import websockets

async def send_audio():
    uri = "ws://127.0.0.1:8000/ws/audio"

    async with websockets.connect(uri) as ws:
        print("Connected to server")

        with open("test.wav", "rb") as f:
            audio = f.read()

        chunk_size = 16000 * 4  # simulate ~4 sec chunks

        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i+chunk_size]
            await ws.send(chunk)
            print("Sent chunk")
            await asyncio.sleep(4)

asyncio.run(send_audio())
