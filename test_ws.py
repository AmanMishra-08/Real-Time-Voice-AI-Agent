import asyncio #Python's library for asynchronous programming.
import websockets


async def test_websocket():

    uri = "ws://127.0.0.1:8000/ws/media-stream"

    async with websockets.connect(uri) as websocket:

        # Simulated audio chunks
        audio_chunks = [
            b"audio_chunk_1",
            b"audio_chunk_2",
            b"audio_chunk_3",
            b"audio_chunk_4",
            b"audio_chunk_5"
        ]

        for chunk in audio_chunks:

            await websocket.send(chunk)

            print(f"Sent chunk: {len(chunk)} bytes")

            await asyncio.sleep(0.5)


asyncio.run(test_websocket())