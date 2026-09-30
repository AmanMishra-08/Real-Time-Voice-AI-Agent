import asyncio
import numpy as np
import sounddevice as sd
import websockets


SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_DURATION = 0.1

CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)


async def microphone_stream():

    uri = "ws://127.0.0.1:8000/ws/media-stream"

    async with websockets.connect(uri) as websocket:

        print("WebSocket connected")
        print("Microphone streaming started...")
        print("Speak for 5 seconds.")

        for _ in range(50):

            audio = sd.rec(
                CHUNK_SIZE,
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="int16"
            )

            sd.wait()

            # Calculate audio energy
            rms = np.sqrt(np.mean(audio.astype(np.float32) ** 2))

            print(f"Audio level: {rms:.2f}")

            # Convert audio to bytes
            audio_bytes = audio.tobytes()

            # Send audio through WebSocket
            await websocket.send(audio_bytes)

        print("Microphone streaming finished.")


asyncio.run(microphone_stream())
