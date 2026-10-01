import asyncio

import numpy as np
import sounddevice as sd
import websockets

from audio.vad import StreamingVAD


SAMPLE_RATE = 16000
CHANNELS = 1

CHUNK_DURATION = 0.1
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)


async def microphone_stream():

    uri = "ws://127.0.0.1:8000/ws/media-stream"

    # Create VAD engine
    vad = StreamingVAD(
        threshold=5.0,
        silence_duration=0.5,
        sample_rate=SAMPLE_RATE,
        chunk_duration=CHUNK_DURATION,
        pre_roll_duration=0.2,
    )

    async with websockets.connect(uri) as websocket:

        print("WebSocket connected")
        print("Microphone streaming started...")
        print("Speak for 5 seconds.")

        for _ in range(50):

            # -----------------------------------------
            # Capture microphone audio
            # -----------------------------------------

            audio = sd.rec(
                CHUNK_SIZE,
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="int16",
            )

            sd.wait()

            # -----------------------------------------
            # Send raw audio through WebSocket
            # -----------------------------------------

            audio_bytes = audio.tobytes()

            await websocket.send(audio_bytes)

            # -----------------------------------------
            # Process audio through VAD
            # -----------------------------------------

            result = vad.process(audio)

            rms = result["rms"]

            # -----------------------------------------
            # Speech started
            # -----------------------------------------

            if result["speech_started"]:

                print(
                    f"Speech STARTED | RMS: {rms:.2f}"
                )

            # -----------------------------------------
            # Speech continuing
            # -----------------------------------------

            elif result["is_speaking"]:

                print(
                    f"Speaking | RMS: {rms:.2f}"
                )

            # -----------------------------------------
            # Speech ended
            # -----------------------------------------

            if result["speech_ended"]:

                completed_audio = result["audio"]

                print(
                    f"Speech ENDED | "
                    f"Audio samples: {len(completed_audio)}"
                )

        print("Microphone streaming finished.")


asyncio.run(microphone_stream())