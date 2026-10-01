# from fastapi import FastAPI, WebSocket
# app = FastAPI() #Object
# """
# This is creating an object of the FastAPI class.
# Because later we use app to define things like endpoints and WebSocket routes.
# """


# @app.get("/health")
# def health_check():
#     return {"status": "Voice AI Bot backend is running"}


# @app.websocket("/ws/media-stream") #This tells FastAPI: Create a WebSocket endpoint at /ws/media-stream

# async def media_stream(websocket: WebSocket):
#     await websocket.accept()

#     print("WebSocket connected")

# # Conceptually, FastAPI does something like:

# # websocket_object = WebSocket(...)  object created
# # media_stream(websocket_object)  pass obj as parameter

# # So:
# # WebSocket object
# #        ↓
# # websocket parameter    

# #websocket.receive_bytes() 
# #In this object calling the function name receive_bytes from the class websocket

#     try:
#         while True:
#             # Receive one audio chunk
#             audio_chunk = await websocket.receive_bytes()

#             print(f"Received audio chunk: {len(audio_chunk)} bytes")

#     except Exception as e:
#         print("WebSocket disconnected:", e)

from fastapi import FastAPI, WebSocket
import numpy as np

from audio.vad import StreamingVAD
from stt.streaming_stt import StreamingSTT


app = FastAPI()


@app.get("/health")
def health_check():
    return {
        "status": "Voice AI Bot backend is running"
    }


@app.websocket("/ws/media-stream")
async def media_stream(websocket: WebSocket):

    await websocket.accept()

    print("WebSocket connected")

    # -----------------------------
    # VAD
    # -----------------------------

    vad = StreamingVAD(
        threshold=5.0,
        silence_duration=0.5,
        sample_rate=16000,
        chunk_duration=0.1,
        pre_roll_duration=0.2,
    )

    # -----------------------------
    # Streaming STT
    # -----------------------------

    stt = StreamingSTT(
        model_size="small",
        sample_rate=16000,
        language="en",
        window_seconds=2.0,
    )

    try:

        while True:

            # Receive raw PCM audio
            audio_bytes = await websocket.receive_bytes()

            # Convert bytes → int16 NumPy array
            audio = np.frombuffer(
                audio_bytes,
                dtype=np.int16
            )

            # -----------------------------
            # VAD
            # -----------------------------

            vad_result = vad.process(audio)

            # -----------------------------
            # Speech started
            # -----------------------------

            if vad_result["speech_started"]:

                print(
                    f"\n🎤 Speech STARTED "
                    f"(RMS: {vad_result['rms']:.2f})"
                )

            # -----------------------------
            # Speech is continuing
            # -----------------------------

            if vad_result["is_speaking"]:

                result = await stt.add_audio(audio)

                if result:

                    print(
                        f"📝 Partial: "
                        f"{result['text']}"
                    )

            # -----------------------------
            # Speech ended
            # -----------------------------

            if vad_result["speech_ended"]:

                print("🔴 Speech ENDED")

                final_result = await stt.finalize()

                if final_result:

                    print(
                        f"✅ FINAL: "
                        f"{final_result['text']}"
                    )

    except Exception as e:

        print(
            "WebSocket disconnected:",
            e
        )