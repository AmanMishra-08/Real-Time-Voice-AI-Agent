from fastapi import FastAPI, WebSocket
app = FastAPI() #Object
"""
This is creating an object of the FastAPI class.
Because later we use app to define things like endpoints and WebSocket routes.
"""


@app.get("/health")
def health_check():
    return {"status": "Voice AI Bot backend is running"}


@app.websocket("/ws/media-stream") #This tells FastAPI: Create a WebSocket endpoint at /ws/media-stream

async def media_stream(websocket: WebSocket):
    await websocket.accept()

    print("WebSocket connected")

# Conceptually, FastAPI does something like:

# websocket_object = WebSocket(...)  object created
# media_stream(websocket_object)  pass obj as parameter

# So:
# WebSocket object
#        ↓
# websocket parameter    

#websocket.receive_bytes() 
#In this object calling the function name receive_bytes from the class websocket

    try:
        while True:
            # Receive one audio chunk
            audio_chunk = await websocket.receive_bytes()

            print(f"Received audio chunk: {len(audio_chunk)} bytes")

    except Exception as e:
        print("WebSocket disconnected:", e)