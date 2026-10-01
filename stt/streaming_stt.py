import asyncio
import numpy as np
from faster_whisper import WhisperModel


class StreamingSTT:
    """
    Incremental Speech-to-Text engine.

    Audio arrives continuously in small chunks.
    Instead of waiting for the complete call, we periodically
    transcribe a rolling window of recent audio.

    Architecture:

        Audio chunks
              ↓
        Audio buffer
              ↓
        Rolling window
              ↓
        Whisper model
              ↓
        Partial transcript
              ↓
        Final transcript
    """

    def __init__(
        self,
        model_size="small",
        sample_rate=16000,
        language="en",
        window_seconds=2.0,
        update_interval=0.5,
    ):
        self.sample_rate = sample_rate
        self.language = language

        self.window_samples = int(
            sample_rate * window_seconds
        )

        self.update_interval = update_interval

        print("Loading Whisper model...")

        self.model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
        )

        print("Whisper model loaded.")

        # All audio received so far for the current utterance
        self.audio_buffer = np.array([], dtype=np.float32)

        self.last_transcript = ""

        self.last_processed_samples = 0

    async def add_audio(self, audio: np.ndarray):
        """
        Add a new audio chunk.

        audio:
            int16 or float32 mono PCM audio
        """

        # Convert int16 PCM → float32
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0

        audio = audio.flatten()

        self.audio_buffer = np.concatenate(
            [self.audio_buffer, audio]
        )

        # Don't run Whisper until enough audio exists.
        if len(self.audio_buffer) < self.window_samples:
            return None

        return await self._transcribe_window()

    async def _transcribe_window(self):
        """
        Transcribe the most recent rolling audio window.

        Whisper inference is CPU-heavy, so it runs in a
        background thread instead of blocking FastAPI's
        async event loop.
        """

        audio = self.audio_buffer[-self.window_samples:].copy()

        segments, info = await asyncio.to_thread(
            self.model.transcribe,
            audio,
            language=self.language,
            beam_size=5,
            vad_filter=True,
        )

        text = " ".join(
            segment.text.strip()
            for segment in segments
        ).strip()

        if not text:
            return None

        # Avoid repeatedly returning exactly the same partial.
        if text == self.last_transcript:
            return None

        self.last_transcript = text

        return {
            "type": "partial",
            "text": text,
            "language": info.language,
            "probability": info.language_probability,
        }

    async def finalize(self):
        """
        Called when VAD detects speech end.

        The complete utterance is transcribed one final time.
        """

        if len(self.audio_buffer) == 0:
            return None

        audio = self.audio_buffer.copy()

        segments, info = await asyncio.to_thread(
            self.model.transcribe,
            audio,
            language=self.language,
            beam_size=5,
            vad_filter=True,
        )

        text = " ".join(
            segment.text.strip()
            for segment in segments
        ).strip()

        result = {
            "type": "final",
            "text": text,
            "language": info.language,
            "probability": info.language_probability,
        }

        self.reset()

        return result

    def reset(self):
        """
        Reset the current utterance.
        """

        self.audio_buffer = np.array([], dtype=np.float32)

        self.last_transcript = ""

        self.last_processed_samples = 0