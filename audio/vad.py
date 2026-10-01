import numpy as np


class StreamingVAD:
    """
    Stateful Voice Activity Detection engine.

    Responsibilities:
    - Analyze incoming audio chunks
    - Detect speech start
    - Keep track of active speech
    - Handle short silence periods
    - Detect speech end
    - Maintain a small pre-roll buffer
    - Return completed speech audio
    """

    def __init__(
        self,
        threshold=5.0,
        silence_duration=0.5,
        sample_rate=16000,
        chunk_duration=0.1,
        pre_roll_duration=0.2,
    ):
        self.threshold = threshold

        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration

        # Number of consecutive silent chunks required
        # before declaring that speech has ended.
        self.silence_chunks_required = int(
            silence_duration / chunk_duration
        )

        # Number of chunks kept before speech detection.
        # This protects the beginning of an utterance.
        self.pre_roll_chunks = int(
            pre_roll_duration / chunk_duration
        )

        # Current VAD state
        self.is_speaking = False

        # Number of consecutive silent chunks
        self.silence_chunks = 0

        # Audio belonging to the current utterance
        self.audio_buffer = []

        # Small rolling buffer before speech starts
        self.pre_roll_buffer = []

    def calculate_rms(self, audio):
        """
        Calculate RMS energy of an audio chunk.
        """

        audio_float = audio.astype(np.float32)

        rms = np.sqrt(np.mean(audio_float ** 2))

        return rms

    def process(self, audio):
        """
        Process one incoming audio chunk.

        Returns a dictionary describing the current VAD event.
        """

        rms = self.calculate_rms(audio)

        speech_detected = rms > self.threshold

        speech_started = False
        speech_ended = False
        completed_audio = None

        # --------------------------------------------------
        # Maintain pre-roll buffer
        # --------------------------------------------------

        self.pre_roll_buffer.append(audio.copy())

        if len(self.pre_roll_buffer) > self.pre_roll_chunks:
            self.pre_roll_buffer.pop(0)

        # --------------------------------------------------
        # SPEECH DETECTED
        # --------------------------------------------------

        if speech_detected:

            # ----------------------------------------------
            # Speech has just started
            # ----------------------------------------------

            if not self.is_speaking:

                self.is_speaking = True
                speech_started = True

                # Add pre-roll audio so we don't lose
                # the beginning of the utterance.
                self.audio_buffer.extend(
                    self.pre_roll_buffer
                )

                # Reset silence counter.
                self.silence_chunks = 0

            # ----------------------------------------------
            # Speech is continuing
            # ----------------------------------------------

            else:

                self.silence_chunks = 0
                self.audio_buffer.append(audio.copy())

        # --------------------------------------------------
        # SILENCE DETECTED
        # --------------------------------------------------

        else:

            # If we are currently speaking,
            # this silence may be temporary.
            if self.is_speaking:

                self.silence_chunks += 1

                # Keep the silent chunk temporarily.
                self.audio_buffer.append(audio.copy())

                # ------------------------------------------
                # Speech has ended
                # ------------------------------------------

                if (
                    self.silence_chunks
                    >= self.silence_chunks_required
                ):

                    self.is_speaking = False
                    speech_ended = True

                    completed_audio = np.concatenate(
                        self.audio_buffer,
                        axis=0
                    )

                    # Reset buffers for next utterance.
                    self.audio_buffer = []
                    self.silence_chunks = 0

        return {
            "rms": rms,
            "speech_detected": speech_detected,
            "speech_started": speech_started,
            "speech_ended": speech_ended,
            "is_speaking": self.is_speaking,
            "audio": completed_audio,
        }