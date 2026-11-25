
import threading
import time
from collections import deque

import numpy as np


class AudioStream:
    """Captures microphone blocks and stores timestamped RMS energy."""

    def __init__(self, samplerate=16000, blocksize=1024):
        self.samplerate = samplerate
        self.blocksize = blocksize

        self.level = None
        self.available = False
        self.error = None
        self.stream = None

        # Keep approximately 16 seconds of audio measurements.
        self.samples = deque(maxlen=256)
        self.lock = threading.Lock()

    def start(self):
        try:
            import sounddevice as sd

            def callback(indata, frames, time_info, status):
                if status:
                    self.error = str(status)

                level = float(
                    np.sqrt(
                        np.mean(
                            np.square(indata.astype(np.float32))
                        )
                    )
                )

                timestamp = time.monotonic()

                self.level = level

                with self.lock:
                    self.samples.append((timestamp, level))

            self.stream = sd.InputStream(
                samplerate=self.samplerate,
                channels=1,
                blocksize=self.blocksize,
                callback=callback
            )

            self.stream.start()
            self.available = True

        except Exception as exc:
            self.available = False
            self.error = str(exc)

    def get_samples(self):
        """Return a safe snapshot of recent timestamped audio samples."""
        with self.lock:
            return list(self.samples)

    def stop(self):
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass

        self.available = False
