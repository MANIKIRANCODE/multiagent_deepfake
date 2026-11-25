
import time
import numpy as np

from .base_agent import BaseAgent


class LipSyncAgent(BaseAgent):
    name = "lip_sync"

    UPPER_LIP = 13
    LOWER_LIP = 14
    LEFT_MOUTH = 61
    RIGHT_MOUTH = 291

    def __init__(
        self,
        history_seconds=8,
        max_audio_delay=0.4,
        sample_interval=0.06
    ):
        self.history_seconds = history_seconds
        self.max_audio_delay = max_audio_delay
        self.sample_interval = sample_interval

        self.samples = []
        self.mesh = None

        try:
            import mediapipe as mp

            self.mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

        except Exception as exc:
            print(f"[LipSyncAgent] MediaPipe error: {exc}")

    def analyze(
        self,
        frame,
        audio_level=0.0,
        audio_samples=None,
        timestamp=None
    ):

        timestamp = (
            time.monotonic()
            if timestamp is None
            else timestamp
        )

        if self.mesh is None:
            return self.result(
                status="NOT_CONFIGURED",
                score=None,
                details="MediaPipe Face Mesh is unavailable."
            )

        rgb = frame[:, :, ::-1].copy()
        result = self.mesh.process(rgb)

        if not result.multi_face_landmarks:
            self.samples.append(
                (timestamp, 0.0, False)
            )

            self._trim(timestamp)

            return self.result(
                status="WARNING",
                score=None,
                details="No face landmarks available."
            )

        landmarks = result.multi_face_landmarks[0].landmark

        upper = landmarks[self.UPPER_LIP]
        lower = landmarks[self.LOWER_LIP]

        left = landmarks[self.LEFT_MOUTH]
        right = landmarks[self.RIGHT_MOUTH]

        mouth_open = float(
            np.hypot(
                upper.x - lower.x,
                upper.y - lower.y
            )
        )

        mouth_width = float(
            np.hypot(
                left.x - right.x,
                left.y - right.y
            )
        )

        normalized_open = mouth_open / max(
            mouth_width,
            1e-6
        )

        self.samples.append(
            (timestamp, normalized_open, True)
        )

        self._trim(timestamp)

        if audio_samples is None:
            return self.result(
                status="WARNING",
                score=None,
                details="Timestamped audio history unavailable."
            )

        score, best_delay, sample_count = (
            self._calculate_sync(
                audio_samples,
                timestamp
            )
        )

        if score is None:
            return self.result(
                status="WARNING",
                score=None,
                details=(
                    "Collecting synchronized audio and mouth "
                    "movement samples."
                ),
                mouth_open_ratio=normalized_open,
                sample_count=sample_count
            )

        return self.result(
            status="OK",
            score=score,
            details=(
                "Exploratory audio/mouth-motion correlation. "
                "Not a validated lip-sync or spoof detector."
            ),
            sync_score=score,
            estimated_audio_delay=best_delay,
            mouth_open_ratio=normalized_open,
            sample_count=sample_count
        )

    def _trim(self, now):
        self.samples = [
            sample
            for sample in self.samples
            if now - sample[0] <= self.history_seconds
        ]

    def _calculate_sync(
        self,
        audio_samples,
        now
    ):

        valid = [
            sample
            for sample in self.samples
            if sample[2]
            and now - sample[0] <= self.history_seconds
        ]

        if len(valid) < 30 or len(audio_samples) < 10:
            return None, None, len(valid)

        mouth_times = np.array(
            [sample[0] for sample in valid],
            dtype=float
        )

        mouth_values = np.array(
            [sample[1] for sample in valid],
            dtype=float
        )

        audio_times = np.array(
            [sample[0] for sample in audio_samples],
            dtype=float
        )

        audio_values = np.array(
            [sample[1] for sample in audio_samples],
            dtype=float
        )

        # Restrict comparison to the recent common time interval.
        start = max(
            mouth_times[0],
            audio_times[0]
        )

        end = min(
            mouth_times[-1],
            audio_times[-1]
        )

        if end - start < 2.0:
            return None, None, len(valid)

        grid = np.arange(
            start,
            end,
            self.sample_interval
        )

        if len(grid) < 30:
            return None, None, len(valid)

        mouth = np.interp(
            grid,
            mouth_times,
            mouth_values
        )

        best_score = None
        best_delay = None

        delays = np.arange(
            -self.max_audio_delay,
            self.max_audio_delay + 0.001,
            self.sample_interval
        )

        for delay in delays:

            # Positive delay means compare mouth activity
            # with audio activity slightly later.
            shifted_audio = np.interp(
                grid + delay,
                audio_times,
                audio_values,
                left=np.nan,
                right=np.nan
            )

            valid_mask = np.isfinite(shifted_audio)

            if np.count_nonzero(valid_mask) < 30:
                continue

            m = mouth[valid_mask]
            a = shifted_audio[valid_mask]

            if np.std(m) < 1e-5 or np.std(a) < 1e-5:
                continue

            correlation = float(
                np.corrcoef(m, a)[0, 1]
            )

            if not np.isfinite(correlation):
                continue

            if (
                best_score is None
                or correlation > best_score
            ):
                best_score = correlation
                best_delay = float(delay)

        if best_score is None:
            return None, None, len(valid)

        # Convert correlation to a descriptive 0-100 score.
        descriptive_score = round(
            (best_score + 1.0) * 50.0,
            1
        )

        return (
            descriptive_score,
            round(best_delay, 3),
            len(valid)
        )
