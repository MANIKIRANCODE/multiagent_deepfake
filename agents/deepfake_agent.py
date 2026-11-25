
from pathlib import Path

import cv2
import numpy as np
import torch
import timm

from huggingface_hub import hf_hub_download
from .base_agent import BaseAgent


class DeepfakeAgent(BaseAgent):
    name = "deepfake_detection"

    def __init__(self, model_path=None):
        self.ready = False
        self.error = None
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

        try:
            # Locate the trained checkpoint.
            if model_path is None:
                model_path = hf_hub_download(
                    repo_id="mahedi420/deepfake-vit-detector",
                    filename="custom/xception_model_custom.pt"
                )

            checkpoint = torch.load(
                model_path,
                map_location=self.device,
                weights_only=True
            )

            # Construct the exact Xception architecture.
            self.model = timm.create_model(
                "legacy_xception",
                pretrained=False,
                num_classes=2
            ).to(self.device)

            self.model.load_state_dict(
                checkpoint["model_state"],
                strict=True
            )

            self.model.eval()

            # ImageNet normalization.
            self.mean = torch.tensor(
                [0.485, 0.456, 0.406],
                dtype=torch.float32
            ).view(1, 3, 1, 1).to(self.device)

            self.std = torch.tensor(
                [0.229, 0.224, 0.225],
                dtype=torch.float32
            ).view(1, 3, 1, 1).to(self.device)

            self.ready = True
            print(
                f"[DeepfakeAgent] Xception loaded on {self.device}"
            )

        except Exception as exc:
            self.error = str(exc)
            print(f"[DeepfakeAgent] Initialization failed: {exc}")

    def analyze(self, frame=None, face=None):

        if not self.ready:
            return self.result(
                status="NOT_CONFIGURED",
                score=None,
                details=f"Xception model unavailable: {self.error}"
            )

        if frame is None or face is None:
            return self.result(
                status="WARNING",
                score=None,
                details="No face available for deepfake analysis."
            )

        try:
            x, y, w, h = map(int, face)

            if w < 40 or h < 40:
                return self.result(
                    status="WARNING",
                    score=None,
                    details="Face crop is too small for reliable analysis."
                )

            image_h, image_w = frame.shape[:2]

            # Add padding around the detected face.
            pad_x = int(w * 0.15)
            pad_y = int(h * 0.15)

            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(image_w, x + w + pad_x)
            y2 = min(image_h, y + h + pad_y)

            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size == 0:
                raise ValueError("Empty face crop.")

            # OpenCV BGR -> RGB.
            rgb = cv2.cvtColor(
                face_crop,
                cv2.COLOR_BGR2RGB
            )

            # Model input: 224 x 224.
            resized = cv2.resize(
                rgb,
                (224, 224),
                interpolation=cv2.INTER_AREA
            )

            # Convert image to PyTorch tensor.
            tensor = torch.from_numpy(
                np.ascontiguousarray(resized)
            )

            tensor = tensor.permute(2, 0, 1).float() / 255.0
            tensor = tensor.unsqueeze(0).to(self.device)

            # Normalize using ImageNet statistics.
            tensor = (tensor - self.mean) / self.std

            # Inference.
            with torch.inference_mode():
                logits = self.model(tensor)
                probabilities = torch.softmax(logits, dim=1)

            probabilities = probabilities[0].cpu().numpy()

            real_score = float(probabilities[0])
            fake_score = float(probabilities[1])

            prediction = (
                "FAKE_SUSPECTED"
                if fake_score >= 0.5
                else "REAL_PREDICTED"
            )

            return self.result(
                status="OK",
                score=fake_score,
                details=(
                    "Xception deepfake classification estimate. "
                    "Not forensic proof of manipulation. "
                    "Face crop uses the existing detector rather "
                    "than the model's original MTCNN pipeline."
                ),
                prediction=prediction,
                real_score=real_score,
                fake_score=fake_score,
                model_name="Xception",
                class_mapping={
                    "0": "REAL",
                    "1": "FAKE"
                }
            )

        except Exception as exc:
            return self.result(
                status="ERROR",
                score=None,
                details=f"Deepfake inference failed: {exc}"
            )
