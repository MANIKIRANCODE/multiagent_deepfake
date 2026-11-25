
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .base_agent import BaseAgent


class AntiSpoofAgent(BaseAgent):
    name = "anti_spoofing"

    def __init__(self, model_path=None):
        self.ready = False
        self.models = []
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

        # Locate the downloaded Silent-Face repository.
        project_root = Path(__file__).resolve().parents[1]
        self.repo_root = project_root / "Silent-Face-Anti-Spoofing"

        if not self.repo_root.exists():
            self.error = f"Repository not found: {self.repo_root}"
            return

        if str(self.repo_root) not in sys.path:
            sys.path.insert(0, str(self.repo_root))

        try:
            from src.model_lib.MiniFASNet import (
                MiniFASNetV1,
                MiniFASNetV2,
                MiniFASNetV1SE,
                MiniFASNetV2SE,
            )
            from src.utility import parse_model_name, get_kernel
            from src.generate_patches import CropImage
            from src.data_io.transform import ToTensor

            self.parse_model_name = parse_model_name
            self.get_kernel = get_kernel
            self.CropImage = CropImage()
            self.to_tensor = ToTensor()

            model_mapping = {
                "MiniFASNetV1": MiniFASNetV1,
                "MiniFASNetV2": MiniFASNetV2,
                "MiniFASNetV1SE": MiniFASNetV1SE,
                "MiniFASNetV2SE": MiniFASNetV2SE,
            }

            self.model_dir = (
                Path(model_path)
                if model_path
                else self.repo_root / "resources" / "anti_spoof_models"
            )

            model_files = sorted(self.model_dir.glob("*.pth"))

            if not model_files:
                raise FileNotFoundError(
                    f"No .pth model files found in {self.model_dir}"
                )

            for model_file in model_files:
                h, w, model_type, scale = parse_model_name(
                    model_file.name
                )

                model_class = model_mapping[model_type]

                model = model_class(
                    conv6_kernel=get_kernel(h, w)
                ).to(self.device)

                state_dict = torch.load(
                    model_file,
                    map_location=self.device,
                    weights_only=True,
                )

                # Handle weights saved using DataParallel.
                if any(key.startswith("module.") for key in state_dict):
                    state_dict = {
                        key.replace("module.", "", 1): value
                        for key, value in state_dict.items()
                    }

                model.load_state_dict(state_dict)
                model.eval()

                self.models.append({
                    "model": model,
                    "height": h,
                    "width": w,
                    "scale": scale,
                    "name": model_file.name,
                })

            self.ready = True
            self.error = None

            print(
                f"[AntiSpoofAgent] Loaded {len(self.models)} model(s) "
                f"on {self.device}"
            )

        except Exception as exc:
            self.ready = False
            self.error = str(exc)
            print(f"[AntiSpoofAgent] Initialization failed: {exc}")

    def analyze(self, frame=None, face=None):

        if not self.ready:
            return self.result(
                status="NOT_CONFIGURED",
                score=None,
                details=f"MiniFASNet unavailable: {self.error}",
            )

        if frame is None or face is None:
            return self.result(
                status="WARNING",
                score=None,
                details="No face available for anti-spoofing analysis.",
                prediction="NO_FACE",
            )

        try:
            x, y, w, h = map(int, face)

            if w <= 0 or h <= 0:
                raise ValueError("Invalid face bounding box.")

            image_h, image_w = frame.shape[:2]

            if x < 0 or y < 0 or x + w > image_w or y + h > image_h:
                raise ValueError("Face bounding box is outside the frame.")

            prediction = np.zeros((1, 3), dtype=np.float64)

            # Run every pretrained model on the same detected face.
            for item in self.models:
                scale = item["scale"]

                patch = self.CropImage.crop(
                    org_img=frame,
                    bbox=(x, y, w, h),
                    scale=scale,
                    out_w=item["width"],
                    out_h=item["height"],
                    crop=(scale is not None),
                )

                tensor = self.to_tensor(patch)
                tensor = tensor.unsqueeze(0).to(self.device)

                with torch.no_grad():
                    output = item["model"](tensor)
                    probabilities = F.softmax(output, dim=1)

                prediction += probabilities.cpu().numpy()

            # Average predictions from all loaded models.
            prediction /= len(self.models)

            predicted_class = int(np.argmax(prediction))
            confidence = float(prediction[0][predicted_class])

            # Official repository: class 1 means real face.
            real_score = float(prediction[0][1])
            fake_score = float(
                prediction[0][0] + prediction[0][2]
            )

            verdict = (
                "REAL_FACE"
                if predicted_class == 1
                else "SPOOF_SUSPECTED"
            )

            return self.result(
                status="OK",
                score=real_score,
                details=(
                    f"MiniFASNet prediction: {verdict}. "
                    "This is a model estimate, not proof of authenticity."
                ),
                prediction=verdict,
                real_score=real_score,
                fake_score=fake_score,
                confidence=confidence,
                class_probabilities=prediction[0].tolist(),
                model_count=len(self.models),
            )

        except Exception as exc:
            return self.result(
                status="ERROR",
                score=None,
                details=f"Anti-spoofing inference failed: {exc}",
            )
