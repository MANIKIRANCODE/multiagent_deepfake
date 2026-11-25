import cv2
from .base_agent import BaseAgent

class FaceDetectionAgent(BaseAgent):
    name = "face_detection"

    def __init__(self, scale_factor=1.1, min_neighbors=5):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade_path)
        if self.detector.empty():
            raise RuntimeError("OpenCV Haar face detector could not be loaded.")
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors

    def analyze(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray, scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors, minSize=(60, 60)
        )
        boxes = [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]
        return self.result(
            status="OK" if len(boxes) else "WARNING",
            score=None,
            details=f"Detected {len(boxes)} face(s). Detection is not identity verification.",
            faces=boxes,
            face_count=len(boxes),
        )
