
import cv2
import time

from agents.face_detection_agent import FaceDetectionAgent
from agents.deepfake_agent import DeepfakeAgent

cap = cv2.VideoCapture(0)

detector = FaceDetectionAgent()
deepfake = DeepfakeAgent()

time.sleep(2)

for i in range(15):
    ok, frame = cap.read()

    if not ok:
        continue

    result = detector.analyze(frame)

    if result["face_count"] > 0:
        face = result["faces"][0]

        print("Face detected:", face)

        prediction = deepfake.analyze(
            frame=frame,
            face=face
        )

        print("\nDeepfake analysis:")
        print("Status:", prediction["status"])
        print("Prediction:", prediction.get("prediction"))
        print("Real score:", prediction.get("real_score"))
        print("Fake score:", prediction.get("fake_score"))
        print("Details:", prediction["details"])

        break

else:
    print("No face detected after 15 attempts.")

cap.release()

