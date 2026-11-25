
import cv2
import time

from agents.face_detection_agent import FaceDetectionAgent

cap = cv2.VideoCapture(0)
detector = FaceDetectionAgent()

time.sleep(2)

for i in range(15):
    ok, frame = cap.read()

    if not ok:
        print("Camera frame unavailable")
        continue

    result = detector.analyze(frame)

    print("Attempt:", i + 1)
    print("Faces detected:", result["face_count"])

    if result["face_count"] > 0:
        print("Face coordinates:", result["faces"][0])
        break

cap.release()
print("Test completed.")
