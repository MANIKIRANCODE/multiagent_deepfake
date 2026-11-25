import cv2

class VideoStream:
    def __init__(self, camera_index=0, width=640, height=480):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap = None

    def start(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam. Check camera permissions or camera index.")
        return self

    def read(self):
        if self.cap is None:
            return False, None
        return self.cap.read()

    def stop(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
