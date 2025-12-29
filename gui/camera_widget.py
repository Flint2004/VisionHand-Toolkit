import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QImage
from core.detector import HandTracker
from utils.config import WIDTH, HEIGHT

class CameraThread(QThread):
    change_pixmap_signal = Signal(np.ndarray, list)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.tracker = HandTracker()

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

        while self._run_flag:
            success, img = cap.read()
            if success:
                img = cv2.flip(img, 1)
                hands, img = self.tracker.find_hands(img)
                self.change_pixmap_signal.emit(img, hands)
            else:
                break
        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

def cv_to_qimage(cv_img):
    """Convert from an opencv image to QImage"""
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return convert_to_Qt_format
