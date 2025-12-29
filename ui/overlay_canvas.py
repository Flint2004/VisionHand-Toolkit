import cv2
import numpy as np
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QPainter, QImage, QPixmap
from utils.theme import SCREEN_SIZE

class OverlayCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(SCREEN_SIZE[0], SCREEN_SIZE[1])
        
        # Multiple Layers: One for each tool that needs persistence
        self.layers = {
            "PAINTER": np.zeros((SCREEN_SIZE[1], SCREEN_SIZE[0], 4), dtype=np.uint8),
            "PAINTER_ALT": np.zeros((SCREEN_SIZE[1], SCREEN_SIZE[0], 4), dtype=np.uint8)
        }
        
        self.xp, self.yp = 0, 0
        self.thickness = 10

    def draw_line(self, x, y, is_drawing, tool_name="PAINTER", color=(254, 242, 0, 255), thickness=None):
        """
        Draws onto the specified tool's layer.
        """
        if is_drawing:
            if self.xp == 0 and self.yp == 0:
                self.xp, self.yp = x, y
            
            target_layer = self.layers.get(tool_name, self.layers["PAINTER"])
            draw_thickness = thickness if thickness is not None else self.thickness
            cv2.line(target_layer, (self.xp, self.yp), (x, y), color, draw_thickness)
            
            self.xp, self.yp = x, y
            self.update()
        else:
            self.xp, self.yp = 0, 0

    def clear_layer(self, tool_name):
        if tool_name in self.layers:
            self.layers[tool_name].fill(0)
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        h, w = SCREEN_SIZE[1], SCREEN_SIZE[0]
        
        # Merge and draw layers
        for layer_name in ["PAINTER", "PAINTER_ALT"]:
            layer = self.layers[layer_name]
            qi = QImage(layer.data, w, h, w * 4, QImage.Format_RGBA8888)
            painter.drawImage(0, 0, qi)
