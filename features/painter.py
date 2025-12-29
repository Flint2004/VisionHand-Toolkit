import cv2
import numpy as np
from utils.config import WIDTH, HEIGHT, COLORS, BRUSH_THICKNESS, ERASER_THICKNESS

class PaintService:
    def __init__(self):
        self.canvas = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
        self.brush_color = COLORS["blue"]
        self.thickness = BRUSH_THICKNESS
        self.xp, self.yp = 0, 0

    def draw(self, img, x, y, is_drawing, is_erasing=False):
        """
        Draws on the canvas based on coordinates.
        Does NOT merge with original image yet.
        """
        color = COLORS["black"] if is_erasing else self.brush_color
        thickness = ERASER_THICKNESS if is_erasing else self.thickness

        if is_drawing or is_erasing:
            if self.xp == 0 and self.yp == 0:
                self.xp, self.yp = x, y

            cv2.line(self.canvas, (self.xp, self.yp), (x, y), color, thickness)
            self.xp, self.yp = x, y
        else:
            self.xp, self.yp = 0, 0
        
        return self.canvas

    def merge_canvas(self, img):
        """
        Merges the persistent canvas with the input image.
        """
        img_gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, img_inv = cv2.threshold(img_gray, 5, 255, cv2.THRESH_BINARY_INV)
        img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, img_inv)
        img = cv2.bitwise_or(img, self.canvas)
        return img

    def set_color(self, color_name):
        if color_name in COLORS:
            self.brush_color = COLORS[color_name]

    def clear_canvas(self):
        self.canvas = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
