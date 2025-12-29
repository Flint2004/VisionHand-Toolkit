import cv2
import numpy as np
import time

class VirtualKeyboard:
    def __init__(self):
        self.keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
                     ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
                     ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "CLR"]]
        self.key_width = 80
        self.key_height = 80
        self.padding = 15
        self.start_x = 150
        self.start_y = 250
        self.text = ""
        self.last_press_time = 0
        self.cooldown = 0.5 # Seconds between typing

    def draw(self, img, hands=None):
        """
        Draws the keyboard and handles typing logic.
        """
        for i, row in enumerate(self.keys):
            for j, key in enumerate(row):
                x = self.start_x + j * (self.key_width + self.padding)
                y = self.start_y + i * (self.key_height + self.padding)
                
                is_hovered = False
                is_pressed = False
                
                if hands:
                    hx, hy = hands[0]['landmarks'][8][:2] # Index tip
                    if x < hx < x + self.key_width and y < hy < y + self.key_height:
                        is_hovered = True
                        # Click logic: Pinch (Thumb + Index distance)
                        dist = self._get_dist(hands[0]['landmarks'][4], hands[0]['landmarks'][8])
                        scale = hands[0].get('scale', 100)
                        
                        # Normalized pinch threshold (relative to hand size)
                        if dist < (0.35 * scale): 
                            is_pressed = True
                            self._on_key_press(key)

                # Visual Feedback
                color = (0, 242, 254) if is_pressed else ((0, 255, 0) if is_hovered else (255, 255, 255))
                alpha = 220 if is_pressed else (150 if is_hovered else 60)
                
                overlay = img.copy()
                cv2.rectangle(overlay, (x, y), (x + self.key_width, y + self.key_height), color, cv2.FILLED)
                cv2.rectangle(overlay, (x, y), (x + self.key_width, y + self.key_height), (255, 255, 255), 2)
                cv2.addWeighted(overlay, alpha/255, img, 1 - alpha/255, 0, img)
                
                font_scale = 0.8 if len(key) > 1 else 1.2
                cv2.putText(img, key, (x + 15, y + 55), cv2.FONT_HERSHEY_DUPLEX, font_scale, (255, 255, 255), 2)
        
        # Text Bar
        cv2.rectangle(img, (self.start_x, self.start_y - 100), (self.start_x + 900, self.start_y - 20), (20, 20, 20), cv2.FILLED)
        cv2.rectangle(img, (self.start_x, self.start_y - 100), (self.start_x + 900, self.start_y - 20), (0, 242, 254), 2)
        cv2.putText(img, self.text + "|", (self.start_x + 20, self.start_y - 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        return img

    def _get_dist(self, p1, p2):
        return np.hypot(p1[0]-p2[0], p1[1]-p2[1])

    def _on_key_press(self, key):
        if time.time() - self.last_press_time > self.cooldown:
            if key == "CLR":
                self.text = ""
            else:
                self.text += key
            self.last_press_time = time.time()
