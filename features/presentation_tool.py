import os
import cv2
import numpy as np
import time

class PresentationTool:
    def __init__(self, folder_path="images"):
        self.folder_path = folder_path
        self.slides = []
        self.current_idx = 0
        self.visible = True
        self.load_slides()
        
        self.last_pos = None
        self.last_time = 0
        self.swipe_velocity_threshold = 1.8 # Pixels per millisecond
        self.swipe_cooldown = 0.5 # Seconds
        self.last_swipe_time = 0
        self.opacity = 0.8

    def load_slides(self):
        if not os.path.exists(self.folder_path): return
        exts = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
        files = [f for f in os.listdir(self.folder_path) if f.lower().endswith(exts)]
        self.slides = []
        for f in sorted(files):
            img = cv2.imread(os.path.join(self.folder_path, f), cv2.IMREAD_UNCHANGED)
            if img is None: img = cv2.imread(os.path.join(self.folder_path, f))
            if img is not None: self.slides.append(img)
        print(f"PresentationTool: Loaded {len(self.slides)} images.")

    def update_gestures(self, hand):
        fingers = hand['fingers']
        curr_time = time.time()
        
        # 1. Visibility Handling (Palm to Show, Fist/Thumb-only to Hide)
        if fingers == [1, 1, 1, 1, 1]: # Palm
            self.visible = True
        elif fingers[1:] == [0, 0, 0, 0]: # Hide if digits are down (thumb can be up/down)
            self.visible = False
            
        # 2. Velocity-based Swipe Detection (only if visible and palm)
        if self.visible and fingers == [1, 1, 1, 1, 1]:
            curr_x = hand['landmarks'][9][0]
            if self.last_pos is not None and (curr_time - self.last_swipe_time > self.swipe_cooldown):
                dt = (curr_time - self.last_time) * 1000 # ms
                if dt > 0:
                    velocity = (curr_x - self.last_pos) / dt
                    if abs(velocity) > self.swipe_velocity_threshold:
                        if len(self.slides) > 1:
                            if velocity > 0: self.current_idx = (self.current_idx - 1) % len(self.slides)
                            else: self.current_idx = (self.current_idx + 1) % len(self.slides)
                            print(f"Slide Switched: {self.current_idx} (Vel: {velocity:.2f})")
                            self.last_swipe_time = curr_time
            
            self.last_pos = curr_x
            self.last_time = curr_time
        else:
            self.last_pos = None

    def draw(self, frame, scale=1.0, offset=(0,0)):
        if not self.slides or not self.visible: return frame

        slide = self.slides[self.current_idx]
        fh, fw = frame.shape[:2]
        
        # Aspect Ratio Fit
        sh, sw = slide.shape[:2]
        aspect = sw / sh
        if fw / fh > aspect:
            base_h, base_w = fh * 0.9, (fh * 0.9) * aspect
        else:
            base_w, base_h = fw * 0.9, (fw * 0.9) / aspect
            
        nw, nh = int(base_w * scale), int(base_h * scale)
        if nw <= 0 or nh <= 0: return frame
        slide_resized = cv2.resize(slide, (nw, nh))
        
        # Center + Offset
        x1, y1 = fw // 2 - nw // 2 + int(offset[0]), fh // 2 - nh // 2 + int(offset[1])
        x2, y2 = x1 + nw, y1 + nh
        
        # Clipping
        ox1, oy1, ox2, oy2 = max(0, x1), max(0, y1), min(fw, x2), min(fh, y2)
        if ox1 >= ox2 or oy1 >= oy2: return frame
        
        slide_part = slide_resized[oy1-y1:oy2-y1, ox1-x1:ox2-x1]
        roi = frame[oy1:oy2, ox1:ox2]
        
        if slide_part.shape[2] == 4:
            alpha = (slide_part[:, :, 3] / 255.0) * self.opacity
            for c in range(3):
                roi[:, :, c] = (alpha * slide_part[:, :, c] + (1 - alpha) * roi[:, :, c]).astype(np.uint8)
        else:
            cv2.addWeighted(slide_part, self.opacity, roi, 1 - self.opacity, 0, roi)
        return frame
