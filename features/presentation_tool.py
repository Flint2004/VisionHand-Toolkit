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
        
        # Swipe tracking
        self.start_x = None
        self.swipe_threshold = 150 # Absolute pixels for swipe
        self.swipe_cooldown = 0.6
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
        
        # 1. Visibility Logic (Palm to Show, Fist/Thumb-only to Hide)
        if fingers == [1, 1, 1, 1, 1]: # Palm
            self.visible = True
        elif fingers[1:] == [0, 0, 0, 0]: # Hide if 4 digits are closed (Thumb can be anything)
            self.visible = False
            
        # 2. Swipe Detection (Only if visible and Palm gesture)
        if self.visible and fingers == [1, 1, 1, 1, 1]:
            curr_x = hand['landmarks'][9][0] # Middle knuckle
            
            if self.last_swipe_time + self.swipe_cooldown < curr_time:
                if self.start_x is None:
                    self.start_x = curr_x
                else:
                    dx = curr_x - self.start_x
                    # We need a significant displacement to trigger
                    if abs(dx) > self.swipe_threshold:
                        if len(self.slides) > 1:
                            if dx > 0: self.current_idx = (self.current_idx - 1) % len(self.slides)
                            else: self.current_idx = (self.current_idx + 1) % len(self.slides)
                            print(f"Slide Switched: {self.current_idx}")
                        
                        self.last_swipe_time = curr_time
                        self.start_x = None # Reset
            else:
                self.start_x = None # In cooldown
        else:
            self.start_x = None

    def draw(self, frame, scale=1.0, offset=(0,0)):
        if not self.slides or not self.visible: return frame

        slide = self.slides[self.current_idx]
        fh, fw = frame.shape[:2]
        
        # Aspect Ratio Fit with Zoom Scale
        sh, sw = slide.shape[:2]
        aspect = sw / sh
        if fw / fh > aspect:
            base_h, base_w = fh * 0.9, (fh * 0.9) * aspect
        else:
            base_w, base_h = fw * 0.9, (fw * 0.9) / aspect
            
        nw, nh = int(base_w * scale), int(base_h * scale)
        if nw <= 0 or nh <= 0: return frame
        slide_resized = cv2.resize(slide, (nw, nh))
        
        # Centering and Offset
        x1 = fw // 2 - nw // 2 + int(offset[0])
        y1 = fh // 2 - nh // 2 + int(offset[1])
        x2, y2 = x1 + nw, y1 + nh
        
        # Dynamic Boundary Clipping
        ox1, oy1 = max(0, x1), max(0, y1)
        ox2, oy2 = min(fw, x2), min(fh, y2)
        
        if ox1 >= ox2 or oy1 >= oy2: return frame
        
        # Source ROI within the (resized) slide
        sx1, sy1 = ox1 - x1, oy1 - y1
        sx2, sy2 = sx1 + (ox2 - ox1), sy1 + (oy2 - oy1)
        
        slide_part = slide_resized[sy1:sy2, sx1:sx2]
        roi = frame[oy1:oy2, ox1:ox2]
        
        # Weighted Blending (80% opacity)
        if slide_part.shape[2] == 4: # Source Alpha Support
            alpha = (slide_part[:, :, 3] / 255.0) * self.opacity
            for c in range(3):
                roi[:, :, c] = (alpha * slide_part[:, :, c] + (1 - alpha) * roi[:, :, c]).astype(np.uint8)
        else:
            cv2.addWeighted(slide_part, self.opacity, roi, 1 - self.opacity, 0, roi)
            
        return frame
