import os
import cv2
import numpy as np
import time
from collections import deque

class PresentationTool:
    def __init__(self, folder_path="images", use_kia=False):
        self.folder_path = folder_path
        self.slides = []
        self.current_idx = 0
        self.visible = True
        self.load_slides()
        
        self.use_kia = use_kia
        
        # KIA (Kinetic Intent Analysis) Parameters
        self.history = deque(maxlen=8)
        self.kia_threshold = 0.4       # Increased sensitivity (was 0.8)
        self.consensus_req = 0.7
        
        # Basic Swipe Parameters (Used if kia=False)
        self.start_x = None
        self.swipe_threshold_ratio = 1.2 # Threshold relative to hand scale
        
        self.swipe_cooldown = 0.6
        self.last_swipe_time = 0
        self.opacity = 1.0             # 100% Opacity as requested

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
        if self.visible and fingers == [0, 1, 1, 1, 1]:
            if self.use_kia:
                # KIA Logic
                curr_pos = np.array(hand['landmarks'][9][:2])
                scale = hand.get('scale', 100)
                self.history.append(curr_pos)
                
                if len(self.history) == self.history.maxlen and (curr_time - self.last_swipe_time > self.swipe_cooldown):
                    vectors = []
                    for i in range(1, len(self.history)):
                        vectors.append((self.history[i] - self.history[i-1]) / scale)
                    
                    dx_sum = sum(v[0] for v in vectors)
                    directions = [np.sign(v[0]) for v in vectors if abs(v[0]) > 0.01]
                    
                    if directions:
                        most_common_dir = np.sign(dx_sum)
                        consensus = directions.count(most_common_dir) / len(directions)
                        
                        if consensus >= self.consensus_req and abs(dx_sum) > self.kia_threshold:
                            if len(self.slides) > 1:
                                if dx_sum > 0: self.current_idx = (self.current_idx - 1) % len(self.slides)
                                else: self.current_idx = (self.current_idx + 1) % len(self.slides)
                                print(f"KIA Swipe Triggered: {self.current_idx}")
                            self.last_swipe_time = curr_time
                            self.history.clear()
            else:
                # Basic Displacement Logic (Adaptive)
                curr_x = hand['landmarks'][9][0]
                scale = hand.get('scale', 100)
                if self.last_swipe_time + self.swipe_cooldown < curr_time:
                    if self.start_x is None:
                        self.start_x = curr_x
                    else:
                        dx = curr_x - self.start_x
                        # Adaptive threshold: 1.2 * hand size
                        if abs(dx) > (self.swipe_threshold_ratio * scale):
                            if len(self.slides) > 1:
                                if dx > 0: self.current_idx = (self.current_idx - 1) % len(self.slides)
                                else: self.current_idx = (self.current_idx + 1) % len(self.slides)
                                print(f"Basic Swipe Triggered: {self.current_idx}")
                            self.last_swipe_time = curr_time
                            self.start_x = None
                else:
                    self.start_x = None
        else:
            self.history.clear()
            self.start_x = None

    def draw(self, frame, scale=1.0, offset=(0,0), opacity=None):
        if not self.slides or not self.visible: return frame

        # Use custom opacity if provided, else fall back to default
        active_opacity = opacity if opacity is not None else self.opacity
        
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
        
        # Weighted Blending
        if slide_part.shape[2] == 4: # Source Alpha Support
            alpha = (slide_part[:, :, 3] / 255.0) * active_opacity
            for c in range(3):
                roi[:, :, c] = (alpha * slide_part[:, :, c] + (1 - alpha) * roi[:, :, c]).astype(np.uint8)
        else:
            cv2.addWeighted(slide_part, active_opacity, roi, 1 - active_opacity, 0, roi)
            
        return frame
