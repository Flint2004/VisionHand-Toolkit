import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time

class VisionEngine:
    def __init__(self, model_path="hand_landmarker.task", draw_landmarks=True):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.last_timestamp = 0
        self.draw_landmarks = draw_landmarks

    def process_frame(self, img):
        """
        Processes a frame and returns hand data.
        """
        h, w, _ = img.shape
        timestamp = int(time.time() * 1000)
        if timestamp <= self.last_timestamp:
            timestamp = self.last_timestamp + 1
        self.last_timestamp = timestamp

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        result = self.detector.detect_for_video(mp_image, timestamp)
        
        hands_data = []
        if result.hand_landmarks:
            for i, (landmarks, handedness) in enumerate(zip(result.hand_landmarks, result.handedness)):
                lms = [(int(lm.x * w), int(lm.y * h), lm.z) for lm in landmarks]
                hand = {
                    'type': handedness[0].category_name,
                    'landmarks': lms,
                    'raw_landmarks': landmarks
                }
                hand['fingers'] = self._get_fingers(lms, hand['type'])
                hands_data.append(hand)

                if self.draw_landmarks:
                    self._draw_landmarks_and_connections(img, lms)
                
        return hands_data

    def _draw_landmarks_and_connections(self, img, lms):
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17)
        ]
        for start, end in connections:
            cv2.line(img, (lms[start][0], lms[start][1]), (lms[end][0], lms[end][1]), (0, 255, 0), 2)
        for lm in lms:
            cv2.circle(img, (lm[0], lm[1]), 5, (255, 0, 255), cv2.FILLED)

    def _get_fingers(self, lms, hand_type):
        fingers = []
        tip_ids = [4, 8, 12, 16, 20]
        
        # Thumb
        if hand_type == "Right":
            fingers.append(1 if lms[tip_ids[0]][0] < lms[tip_ids[0]-1][0] else 0)
        else:
            fingers.append(1 if lms[tip_ids[0]][0] > lms[tip_ids[0]-1][0] else 0)
            
        # 4 Fingers
        for id in range(1, 5):
            fingers.append(1 if lms[tip_ids[id]][1] < lms[tip_ids[id]-2][1] else 0)
            
        return fingers
