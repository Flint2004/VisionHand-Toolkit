import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os

class HandTracker:
    def __init__(self, model_path="hand_landmarker.task", max_hands=2):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.results = None

    def find_hands(self, img, draw=True):
        """
        Detects hands using Mediapipe Tasks API.
        Returns a list of 'hand' dictionaries.
        """
        h, w, _ = img.shape
        # Convert the image to MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # Detect landmarks
        self.results = self.detector.detect(mp_image)
        
        hands_data = []
        if self.results.hand_landmarks:
            for i, (landmarks, handedness) in enumerate(zip(self.results.hand_landmarks, self.results.handedness)):
                hand = {}
                lm_list = []
                for id, lm in enumerate(landmarks):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append([id, cx, cy, lm.z])
                
                hand['lmList'] = lm_list
                hand['type'] = handedness[0].category_name
                hand['center'] = (lm_list[9][1], lm_list[9][2]) 
                
                # Bounding box
                x_coords = [lm[1] for lm in lm_list]
                y_coords = [lm[2] for lm in lm_list]
                hand['bbox'] = (min(x_coords), min(y_coords), max(x_coords) - min(x_coords), max(y_coords) - min(y_coords))
                
                hands_data.append(hand)
                
                if draw:
                    # Draw landmarks and connections
                    for lm in lm_list:
                        cv2.circle(img, (lm[1], lm[2]), 5, (255, 0, 255), cv2.FILLED)
                    
                    # Connection mapping (hand landmarks)
                    connections = [
                        (0, 1), (1, 2), (2, 3), (3, 4),
                        (0, 5), (5, 6), (6, 7), (7, 8),
                        (5, 9), (9, 10), (10, 11), (11, 12),
                        (9, 13), (13, 14), (14, 15), (15, 16),
                        (13, 17), (17, 18), (18, 19), (19, 20),
                        (0, 17)
                    ]
                    for start, end in connections:
                        c1 = lm_list[start]
                        c2 = lm_list[end]
                        cv2.line(img, (c1[1], c1[2]), (c2[1], c2[2]), (0, 255, 0), 2)

        return hands_data, img

    def get_finger_status(self, hand):
        """
        Custom fingersUp implementation for Mediapipe Tasks landmarks.
        0: Thumb, 1: Index, 2: Middle, 3: Ring, 4: Pinky
        1 for UP, 0 for DOWN.
        """
        fingers = []
        lmList = hand['lmList']
        tipIds = [4, 8, 12, 16, 20]

        # Thumb (Horizontal comparison for Right/Left hand)
        if hand['type'] == "Right": # Right hand in mirror (actually left hand?)
            if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
        else: # Left hand in mirror
            if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

        # 4 Fingers
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    def get_distance(self, p1, p2, hand, img=None, draw=True):
        """
        Calculates distance between two landmarks in a specific hand.
        """
        lm1 = hand['lmList'][p1]
        lm2 = hand['lmList'][p2]
        x1, y1 = lm1[1], lm1[2]
        x2, y2 = lm2[1], lm2[2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
        
        info = [x1, y1, x2, y2, cx, cy]
        if img is not None and draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (x1, y1), 8, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 8, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), 8, (0, 0, 255), cv2.FILLED)
            
        return length, info, img
