import sys
import cv2
import numpy as np
import argparse
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QStackedWidget
from PySide6.QtCore import Qt, QTimer, QPoint
from PySide6.QtGui import QImage, QPixmap

from engine.vision_engine import VisionEngine
from engine.vision_engine import VisionEngine
from engine.gesture_engine import GestureEngine
from ui.radial_widget import RadialMenuWidget
from ui.overlay_canvas import OverlayCanvas
from features.keyboard_tool import VirtualKeyboard
from features.zoom_tool import ZoomTool
from features.presentation_tool import PresentationTool
from utils.theme import SCREEN_SIZE

class AIModernPainter(QMainWindow):
    def __init__(self, show_landmarks=True, use_gpu=False, use_smooth=False, adaptive=False):
        super().__init__()
        self.setWindowTitle("AI Modern Virtual Painter - Pro Edition")
        self.setFixedSize(SCREEN_SIZE[0], SCREEN_SIZE[1])
        self.adaptive = adaptive
        
        # 1. Video Layer
        self.video_label = QLabel(self)
        self.video_label.setFixedSize(self.size())
        
        # 2. Drawing Layer
        self.canvas = OverlayCanvas(self)
        
        # 3. Radial Menu Layer
        self.radial_menu = RadialMenuWidget(self)
        
        # Tools
        self.vision = VisionEngine(draw_landmarks=show_landmarks, 
                                   use_gpu=use_gpu, 
                                   use_smoothing=use_smooth)
        self.gestures = GestureEngine()
        self.keyboard = VirtualKeyboard()
        self.zoom_tool = ZoomTool()
        self.present_tool = PresentationTool()
        
        # App State
        self.current_tool = "PAINTER"
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_SIZE[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_SIZE[1])
        
        # Main Loop
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(16) # ~60 FPS

    def update_frame(self):
        success, frame = self.cap.read()
        if not success: return
        
        frame = cv2.flip(frame, 1)
        hands = self.vision.process_frame(frame)
        
        # 1. Global Slides Rendering (Persistent visibility + Zoom state)
        frame = self.present_tool.draw(frame, 
                                       scale=self.zoom_tool.scale,
                                       offset=self.zoom_tool.offset)
        
        # 2. Gesture Analysis
        state = self.gestures.update_state(hands)
        
        if hands:
            hand = hands[0]
            index_pos = QPoint(hand['landmarks'][8][0], hand['landmarks'][8][1])
            
            if state == "MENU_OPENED":
                self.radial_menu.show_at(index_pos)
            elif state == "MENU_ACTIVE":
                angle, dist = self.gestures.get_current_angle_and_dist((index_pos.x(), index_pos.y()))
                self.radial_menu.update_state(index_pos, angle)
            elif state == "SELECTED":
                self.radial_menu.hide()
                if self.gestures.selected_tool:
                    self.current_tool = self.gestures.selected_tool
                    print(f"Tool Switched to: {self.current_tool}")
            elif state == "IDLE":
                frame = self._handle_tool_logic(frame, hand)
        
        # 3. Localized Clearing (Only in Painter modes, clears specific layer)
        if hands and hands[0]['fingers'] == [1, 1, 1, 1, 1]:
            if self.current_tool in ["PAINTER", "PAINTER_ALT"]:
                self.canvas.clear_layer(self.current_tool)

        # UI Rendering
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        qi = QImage(rgb_frame.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qi))

    def _handle_tool_logic(self, frame, hand):
        x, y = hand['landmarks'][8][:2]
        fingers = hand['fingers']
        
        if self.current_tool == "PAINTER":
            # Cyan (BGR) | Condition: Thumb Closed (1), Index Up (1), Middle Closed (0)
            is_drawing = (fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 0)
            self.canvas.draw_line(x, y, is_drawing, tool_name="PAINTER", color=(254, 242, 0, 255))
        elif self.current_tool == "PAINTER_ALT":
            # Vivid Pink (BGR) | Condition: Thumb Closed (1), Index Up (1), Middle Closed (0)
            is_drawing = (fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 0)
            self.canvas.draw_line(x, y, is_drawing, tool_name="PAINTER_ALT", color=(128, 0, 255, 255))
        elif self.current_tool == "KEYBOARD":
            frame = self.keyboard.draw(frame, [hand])
        elif self.current_tool == "MEDIA":
            # Combined Zoom + Presentation
            # 1. Zoom Logic
            raw_dist, center, is_pinching, norm_dist = self.zoom_tool.get_pinch_data(hand)
            # If adaptive is True, use normalized distance (scaled up) to maintain sensitivity
            dist_to_use = norm_dist * 150 if self.adaptive else raw_dist
            scale, offset = self.zoom_tool.update(dist_to_use, center, is_pinching)
            cv2.putText(frame, f"Media Mode | Scale: {scale:.2f}x", 
                        (50, 650), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 242, 254), 2)
            
            # 2. Presentation Logic (Visibility + Swipe)
            self.present_tool.update_gestures(hand)
        
        return frame

    def closeEvent(self, event):
        self.cap.release()
        event.accept()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Modern Virtual Painter - Pro Edition")
    parser.add_argument("--hide-landmarks", action="store_true")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU acceleration")
    parser.add_argument("--smooth", action="store_true", help="Enable OneEuro smoothing")
    parser.add_argument("--adaptive", action="store_true", help="Enable distance-adaptive thresholds")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    window = AIModernPainter(show_landmarks=not args.hide_landmarks,
                             use_gpu=args.gpu,
                             use_smooth=args.smooth,
                             adaptive=args.adaptive)
    window.show()
    sys.exit(app.exec())
