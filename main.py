import sys
import cv2
import numpy as np
import argparse
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QStackedWidget
from PySide6.QtCore import Qt, QTimer, QPoint
from PySide6.QtGui import QImage, QPixmap

from engine.vision_engine import VisionEngine
from engine.gesture_engine import GestureEngine
from ui.radial_widget import RadialMenuWidget
from ui.overlay_canvas import OverlayCanvas
from features.keyboard_tool import VirtualKeyboard
from features.zoom_tool import ZoomTool
from features.presentation_tool import PresentationTool
from utils.theme import SCREEN_SIZE

class AudienceWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Virtual Painter - Audience View")
        self.resize(SCREEN_SIZE[0], SCREEN_SIZE[1])
        self.label = QLabel(self)
        self.label.setScaledContents(True)

    def resizeEvent(self, event):
        # Lock 16:9 Aspect Ratio
        w = event.size().width()
        h = int(w * 9 / 16)
        if h != event.size().height():
            self.resize(w, h)
        self.label.setFixedSize(self.size())
        super().resizeEvent(event)

class AIModernPainter(QMainWindow):
    def __init__(self, show_landmarks=True, use_gpu=False, use_smooth=False, adaptive=False, dual_window=False, use_kia=False):
        super().__init__()
        self.setWindowTitle("AI Modern Virtual Painter - Pro Edition")
        self.resize(SCREEN_SIZE[0], SCREEN_SIZE[1])
        self.adaptive = adaptive
        
        # Dual Window Support
        self.audience_win = AudienceWindow() if dual_window else None
        if self.audience_win:
            self.audience_win.show()
        
        # 1. Video Layer
        self.video_label = QLabel(self)
        self.video_label.setScaledContents(True)
        
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
        self.present_tool = PresentationTool(use_kia=use_kia)
        
        # App State
        self.current_tool = "PAINTER"
        self.brush_thickness = 10
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
        
        # Prepare Audience Frame (Clean + 100% Opacity)
        if self.audience_win:
            clean_frame = frame.copy()
            clean_frame = self.present_tool.draw(clean_frame, 
                                                scale=self.zoom_tool.scale, 
                                                offset=self.zoom_tool.offset,
                                                opacity=1.0)
            # Overlay drawings manually on clean frame
            for layer_name in ["PAINTER", "PAINTER_ALT"]:
                layer = self.canvas.layers[layer_name]
                mask = layer[:, :, 3] > 0
                clean_frame[mask] = layer[mask, :3]
            
            self._show_on_label(self.audience_win.label, clean_frame)

        hands = self.vision.process_frame(frame)
        
        # 1. Global Slides Rendering (Operator View - 60% Opacity)
        frame = self.present_tool.draw(frame, 
                                       scale=self.zoom_tool.scale,
                                       offset=self.zoom_tool.offset,
                                       opacity=0.6)
        
        # 2. Gesture Analysis
        state = self.gestures.update_state(hands)
        
        if hands:
            hand = hands[0]
            # Internal coordinates (1280x720)
            ix, iy = hand['landmarks'][8][0], hand['landmarks'][8][1]
            
            # Map to Window Coordinates for UI elements (Dynamic Scaling)
            sw, sh = self.width() / SCREEN_SIZE[0], self.height() / SCREEN_SIZE[1]
            win_pos = QPoint(int(ix * sw), int(iy * sh))
            
            if state == "MENU_OPENED":
                self.radial_menu.show_at(win_pos)
            elif state == "MENU_ACTIVE":
                angle, dist = self.gestures.get_current_angle_and_dist((ix, iy))
                self.radial_menu.update_state(win_pos, angle)
            elif state == "SELECTED":
                self.radial_menu.hide()
                if self.gestures.selected_tool:
                    self.current_tool = self.gestures.selected_tool
                    print(f"Tool Switched to: {self.current_tool}")
            elif state == "IDLE":
                frame = self._handle_tool_logic(frame, hand)
        
        # 3. Localized Clearing (Only in Painter modes, clears specific layer)
        if hands and hands[0]['fingers'] == [0, 1, 1, 1, 1]:
            if self.current_tool in ["PAINTER", "PAINTER_ALT"]:
                self.canvas.clear_layer(self.current_tool)

        # UI Rendering
        self._show_on_label(self.video_label, frame)

    def _show_on_label(self, label, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        qi = QImage(rgb_frame.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qi)
        # Scaled contents is on, but we want smooth scaling
        label.setPixmap(pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        # Lock 16:9 Aspect Ratio
        w = event.size().width()
        h = int(w * 9 / 16)
        if h != event.size().height():
            self.resize(w, h)
        
        self.video_label.setFixedSize(self.size())
        self.canvas.setFixedSize(self.size())
        self.radial_menu.setFixedSize(self.size())
        super().resizeEvent(event)

    def _handle_tool_logic(self, frame, hand):
        x, y = hand['landmarks'][8][:2]
        fingers = hand['fingers']
        
        # Consistent Drawing Condition: Index Up, Middle Down (Thumb controls thickness mode)
        drawing_gest = (fingers[1] == 1 and fingers[2] == 0)
        
        if self.current_tool == "PAINTER":
            # Cyan (BGR) 
            if fingers[0] == 0:
                _, _, _, norm_dist = self.zoom_tool.get_pinch_data(hand)
                # Highly Sensitive Mapping: 2px - 80px range
                self.brush_thickness = int(np.clip((norm_dist - 0.05) * 150, 2, 100) * 0.2)
                status = f"Size: {self.brush_thickness}"
                is_drawing = False # Disable drawing while sizing
            else:
                status = f"Locked: {self.brush_thickness}"
                is_drawing = drawing_gest
            
            self.canvas.draw_line(x, y, is_drawing, tool_name="PAINTER", 
                                 color=(254, 242, 0, 255), thickness=self.brush_thickness)
            # Visual Feedback (Operator only)
            cv2.circle(frame, (x, y), self.brush_thickness//2, (254, 242, 0), 2)
            cv2.putText(frame, status, (x + 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (254, 242, 0), 1)
            
        elif self.current_tool == "PAINTER_ALT":
            # Vivid Pink Highlighter (BGR + Alpha)
            if fingers[0] == 0:
                _, _, _, norm_dist = self.zoom_tool.get_pinch_data(hand)
                # Highly Sensitive Mapping: 2px - 80px range
                self.brush_thickness = int(np.clip((norm_dist - 0.05) * 150, 2, 80) * 0.5)
                status = f"Highlighter: {self.brush_thickness}"
                is_drawing = False # Disable drawing while sizing
            else:
                status = f"Locked: {self.brush_thickness}"
                is_drawing = drawing_gest
            
            # Semi-transparent alpha (120) for fluorescent effect
            self.canvas.draw_line(x, y, is_drawing, tool_name="PAINTER_ALT", 
                                 color=(128, 0, 255, 120), thickness=self.brush_thickness)
            # Visual Feedback (Operator only)
            cv2.circle(frame, (x, y), self.brush_thickness//2, (128, 0, 255), 2)
            cv2.putText(frame, status, (x + 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 0, 255), 1)
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
    parser.add_argument("--dual", action="store_true", help="Enable dual-window mode (Clean Audience View)")
    parser.add_argument("--kia", action="store_true", help="Enable Kinetic Intent Analysis for swipes")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    window = AIModernPainter(show_landmarks=not args.hide_landmarks,
                             use_gpu=args.gpu,
                             use_smooth=args.smooth,
                             adaptive=args.adaptive,
                             dual_window=args.dual,
                             use_kia=args.kia)
    window.show()
    sys.exit(app.exec())
