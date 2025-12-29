import math
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, QPoint, QRectF
from PySide6.QtGui import QPainter, QColor, QRadialGradient, QFont, QPen, QBrush
from PySide6.QtSvg import QSvgRenderer
from utils.theme import THEME, ICONS, RADIAL_RADIUS, RADIAL_INNER_RADIUS

class RadialMenuWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.hide()
        
        self.center_fixed = QPoint(0, 0)
        self.hand_pos = QPoint(0, 0)
        self.current_angle = 0 # 0-360, South=90
        self.active_sector = -1
        
        # Consistent mapping with GestureEngine
        # 0:East, 1:South, 2:West, 3:North
        self.tool_names = ["KEYBOARD", "PAINTER_ALT", "MEDIA", "PAINTER"]
        
        self.renderers = {
            "PAINTER": self._load_svg(ICONS["paint"]),
            "PAINTER_ALT": self._load_svg(ICONS["paint"]),
            "MEDIA": self._load_svg(ICONS["media"]),
            "KEYBOARD": self._load_svg(ICONS["keyboard"])
        }

    def _load_svg(self, svg_str):
        return QSvgRenderer(svg_str.encode())

    def show_at(self, pos):
        self.center_fixed = pos
        self.setGeometry(pos.x() - RADIAL_RADIUS, pos.y() - RADIAL_RADIUS, 
                         RADIAL_RADIUS * 2, RADIAL_RADIUS * 2)
        self.show()

    def update_state(self, hand_pos, angle):
        self.hand_pos = hand_pos
        self.current_angle = angle
        
        # Logic matches GestureEngine: 0:East, 90:South, 180:West, 270:North
        if 45 <= angle < 135: self.active_sector = 1   # South
        elif 135 <= angle < 225: self.active_sector = 2 # West
        elif 225 <= angle < 315: self.active_sector = 3 # North
        else: self.active_sector = 0                   # East
        
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        cx, cy = RADIAL_RADIUS, RADIAL_RADIUS
        
        # 1. Background
        painter.setBrush(QColor(10, 10, 10, 180))
        painter.setPen(QPen(QColor(255, 255, 255, 50), 2))
        painter.drawEllipse(5, 5, RADIAL_RADIUS*2-10, RADIAL_RADIUS*2-10)

        # 2. Draw Sectors
        # In Screen Coords (Y-down): East=0, South=90, West=180, North=270
        # Qt drawPie angles: 0 is East, but CCW. So 90 is North.
        # To draw my "South" (90), I need to draw at Qt's 270 deg.
        for i in range(4):
            engine_center_angle = i * 90
            qt_start_angle = (360 - (engine_center_angle + 45)) % 360
            
            is_active = (i == self.active_sector)
            
            if is_active:
                # Use Alt Color for South (Index 1)
                color_hex = THEME["active_alt"] if i == 1 else THEME["active"]
                color = QColor(color_hex)
                color.setAlpha(150)
                painter.setBrush(color)
                # Draw CCW 90 deg from flipped start
                painter.drawPie(8, 8, RADIAL_RADIUS*2-16, RADIAL_RADIUS*2-16, qt_start_angle * 16, 90 * 16)
            
            # Draw Divider
            painter.setPen(QPen(QColor(255, 255, 255, 40), 1))
            sep_angle_rad = math.radians(engine_center_angle - 45)
            painter.drawLine(cx, cy, cx + RADIAL_RADIUS * math.cos(sep_angle_rad), 
                             cy + RADIAL_RADIUS * math.sin(sep_angle_rad))
            
            self._draw_icon(painter, i, is_active)

        # 3. Inner Mask
        painter.setBrush(QColor(20, 20, 20))
        painter.setPen(QPen(QColor(THEME["active"]), 2))
        painter.drawEllipse(cx - RADIAL_INNER_RADIUS, cy - RADIAL_INNER_RADIUS, 
                            RADIAL_INNER_RADIUS * 2, RADIAL_INNER_RADIUS * 2)

    def _draw_icon(self, painter, index, active):
        radius = (RADIAL_RADIUS + RADIAL_INNER_RADIUS) / 2
        engine_angle_rad = math.radians(index * 90)
        ix = RADIAL_RADIUS + radius * math.cos(engine_angle_rad) - 25
        iy = RADIAL_RADIUS + radius * math.sin(engine_angle_rad) - 25
        
        tool = self.tool_names[index]
        painter.setOpacity(1.0 if active else 0.5)
        self.renderers[tool].render(painter, QRectF(ix, iy, 50, 50))
        painter.setOpacity(1.0)
