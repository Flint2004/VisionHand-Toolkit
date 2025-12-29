import numpy as np
import time

class GestureEngine:
    def __init__(self):
        self.menu_active = False
        self.menu_center = None
        self.selected_tool = None
        self.trigger_start_time = None
        self.last_state = "IDLE"
        
        # Tool Map based on angles
        self.tools = ["ZOOM", "KEYBOARD", "PRESENT", "PAINTER"] 

    def update_state(self, hands):
        """
        Determines the state of the gesture interaction.
        Trigger: Thumb, Index, Middle [1, 1, 1, 0, 0]
        """
        if not hands:
            if self.menu_active:
                self.menu_active = False
                self._reset_trigger()
                return "SELECTED" # Release on hand loss
            self._reset_trigger()
            return "IDLE"

        hand = hands[0]
        fingers = hand['fingers']
        index_pos = hand['landmarks'][8][:2]

        # 1. Trigger Pulse logic
        is_trigger_gesture = (fingers == [1, 1, 1, 0, 0])
        
        if is_trigger_gesture:
            if not self.menu_active:
                if self.trigger_start_time is None:
                    self.trigger_start_time = time.time()
                elif time.time() - self.trigger_start_time > 0.4:
                    self.menu_active = True
                    self.menu_center = index_pos
                    return "MENU_OPENED"
                return "TRIGGERING"
        
        # Selection logic
        if self.menu_active:
            # While the trigger gesture is held, we are in MENU_ACTIVE
            if is_trigger_gesture:
                return "MENU_ACTIVE"
            else:
                # Gesture released -> Selection confirmed
                self.menu_active = False
                selection = self._calculate_selection(index_pos)
                self.selected_tool = selection
                self._reset_trigger()
                return "SELECTED"

        self._reset_trigger()
        return "IDLE"

    def _reset_trigger(self):
        self.trigger_start_time = None

    def _calculate_selection(self, current_pos):
        if self.menu_center is None: return None
        
        dx = current_pos[0] - self.menu_center[0]
        dy = current_pos[1] - self.menu_center[1]
        dist = np.hypot(dx, dy)
        
        if dist < 50: return None # Deadzone
        
        angle = np.degrees(np.arctan2(dy, dx)) % 360
        
        # East: 315-45, South: 45-135, West: 135-225, North: 225-315
        if 45 <= angle < 135: return "PAINTER_ALT" # South
        if 135 <= angle < 225: return "MEDIA"        # West
        if 225 <= angle < 315: return "PAINTER"      # North
        return "KEYBOARD"                           # East

    def get_current_angle_and_dist(self, current_pos):
        if self.menu_center is None: return 0, 0
        dx = current_pos[0] - self.menu_center[0]
        dy = current_pos[1] - self.menu_center[1]
        angle = np.degrees(np.arctan2(dy, dx)) % 360
        dist = np.hypot(dx, dy)
        return angle, dist
