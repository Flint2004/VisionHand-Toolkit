from utils.config import ZOOM_SENSITIVITY, MIN_PINCH_DISTANCE

class ZoomController:
    def __init__(self):
        self.scale = 1.0
        self.last_dist = None

    def update(self, dist, is_pinching):
        """
        Updates the scale based on pinch distance changes.
        """
        if is_pinching:
            if self.last_dist is not None:
                diff = dist - self.last_dist
                self.scale += diff * ZOOM_SENSITIVITY
                self.scale = max(0.5, min(self.scale, 5.0)) # Clamp scale
            self.last_dist = dist
        else:
            self.last_dist = None
        
        return self.scale
