import numpy as np

class ZoomTool:
    def __init__(self):
        self.scale = 1.0
        self.offset = [0, 0] # [x, y]
        
        self.last_dist = None
        self.last_center = None
        self.sensitivity_scale = 0.005
        self.sensitivity_move = 1.0

    def get_pinch_data(self, hand):
        p1 = np.array(hand['landmarks'][4][:2]) # Thumb
        p2 = np.array(hand['landmarks'][8][:2]) # Index
        
        raw_dist = np.linalg.norm(p1 - p2)
        # Normalize distance based on hand scale
        scale = hand.get('scale', 100)
        norm_dist = raw_dist / scale
        
        center = (p1 + p2) / 2
        is_pinching = (hand['fingers'] == [1, 1, 0, 0, 0])
        return raw_dist, center, is_pinching, norm_dist

    def update(self, dist, center, is_pinching):
        if is_pinching:
            # 1. Scaling
            if self.last_dist is not None:
                diff_dist = dist - self.last_dist
                self.scale = max(0.1, min(self.scale + diff_dist * self.sensitivity_scale, 10.0))
            
            # 2. Translation (Moving)
            if self.last_center is not None:
                dx = center[0] - self.last_center[0]
                dy = center[1] - self.last_center[1]
                self.offset[0] += dx * self.sensitivity_move
                self.offset[1] += dy * self.sensitivity_move
            
            self.last_dist = dist
            self.last_center = center
        else:
            self.last_dist = None
            self.last_center = None
            
        return self.scale, self.offset
