import math
import time

class OneEuroFilter:
    def __init__(self, freq, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.freq = float(freq)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = None
        self.dx_prev = 0.0

    def _alpha(self, cutoff):
        te = 1.0 / self.freq
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def filter(self, x, freq=None):
        if freq is not None:
            self.freq = freq
        
        if self.x_prev is None:
            self.x_prev = x
            return x

        te = 1.0 / self.freq
        ad = self._alpha(self.d_cutoff)
        dx = (x - self.x_prev) / te
        dx_hat = ad * dx + (1.0 - ad) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff)
        x_hat = a * x + (1.0 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat

class LandmarkSmoother:
    def __init__(self, num_landmarks=21, freq=30, min_cutoff=0.1, beta=0.01):
        # We need filters for x and y of each landmark
        self.filters_x = [OneEuroFilter(freq, min_cutoff, beta) for _ in range(num_landmarks)]
        self.filters_y = [OneEuroFilter(freq, min_cutoff, beta) for _ in range(num_landmarks)]

    def smooth(self, landmarks):
        """
        Expects landmarks as a list of [x, y, z] or similar.
        Returns smoothed [x, y] coordinates.
        """
        smoothed = []
        for i, lm in enumerate(landmarks):
            sx = self.filters_x[i].filter(lm[0])
            sy = self.filters_y[i].filter(lm[1])
            smoothed.append([int(sx), int(sy)])
        return smoothed
