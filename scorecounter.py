import numpy as np
from collections import deque

class ScoreCounter:

    def __init__(self, window_size=100):
        self.episode = 1
        self.score = 0
        self.scores = []
        self.score_window = deque(maxlen=window_size)
        self.score_window_means = []

    def add(self, value):
        self.score += value

    def next(self):
        self.episode += 1
        self.scores.append(self.score)
        self.score_window.append(self.score)
        self.score_window_means.append(self.window_mean)
        self.score = 0

    @property
    def window_mean(self):
        if len(self.score_window) == 0:
            return 0
        return np.mean(self.score_window)

    @property
    def max_window_mean(self):
        return np.max(self.score_window_means)

    def first_reached(self, value):
        return np.argmax(np.array(self.score_window_means) >= value)