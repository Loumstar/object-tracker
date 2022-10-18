from typing import Tuple

import functools
import numpy as np

from kalman.base import KalmanFilter


class MotionKalmanFilter(KalmanFilter):

    def __init__(self, state: Tuple[float, ...],
                 state_uncertainty: Tuple[float, ...],
                 acceleration_uncertainty: Tuple[float, ...],
                 observation_uncertainty: Tuple[float, ...]) -> None:

        state_size = len(state)

        x = np.zeros(state_size * 2)  # [x, y, ..., x', y', ...]
        x[:state_size] = state

        h = np.zeros((state_size, state_size * 2))
        h[:, state_size:] = x

        p = self.independent_covariance(*state_uncertainty)
        q = self.independent_covariance(*acceleration_uncertainty)
        r = self.independent_covariance(*observation_uncertainty)

        super().__init__(x, p, q, r, h)

        self.A = self.motion_state_transition

    def independent_covariance(self, *vars: float) -> np.ndarray:
        return np.array([[(v1*v2) for v2 in vars] for v1 in vars])

    @functools.cache
    @staticmethod
    def motion_state_transition(t: float) -> np.ndarray:
        return np.array([
            [1, 0, t, 0],
            [0, 1, 0, t],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

    def acceleration_process_noise(t: float) -> np.ndarray:
        return np.array([

        ])
