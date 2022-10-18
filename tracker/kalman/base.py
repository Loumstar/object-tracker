from typing import Callable, NoReturn, Optional

import functools
import numpy as np
from numpy.linalg import inv

import logging

MatrixCallable = Callable[[float], np.ndarray]


class KalmanFilter:

    def __init__(self, x: np.ndarray, p: np.ndarray, q: np.ndarray,
                 r: np.ndarray, h: np.ndarray,
                 u: Optional[np.ndarray] = None) -> None:

        self.x = x
        self.u = u

        self.p = p
        self.q = q
        self.r = r

        self.h = h

        self.__A = self.__unset_matrix
        self.__B = self.__unset_matrix

        self.__G = self.__identity_noise

        self.timer = 0

        self.logger = logging.Logger('KalmanFilter')

    def __unset_matrix(self, _) -> NoReturn:
        raise NotImplementedError

    def __identity_noise(self, _) -> np.ndarray:
        return np.identity(len(self.q))

    @property
    def A(self, t: float) -> np.ndarray:
        return self.__A(t)

    @A.setter
    def A(self, fnc: MatrixCallable) -> None:
        if not isinstance(fnc, MatrixCallable):
            raise ValueError('State transition matrix (A) is not of '
                             'type "MatrixCallable" '
                             '(Callable[[float], ndarray]).')

        a = fnc(0)

        if a.shape[0] != a.shape[1]:
            self.logger.warn('State transition method (A) returned a '
                             'non-square matrix. This will likely lead to '
                             'mismatching state matrix sizes, which will '
                             'throw an exception.')

        if a.shape[1] != self.x.shape[0]:
            self.logger.warn('State transition method (A) returned a matrix '
                             'that cannot be multiplied with the state vector '
                             '(x) due to different column/row sizes.')

        self.__A = fnc

    @property
    def B(self, t: float) -> np.ndarray:
        return self.__B(t)

    @B.setter
    def B(self, fnc: MatrixCallable) -> None:
        if not isinstance(fnc, MatrixCallable):
            raise ValueError('Control-input matrix (B) is not of '
                             'type "MatrixCallable".')

        b = fnc(0)

        if b.shape[0] != b.shape[1]:
            self.logger.warn('Control-input method (B) returned a '
                             'non-square matrix. This will likely lead to '
                             'mismatching state matrix sizes, which will '
                             'throw an exception.')

        if self.u is not None and b.shape[1] != self.u.shape[0]:
            self.logger.warn('Control-input method (B) returned a matrix that '
                             'cannot be multiplied with the control vector '
                             '(y) due to different column/row sizes.')

        self.__B = fnc

    @property
    def G(self, t: float) -> np.ndarray:
        return self.__G(t)

    @G.setter
    def G(self, fnc: MatrixCallable) -> None:
        if not isinstance(fnc, MatrixCallable):
            raise ValueError('Process noise matrix (G) is not of '
                             'type "MatrixCallable".')

        g = fnc(0)

        if g.shape[0] != g.shape[1]:
            self.logger.warn('Process noise method (G) returned a '
                             'non-square matrix. This will likely lead to '
                             'mismatching state matrix sizes, which will '
                             'throw an exception.')

        if g.shape[1] != self.q.shape[0]:
            self.logger.warn('Process noise method (G) returned a matrix that '
                             'cannot be multiplied with the noise vector '
                             '(q) due to different column/row sizes.')

        self.__G = fnc

    @functools.cache
    def Q(self, t: float) -> np.ndarray:
        return self.G(t).dot(self.q).dot(self.G(t).T)

    def __predict_state_estimate(self, t: float) -> np.ndarray:
        estimate = self.A(t).dot(self.x)

        if self.u is not None:
            estimate += self.B(t).dot(self.u)

        return estimate

    def __predict_state_covariance(self, t: float) -> np.ndarray:
        covariance = self.A(t).dot(self.p).dot(self.A(t).T) + self.Q(t)

        return covariance

    def predict(self, t: float) -> np.ndarray:
        self.x = self.__predict_state_estimate(t)
        self.p = self.__predict_state_covariance(t)

        self.timer += t

    def kalman_gain(self) -> np.ndarray:
        covariance = self.h.dot(self.p).dot(self.h.T)

        if self.r is not None:
            covariance += self.r

        return self.p.dot(self.h).dot(inv(covariance))

    def update(self, observation: np.ndarray) -> np.ndarray:
        y = self.h.dot(observation)
        k = self.kalman_gain()

        self.x += k.dot(y - self.h.dot(self.x))
        self.p = (np.identity(k) - k.dot(self.h)).dot(self.p)

        return self.x
