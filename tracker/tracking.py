from .motion import MotionKalmanFilter
from .detection import FrameObjects


class MotionTracker:

    def __init__(self) -> None:
        self.tracked_objects = {}

    def update(self, objects: FrameObjects):
        for obj in objects:
            pass
