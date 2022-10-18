from typing import List, Optional, Tuple

import cv2
from numpy import ndarray

from .detection import FrameObjects


class VideoAnnotater:
    VideoObjects = List[FrameObjects]
    ImageSize = Tuple[int, int]

    FOURCC = cv2.VideoWriter_fourcc(*"MJPG")

    COLOUR = (255, 0, 0)  # red in RGB
    RECT_STROKE = 2  # px

    FONT = cv2.FONT_HERSHEY_PLAIN
    FONT_SIZE = 1.0
    FONT_STROKE = 0.5  # px

    def __init__(self, source: str) -> None:
        self.__capture = cv2.VideoCapture(source)
        self.__width, self.__height = self.frame_size

    @property
    def capture(self) -> cv2.VideoCapture:
        return self.__capture

    @property
    def frame_size(self) -> Tuple[int, int]:
        return (self.capture.get(cv2.CAP_PROP_FRAME_WIDTH),
                self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def frame_rate(self) -> int:
        return self.capture.get(cv2.CAP_PROP_FPS)

    @property
    def frame_count(self) -> int:
        return self.capture.get(cv2.CAP_PROP_FRAME_COUNT)

    @property
    def width(self) -> int:
        return self.__width

    @property
    def height(self) -> int:
        return self.__height

    def __del__(self) -> None:
        self.capture.release()

    def __seek_start_frame(self) -> None:
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def resize_frame(self, frame: ndarray, size: ImageSize) -> ndarray:
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)

    def resize_objects(self, objects: FrameObjects,
                       size: ImageSize) -> FrameObjects:

        source_width, source_height = self.frame_size
        target_width, target_height = size

        width_scale = source_width / target_width
        height_scale = source_height / target_height

        for obj in objects:
            obj.x = tuple(map(lambda x: x // width_scale, obj.x))
            obj.y = tuple(map(lambda y: y // height_scale, obj.y))

    def annotate_frame(self, frame: ndarray, objects: FrameObjects) -> ndarray:
        for obj in objects:
            lb, ub = zip(obj.x, obj.y)

            frame = cv2.rectangle(frame, lb, ub, self.COLOUR, self.RECT_STROKE)

            caption_coord = (lb[0], lb[1] - 5)
            caption = f'{obj.label} ({"no id" if obj.id is None else obj.id})'

            frame = cv2.putText(frame, caption, caption_coord, self.FONT,
                                self.FONT_SIZE, self.COLOUR, self.FONT_STROKE)

        return frame

    def save_annotatation(self, target: str, objects: VideoObjects,
                          size: Optional[ImageSize]) -> None:

        if self.frame_count != len(objects):
            raise ValueError('The number of annotations does not'
                             'match the frame count.')

        target_size = size if size is not None else self.frame_size
        writer = cv2.VideoWriter(
            target, self.FOURCC, self.frame_rate, target_size)

        i = 0
        self.__seek_start_frame()

        while self.capture.isOpened():
            continue_flag, frame = self.capture.read()

            if not continue_flag:
                break

            frame_objects = objects[i]

            if size is not None:
                frame = self.resize_frame(frame, size)
                frame_objects = self.resize_objects(frame_objects, size)

            frame = self.annotate_frame(frame, objects[i])

            writer.write(frame)
            i += 1

        writer.release()
