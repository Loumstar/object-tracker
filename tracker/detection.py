from typing import List, NamedTuple, Optional, Tuple


class DetectedObject(NamedTuple):
    id:  Optional[int]
    label: str
    x: Tuple[int, int]
    y: Tuple[int, int]


FrameObjects = List[DetectedObject]
