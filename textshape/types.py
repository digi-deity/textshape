from typing import TypeVar,Union
import numpy as np

Numeric = Union[int, float]
T = TypeVar("T")
Vector = np.ndarray[tuple[int, ...], np.dtype[T]]
FloatVector = Vector[np.float32]
IntVector = Vector[np.int32]
BoolVector = Vector[np.bool]

Span = tuple[int, int]

CharInfoVectors = tuple[
    str,  # text
    FloatVector,  # x
    FloatVector,  # y
    FloatVector,  # dx (width)
    FloatVector,  # dy (height)
]
