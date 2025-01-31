from typing import TypeVar
import numpy as np

T = TypeVar("T")
type Vector[T] = np.ndarray[tuple[int, ...], np.dtype[T]]  # type: ignore[type-var]
type FloatVector = Vector[np.float32]
type IntVector = Vector[np.int32]
type BoolVector = Vector[np.bool]

type Span = tuple[int, int]

type CharInfoVectors = tuple[
    str,  # text
    FloatVector,  # x
    FloatVector,  # y
    FloatVector,  # dx (width)
    FloatVector,  # dy (height)
]
