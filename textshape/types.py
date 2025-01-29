import numpy as np

type BoolVector = np.array[tuple[int], np.dtype[bool]]
type IntVector = np.array[tuple[int], np.dtype[int]]
type FloatVector = np.array[tuple[int], np.dtype[float]]
type Span = tuple[int, int]
