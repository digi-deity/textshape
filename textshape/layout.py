from . import Fragments
from .types import CharInfoVectors

class MultiColumn:
    def __init__(self, height: int | float, width: int | float, text: Fragments):
        self.height = height
        self.width = width
        self.text = text

    def get_bboxes(self) -> list[CharInfoVectors]:
        column = self.text.to_bounding_boxes(self.width, self.height)