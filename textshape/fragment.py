import re

import numpy as np

from textshape.types import FloatVector, Span

re_words = re.compile(r"\S+")


class Fragments:
    """The minimum set of data needed to run the line breaking algorithm."""

    def __init__(
        self,
        widths: np.array,
        whitespace_widths: np.array,
        penalty_widths: np.array,
    ):
        self.widths = widths
        self.whitespace_widths = whitespace_widths
        self.penalty_widths = penalty_widths

    def unpack(self) -> tuple[FloatVector, FloatVector, FloatVector]:
        return self.widths, self.whitespace_widths, self.penalty_widths

    def __iter__(self) -> iter:
        return iter(self.unpack())

    def __len__(self) -> int:
        return len(self.widths)


def word_fragmenter(s: str) -> list[Span]:
    return [m.span() for m in re_words.finditer(s)]
