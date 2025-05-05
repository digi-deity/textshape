import re

from .types import FloatVector, Span

re_words = re.compile(r"\S+") #|\r?\n")  # matches whole words and newlines


class Fragments:
    """A fragment represents an unbreakable piece of visible text and its associated width. Each fragment has a
    whitespace width value which represents the spacing between that and the next fragment. The penalty width is a
    special spacing that is only used when the fragment appears at the end of line, for example to reserve space for a
    hyphen.

    A fragment is the minimum data structure needed to run the line breaking algorithm.
    """

    def __init__(
        self,
        widths: FloatVector,
        whitespace_widths: FloatVector,
        penalty_widths: FloatVector,
    ):
        self.widths = widths
        self.whitespace_widths = whitespace_widths
        self.penalty_widths = penalty_widths

    def unpack(self) -> tuple[FloatVector, FloatVector, FloatVector]:
        return self.widths, self.whitespace_widths, self.penalty_widths

    def __len__(self) -> int:
        return len(self.widths)


def word_fragmenter(s: str) -> list[Span]:
    return [m.span() for m in re_words.finditer(s)]
