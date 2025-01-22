from typing import Callable, TYPE_CHECKING
from collections import deque

import numpy as np

from textshape.fragment import Fragments, word_fragmenter
from textshape.shape import monospace_measure, FontMeasure
from textshape.types import FloatVector, Span, IntVector
from textshape.wrap import wrap

if TYPE_CHECKING:
    import uharfbuzz._harfbuzz.Buffer

class Text:
    fragments: Fragments

    def __init__(
            self,
            text: str,
            measure: Callable[[str], FloatVector] = monospace_measure,
            fragmenter: Callable[[str], list[Span]] = word_fragmenter,
    ):
        n = len(text)
        self.text = text
        if not text:
            raise ValueError("Text cannot be empty")
        elif text[0].isspace() or text[n-1].isspace():
            raise ValueError("Input text cannot start or end with whitespace.")

        self.measure = measure

        self.widths = np.array(measure(text))
        spans = np.array(fragmenter(text)).T
        self.start = spans[0]
        self.end = spans[1]

        m = len(self.start)
        if self.start[0] != 0:
            raise ValueError("First span must start at the first character.")

        if self.end[m-1] != n:
            raise ValueError("Last span must end at the last character.")

        cumwidths = np.zeros(n+1)
        cumwidths[1:] = self.widths.cumsum()
        ravel = spans.ravel(order='F')
        fragment_widths = cumwidths[ravel[1:]] - cumwidths[ravel[:2*m-1]]
        whitespace = np.zeros(n, dtype=int)
        whitespace[self.end[:m-1]] += 1
        whitespace[self.start[1:]] -= 1
        self.whitespace = whitespace.cumsum()

        self.hyphen_width = measure('-')[0]

        self.fragments = Fragments(
            fragment_widths[::2],
            np.pad(fragment_widths[1::2], (0, 1)),
            np.pad(1-self.whitespace[self.end[:m-1]], (0, 1)) * self.hyphen_width,
        )

    def wrap(self, width: float, fontsize: float) -> tuple[IntVector, IntVector]:
        width = width / fontsize
        fragment_breaks = np.array(wrap(self.fragments, width))
        line_breaks = self.end[fragment_breaks[1:]-1]
        penalties = self.fragments.penalty_widths[fragment_breaks[1:]-1] > 0
        hyphens = line_breaks[penalties]
        return line_breaks, hyphens

    def text_lines(self, linewidth: int):
            fragments = self.fragments
            n = len(fragments)
            breakpoints, hyphens = self.wrap(linewidth)

            lines = []
            a = breakpoints[0]
            for i in range(1, len(breakpoints)):
                line = []
                b = breakpoints[i]
                for i in range(a, b):
                    line.append(self.get_fragment_str(i))
                    if not i + 1 == b:
                        line.append(' ' * round(fragments.whitespace_widths[i]))
                if fragments.penalty_widths[i] > 0.0 and i + 1 < n:
                    line.append('-')
                lines.append(''.join(line))
                a = b

            return lines

    def get_fragment_str(self, i: int) -> str:
        return self.text[self.start[i]:self.end[i]]