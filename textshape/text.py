from typing import Callable, TYPE_CHECKING
from collections import deque
from itertools import pairwise

import numpy as np

from textshape.fragment import Fragments, word_fragmenter
from textshape.shape import monospace_measure, FontMeasure
from textshape.types import FloatVector, Span, IntVector, BoolVector
from textshape.wrap import wrap

if TYPE_CHECKING:
    import uharfbuzz.Buffer

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
        zipped = spans.ravel(order='F')
        fragment_widths = cumwidths[zipped[1:]] - cumwidths[zipped[:2*m-1]]
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

    def wrap(self, width: float, fontsize: float) -> tuple[IntVector, BoolVector]:

        """Wraps the text given a fontsize and a maximum line width.

        Returns a tuple of vector. The first array are the indices for the breakpoints in text string and
        the second array a boolean mask for the breakpoint vector that indicates whether a hyphen must be used to break.
        """
        width = width / fontsize
        fragment_breaks = np.array(wrap(self.fragments, width))[1:-1]
        breakpoints = self.start[fragment_breaks]
        hyphens = self.fragments.penalty_widths[fragment_breaks-1] > 0
        return breakpoints, hyphens

    def hyphenate(self, breakpoints) -> str:
        return '-'.join((self.text[a:b] for a, b in pairwise((0, *breakpoints, len(self.text)))))

    def get_bboxes(self, width: float, fontsize: float):
        assert isinstance(self.measure, FontMeasure), "Calculating bboxes requires a FontMeasure to precisely measure text."
        text = self.text
        breakpoints, hyphens = self.wrap(width, fontsize)

        fm = self.measure
        widths = self.widths
        hyphpoints = breakpoints[hyphens]
        if len(hyphpoints) > 0:
            text = self.hyphenate(hyphpoints)
            hyphwidths = np.full(len(hyphpoints), self.hyphen_width)
            widths = np.insert(widths, hyphpoints, hyphwidths)

        breakpoints = self.adjust_breakpoints(breakpoints, hyphens)

        # Determine height coordinates
        extents = fm.vhb.hbfont.get_font_extents("ltr")
        line_gap = (extents.ascender - extents.descender) / fm.em
        y = np.zeros(len(widths) + 1, dtype=widths.dtype)
        y[breakpoints] = -line_gap
        y = y.cumsum()
        dy = np.full_like(y, line_gap)

        # Determine width coordinates
        dx = np.pad(widths.cumsum(), (1, 0))
        resets = np.zeros_like(dx)
        resets[breakpoints] = np.diff(dx[breakpoints], prepend=0)
        x = dx - resets.cumsum()

        return text, x, dx, y, dy

    def get_lines(self, width: float, fontsize: float) -> np.ndarray:
        """Breaks the text into lines."""

        breakpoints, hyphens = self.wrap(width, fontsize)
        text = self.hyphenate(breakpoints[hyphens])
        breakpoints = self.adjust_breakpoints(breakpoints, hyphens)
        return [text[a:b].rstrip() for a, b in pairwise((0, *breakpoints, len(self.text)))]

    def adjust_breakpoints(self, breakpoints, hyphens):
        """Adjust breakpoints for hyphenation"""

        adjustment = np.zeros_like(breakpoints)
        adjustment[hyphens] += 1
        adjustment = adjustment.cumsum()
        adj_breakpoints = breakpoints + adjustment
        return adj_breakpoints

    def get_fragment_str(self, i: int) -> str:
        return self.text[self.start[i]:self.end[i]]