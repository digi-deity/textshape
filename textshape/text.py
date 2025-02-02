from typing import Callable, TYPE_CHECKING, Optional
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
        elif text[0].isspace() or text[n - 1].isspace():
            raise ValueError("Input text cannot start or end with whitespace.")

        self.measure = measure

        self.widths = np.array(measure(text))
        spans = np.array(fragmenter(text)).T
        self.start = spans[0]
        self.end = spans[1]

        m = len(self.start)
        if self.start[0] != 0:
            raise ValueError("First span must start at the first character.")

        if self.end[m - 1] != n:
            raise ValueError("Last span must end at the last character.")

        cumwidths = np.zeros(n + 1)
        cumwidths[1:] = self.widths.cumsum()
        zipped = spans.ravel(order="F")
        fragment_widths = cumwidths[zipped[1:]] - cumwidths[zipped[: 2 * m - 1]]
        whitespace = np.zeros(n, dtype=int)
        whitespace[self.end[: m - 1]] += 1
        whitespace[self.start[1:]] -= 1
        self.whitespace_mask = whitespace.cumsum()

        self.hyphen_width = measure("-")[0]

        self.fragments = Fragments(
            fragment_widths[::2],
            np.pad(fragment_widths[1::2], (0, 1)),
            np.pad(1 - self.whitespace_mask[self.end[: m - 1]], (0, 1))
            * self.hyphen_width,
        )

    def wrap(
        self, width: FloatVector, fontsize: float
    ) -> tuple[IntVector, IntVector, BoolVector]:
        """Wraps the text given a fontsize and a maximum line width.

        Returns three vector. The first an array indices for the breakpoints in text string, the second an array of indices
         for breakpoints with whitespace trimmed and the third a boolean vector that indicates whether a hyphen must be used to break.
        """
        width = width / fontsize
        fragment_breaks = np.array(wrap(self.fragments, width))
        line_starts = self.start[fragment_breaks[:-1]]
        line_ends = self.end[fragment_breaks[1:] - 1]
        hyphen_mask = self.fragments.penalty_widths[fragment_breaks[:-1] - 1] > 0
        return line_starts, line_ends, hyphen_mask

    def justify(
        self,
        target_width: FloatVector,
        x: FloatVector,
        dx_ws: FloatVector,
        line_starts: IntVector,
        line_ends: IntVector,
    ) -> FloatVector:
        """Justify the text such that each
        """
        x_ws = np.pad(dx_ws, (1, 0)).cumsum()
        linewidths = x[line_ends] - x[line_starts]
        whitewidths = x_ws[line_ends] - x_ws[line_starts]
        remainders = target_width - linewidths
        factors = remainders / whitewidths
        factors[np.isinf(factors)] = 0.0
        factors[-1] = 0.0

        offsets = np.zeros_like(dx_ws)
        offsets[line_starts] = factors
        offsets[line_starts[1:]] -= factors[:-1]
        offsets = offsets.cumsum() * dx_ws
        return offsets

    def hyphenate_text(self, breakpoints) -> str:
        return "-".join(
            (self.text[a:b] for a, b in pairwise((0, *breakpoints, len(self.text))))
        )

    def listify_target_linewidths(
        self, targets: float | list[float], paragraph_indent: float = 0.0
    ) -> FloatVector:
        if isinstance(targets, float | int):
            targets = [targets]

        targets = np.array(targets, dtype=float)

        if paragraph_indent:
            if len(targets) == 1:
                targets = targets.repeat(2)
            targets[0] -= paragraph_indent

        return np.array(targets, dtype=float)

    def get_bboxes(
        self,
        target_width: float | list[float],
        fontsize: float,
        justify: bool = False,
        line_spacing: float = 1.0,
        paragraph_indent: float = 0.0,
    ) -> tuple[str, FloatVector, FloatVector, FloatVector, FloatVector]:
        """Calculate the bounding boxes of the text given a target width and fontsize.

        Returns a tuple containing the (hyphenated) text, and 4 arrays for the x and y coordinates and their respecting widths and heights.

        The target width can be a single float or a list of floats. If a list is provided, the text will be wrapped with different target widths.
        If the list is shorter than the number of lines, the last target width will be repeated.

        If justify is set to True, the text will be justified.

        line_spacing is the factor by which the line height is multiplied to increase interline spacing.

        paragraph_indent is the indentation of the first line of the paragraph.

        All input and output sizes are expressed in em units.
        """

        assert isinstance(
            self.measure, FontMeasure
        ), "Calculating bboxes requires a FontMeasure to precisely measure text."
        text = self.text

        target_width = self.listify_target_linewidths(target_width, paragraph_indent)
        line_starts, line_ends, hyphen_mask = self.wrap(target_width, fontsize)

        target_width = np.pad(target_width, (0, max(0, len(line_starts) - len(target_width))),mode='edge')[:len(line_starts)]
        fm = self.measure
        widths = self.widths.copy()

        hyphpoints = line_starts[hyphen_mask]

        if len(hyphpoints) > 0:
            text = self.hyphenate_text(hyphpoints)
            widths = self.hyph_adjust_chararrays(widths, hyphpoints, self.hyphen_width)

        line_starts, line_ends = self.hyph_adjust_linespans(
            line_starts, line_ends, hyphen_mask
        )

        # Determine height coordinates
        extents = fm.vhb.hbfont.get_font_extents("ltr")
        line_gap = (extents.ascender - extents.descender) / fm.em
        y = np.zeros(len(widths), dtype=widths.dtype)
        y[line_starts[1:]] = -line_gap * line_spacing
        y = y.cumsum() + extents.descender / fm.em
        dy = np.full_like(y, line_gap)

        # Determine width coordinates
        dx = widths
        x = np.pad(dx, (1, 0)).cumsum()

        if justify:
            ws = self.widths * self.whitespace_mask
            if len(hyphpoints) > 0:
                ws = self.hyph_adjust_chararrays(ws, hyphpoints, 0)
            dx = widths + self.justify(
                target_width / fontsize, x, ws, line_starts, line_ends
            )
            x = np.pad(dx, (1, 0)).cumsum()

        if paragraph_indent > 0.0:
            x[:line_starts[1]] += paragraph_indent / fontsize

        resets = np.zeros_like(x)
        resets[line_starts[1:]] = np.diff(x[line_starts[1:]], prepend=0)
        x -= resets.cumsum()

        # Clip mask for zeroing out whitespace at end of lines
        clip_mask = np.zeros(len(text), dtype=int)
        clip_mask[line_ends[:-1]] += 1
        clip_mask[line_starts[1:]] -= 1
        clip_mask = clip_mask.cumsum().astype(bool)
        dx[clip_mask] = 0.0
        dy[clip_mask] = 0.0

        return text, x[:-1] * fontsize, dx * fontsize, y * fontsize, dy * fontsize

    def get_lines(self, width: float | list[float], fontsize: float) -> np.ndarray:
        """Breaks the text into lines."""
        line_starts, line_ends, hyphen_mask = self.wrap(width, fontsize)
        text = self.hyphenate_text(line_starts[hyphen_mask])
        line_starts, line_ends = self.hyph_adjust_linespans(
            line_starts, line_ends, hyphen_mask
        )
        return [text[line_starts[i] : line_ends[i]] for i in range(len(line_starts))]

    def hyph_adjust_chararrays(self, arr, breakpoints, value):
        inserts = np.full(len(breakpoints), value)
        return np.insert(arr, breakpoints, inserts)

    def hyph_adjust_linespans(self, line_starts, line_ends, hyphen_mask):
        """Adjust breakpoints for hyphenation"""

        adjustment = np.zeros(len(line_starts) + 1, dtype=line_starts.dtype)
        adjustment[:-1][hyphen_mask] += 1
        adjustment = adjustment.cumsum()
        line_starts = line_starts + adjustment[:-1]
        line_ends = line_ends + adjustment[1:]
        return line_starts, line_ends

    def get_fragment_str(self, i: int) -> str:
        return self.text[self.start[i] : self.end[i]]
