from typing import Callable, TYPE_CHECKING, TypeVar
from collections import deque
from itertools import pairwise

import numpy as np

from .fragment import Fragments, word_fragmenter
from .shape import monospace_measure, FontMeasure
from .types import FloatVector, Span, IntVector, BoolVector, CharInfoVectors, T, Vector
from .wrap import wrap

if TYPE_CHECKING:
    import uharfbuzz.Buffer

class Text:
    """A class for handling text shaping, wrapping, and justification."""

    fragments: Fragments
    text: str
    measure: Callable[[str], FloatVector]
    widths: FloatVector
    start: IntVector
    end: IntVector
    whitespace_mask: IntVector
    hyphen_width: float

    def __init__(
        self,
        text: str,
        measure: Callable[[str], FloatVector] = monospace_measure,
        fragmenter: Callable[[str], list[Span]] = word_fragmenter,
    ):
        """Initializes a Text object with the given text, measure function, and fragmenter function.

        The text should be a string and cannot start or end with whitespace.
        The measure function should return an array of character widths in em units.
        The fragmenter function should return a list of spans (start, end) that define the fragments of the text.
        """

        n = len(text)
        self.text = text
        if not text:
            raise ValueError("Text cannot be empty")
        elif text[0].isspace() or text[n - 1].isspace():
            raise ValueError("Input text cannot start or end with whitespace.")

        self.measure = measure

        self.widths = np.array(measure(text), dtype=np.float32)
        spans = np.array(fragmenter(text)).T
        self.start = spans[0]
        self.end = spans[1]

        m = len(self.start)
        if self.start[0] != 0:
            raise ValueError("First span must start at the first character.")

        if self.end[m - 1] != n:
            raise ValueError("Last span must end at the last character.")

        cwidths = np.zeros(n + 1, dtype=np.float32)
        cwidths[1:] = self.widths.cumsum()
        zipped = spans.ravel(order="F")
        fragment_widths = cwidths[zipped[1:]] - cwidths[zipped[: 2 * m - 1]]
        whitespace = np.zeros(n, dtype=int)
        whitespace[self.end[: m - 1]] += 1
        whitespace[self.start[1:]] -= 1
        self.whitespace_mask = whitespace.cumsum()

        self.hyphen_width = float(measure("-")[0])

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

        Returns a tuple of arrays containing the start and end indices of each line, and a boolean array indicating whether a hyphen is needed to break that line..
        """
        width = width / fontsize
        fragment_breaks: IntVector = np.array(wrap(self.fragments, width))
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
        """Justify the text such that each line has the same width.

        Returns an array of offsets to be added to the x position of each character.
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

    def hyphenate_text(self, breakpoints: IntVector) -> str:
        """Hyphenate the text at the given breakpoints."""

        return "-".join(
            (self.text[a:b] for a, b in pairwise((0, *breakpoints, len(self.text))))
        )

    def vectorize_target_widths(
        self, targets: int | float | list[float | int] | FloatVector | IntVector, paragraph_indent: float = 0.0
    ) -> FloatVector:
        """Ensure that the target linewidth(s) are in the correct array format and handles paragraph indentation."""
        if isinstance(targets, float | int):
            targets = [targets]

        targets_vector = np.array(targets, dtype=float)

        if paragraph_indent:
            if len(targets_vector) == 1:
                targets_vector = targets_vector.repeat(2)
            targets_vector[0] -= paragraph_indent

        return targets_vector

    def get_bboxes(
        self,
        target_width: int | float | list[float | int] | FloatVector | IntVector,
        fontsize: float,
        justify: bool = False,
        line_spacing: float = 1.0,
        paragraph_indent: float = 0.0,
    ) -> CharInfoVectors:
        """Calculate the bounding boxes of the text given a target width and fontsize.

        Returns a tuple containing the (hyphenated) text, and 4 arrays for the x and y coordinates and their respecting widths and heights.

        The target width can be a single float or a list of floats. If a list is provided, the text will be wrapped with different target widths.
        If the list is shorter than the number of lines, the last target width will be repeated.

        If justify is set to True, the text will be justified.

        line_spacing is the factor by which the line height is multiplied to increase interline spacing.

        paragraph_indent is the indentation of the first line of the paragraph.

        All input and output sizes are expressed in em units.

        Returns a tuple of the hyphenated text, x, dx, y, dy.
        """

        assert isinstance(
            self.measure, FontMeasure
        ), "Calculating bboxes requires a FontMeasure to precisely measure text."
        text = self.text

        targets_vector = self.vectorize_target_widths(target_width, paragraph_indent)
        line_starts, line_ends, hyphen_mask = self.wrap(targets_vector, fontsize)

        targets_vector = np.pad(
            targets_vector, (0, max(0, len(line_starts) - len(targets_vector))), mode="edge"
        )[: len(line_starts)]
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
        y = np.zeros_like(widths)
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
                targets_vector / fontsize, x, ws, line_starts, line_ends
            )
            x = np.pad(dx, (1, 0)).cumsum()

        if paragraph_indent > 0.0:
            x[: line_starts[1]] += paragraph_indent / fontsize

        resets = np.zeros_like(x)
        resets[line_starts[1:]] = np.diff(x[line_starts[1:]], prepend=0)
        x -= resets.cumsum()

        # Clip mask for zeroing out whitespace at end of lines
        _clip_mask = np.zeros(len(text), dtype=np.int32)
        _clip_mask[line_ends[:-1]] += 1
        _clip_mask[line_starts[1:]] -= 1
        clip_mask = _clip_mask.cumsum().astype(np.bool)
        dx[clip_mask] = 0.0
        dy[clip_mask] = 0.0

        return text, x[:-1] * fontsize, dx * fontsize, y * fontsize, dy * fontsize  # type: ignore[return-value]

    def get_lines(
            self,
            target_width: int | float | list[float | int] | FloatVector | IntVector,
            fontsize: int | float
    ) -> list[str]:
        """Breaks the text into lines given a target width and fontsize.

        Note that this method includes hyphenation if words are broken.

        Returns a list of strings, each representing a line of text no longer than the target width.
        """
        targets_vector = self.vectorize_target_widths(target_width, 0)
        line_starts, line_ends, hyphen_mask = self.wrap(targets_vector, fontsize)
        text = self.hyphenate_text(line_starts[hyphen_mask])
        line_starts, line_ends = self.hyph_adjust_linespans(
            line_starts, line_ends, hyphen_mask
        )
        return [text[line_starts[i] : line_ends[i]] for i in range(len(line_starts))]

    def hyph_adjust_chararrays(self, arr: FloatVector, breakpoints: IntVector, value: float) -> FloatVector:
        """Adjust arrays for hyphenation by inserting a value at the hyphenation points

        Returns a new array with the value inserted at the breakpoints.
        """
        inserts = np.full(len(breakpoints), value)
        return np.insert(arr, breakpoints, inserts)

    def hyph_adjust_linespans(self, line_starts: IntVector, line_ends: IntVector, hyphen_mask: BoolVector) -> tuple[IntVector, IntVector]:
        """Adjust line spans for hyphenation by changing the start and end points of lines to match hyphenated text.

        Returns a tuple of adjusted line starts and line ends.
        """

        adjustment: IntVector = np.zeros(len(line_starts) + 1, dtype=line_starts.dtype)
        adjustment[:-1][hyphen_mask] += 1
        adjustment = adjustment.cumsum()
        line_starts = line_starts + adjustment[:-1]
        line_ends = line_ends + adjustment[1:]
        return line_starts, line_ends

    def get_fragment_str(self, i: int) -> str:
        """Helper function to get the text representation of the i-th fragment."""
        return self.text[self.start[i] : self.end[i]]
