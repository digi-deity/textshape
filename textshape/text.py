from typing import Callable

import numpy as np

from .fragment import Fragments, word_fragmenter
from .shape import monospace_measure, FontMeasure
from .types import FloatVector, Span, IntVector, BoolVector, CharInfoVectors
from .wrap import wrap

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
        hyphen_mask = self.fragments.penalty_widths[fragment_breaks[1:] - 1] > 0
        return line_starts, line_ends, hyphen_mask

    def justify(
        self,
        target_width: FloatVector,
        dx: FloatVector,
        x: FloatVector,
        dx_ws: FloatVector,
        ws: FloatVector,
        linebreaks: IntVector,
    ) -> FloatVector:
        """Justify the text such that each line has the same width.

        Returns an array of offsets to be added to the x position of each character.
        """
        linewidths = np.diff(x[linebreaks], prepend=0)
        whitewidths = np.diff(ws[linebreaks], prepend=0)
        remainders = target_width - linewidths
        factors = remainders / whitewidths
        factors[np.isinf(factors)] = 0.0
        factors[-1] = 0.0  # last line in paragraph is exempt from justification

        offsets = np.zeros_like(dx_ws)
        offsets[0] = factors[0]
        offsets[linebreaks[:-1]] = np.diff(factors)
        offsets = offsets.cumsum() * dx_ws

        dx = dx + offsets
        x = np.pad(dx, (1, 0)).cumsum()
        return dx, x

    def vectorize_target_widths(
        self,
        targets: int | float | list[float | int] | FloatVector | IntVector,
        paragraph_indent: float = 0.0,
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
        fm = self.measure
        widths = self.widths
        text = np.frombuffer(self.text.encode('utf-32-le'), dtype=np.uint32)
        if len(text) != len(widths): raise ValueError("Text and widths must have the same length. Something went wrong")

        widths = np.hstack([[0.0, self.hyphen_width], widths])
        text = np.hstack([np.frombuffer('\n-'.encode('utf-32-le'), dtype=np.uint32), text])

        targets_vector = self.vectorize_target_widths(target_width, paragraph_indent)
        line_starts, line_ends, hyphen_mask = self.wrap(targets_vector, fontsize)

        pad = (0, max(0, len(line_starts) - len(targets_vector)))
        targets_vector = np.pad(targets_vector,pad, mode="edge",)[: len(line_starts)]

        # Create a mapping that will be used to construct the newly wrapped text from the original
        mapping = np.arange(len(text) - 2) + 2
        mask = np.zeros(len(text) - 2, dtype=np.int32)
        mask[line_ends[:-1]] += 1
        mask[line_starts[1:]] -= 1
        mask = (1 - mask.cumsum()).astype(np.bool)
        mapping = mapping[mask]

        # Insert newline characters
        linebreaks = line_ends - np.cumsum(np.pad(line_starts[1:] - line_ends[:-1], (1, 0)))
        mapping = np.insert(mapping, linebreaks[:-1], 0)
        linebreaks[:-1] += np.arange(len(linebreaks) -1 ) + 1

        # Insert hyphens
        hyphbreaks = linebreaks[hyphen_mask] - 1
        mapping = np.insert(mapping, hyphbreaks, 1)
        linebreaks = (linebreaks + np.cumsum(hyphen_mask))
        linebreaks_ = linebreaks[:-1]

        text = text[mapping].tobytes().decode('utf-32-le')
        widths = widths[mapping]

        # Determine height coordinates
        extents = fm.vhb.hbfont.get_font_extents("ltr")
        line_gap = (extents.ascender - extents.descender) / fm.em
        y = np.zeros_like(widths)
        y[linebreaks_] = -line_gap * line_spacing
        y = y.cumsum() + extents.descender / fm.em
        dy = np.full_like(y, line_gap)

        # Determine width coordinates
        dx = widths
        x = np.pad(dx, (1, 0)).cumsum()

        if justify:
            dx_ws = widths * self.whitespace_mask[mapping-2]
            ws = np.pad(dx_ws, (1, 0)).cumsum()
            dx, x = self.justify(
                targets_vector / fontsize, dx, x, dx_ws, ws, linebreaks
            )

        resets = np.zeros_like(x)
        resets[linebreaks_] = np.diff(x[linebreaks_], prepend=0)
        x -= resets.cumsum()

        if paragraph_indent > 0.0:
            x[: linebreaks[0]] += paragraph_indent / fontsize

        return text, x[:-1] * fontsize, dx * fontsize, y * fontsize, dy * fontsize  # type: ignore[return-value]

    def get_lines(
        self,
        target_width: int | float | list[float | int] | FloatVector | IntVector,
        fontsize: int | float,
    ) -> list[str]:
        """Breaks the text into lines given a target width and fontsize.

        Note that this method includes hyphenation if words are broken.

        Returns a list of strings, each representing a line of text no longer than the target width.
        """
        targets_vector = self.vectorize_target_widths(target_width, 0)
        line_starts, line_ends, hyphen_mask = self.wrap(targets_vector, fontsize)
        return [self.text[a:b] + ('-' if h else '') for a, b, h in zip(line_starts, line_ends, hyphen_mask)]


    def get_fragment_str(self, i: int) -> str:
        """Helper function to get the text representation of the i-th fragment."""
        return self.text[self.start[i] : self.end[i]]
