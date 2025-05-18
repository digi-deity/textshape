import re
from functools import cached_property
from typing import Callable

import numpy as np

from .shape import FontMeasure, monospace_measure

from .types import FloatVector, Span, IntVector, BoolVector, CharInfoVectors
from .wrap import wrap, TextFragmentsBase

re_words = re.compile(r"\S+") #|\r?\n")  # matches whole words and newlines


def word_splitter(s: str) -> list[Span]:
    return [m.span() for m in re_words.finditer(s)]

class TextFragmenter:
    _re_nt = re.compile(r"[\n\t]")

    def __init__(
            self,
            measure: Callable[[str], FloatVector] = None,
            splitter: Callable[[str], list[Span]] = None,
            tab_width: float | int = 4
    ):
        if measure is None:
            measure = monospace_measure

        if splitter is None:
            splitter = word_splitter

        self.measure = measure
        self.splitter = splitter
        self.tab_width = tab_width
        self.hyphen_width = float(self.measure("-")[0])

    def __call__(self, text: str) -> "TextFragments":
        n = len(text)

        if not text:
            raise ValueError("Text cannot be empty")
        elif (text[0].isspace() and text[0] != '\t') or text[n - 1].isspace():
            raise ValueError("Input text cannot start or end with whitespace.")

        widths = np.array(self.measure(text), dtype=np.float32)
        spans = np.array(self.splitter(text)).T

        # Create extra fragments for newline characters or tabs
        nt = np.array([(m.start(), text[m.start()] == '\t') for m in self._re_nt.finditer(text)]).T
        if len(nt):
            nt_pos, nt_tab = nt[0], nt[1].astype(bool)
            nt_fragment_idx = np.searchsorted(spans[0], nt_pos)
            widths[nt_pos[nt_tab]] = self.tab_width
            widths[nt_pos[~nt_tab]] = 0
            spans = np.insert(spans, nt_fragment_idx, np.stack([nt_pos, nt_pos+1]), axis=1)
            nt_fragment_idx = nt_fragment_idx + np.arange(len(nt_pos))

        start = spans[0]
        end = spans[1]

        m = len(start)
        if start[0] != 0:
            raise ValueError("First span must start at the first character.")

        if end[m - 1] != n:
            raise ValueError("Last span must end at the last character.")

        cwidths = np.zeros(n + 1, dtype=np.float32)
        cwidths[1:] = widths.cumsum()
        zipped = spans.ravel(order="F")
        pre_fragment_widths = cwidths[zipped[1:]] - cwidths[zipped[: 2 * m - 1]]
        whitespace = np.zeros(n, dtype=int)
        whitespace[end[: m - 1]] += 1
        whitespace[start[1:]] -= 1
        whitespace_mask = whitespace.cumsum()

        fragment_widths = pre_fragment_widths[::2]
        whitespace_widths = np.pad(pre_fragment_widths[1::2], (0, 1))
        penalty_widths = np.pad(self.hyphen_width * (1 - whitespace_mask[end[: m - 1]]), (0, 1), constant_values=-1)

        # Create conditions for forced linebreaks and tabs
        if len(nt):
            whitespace_widths[nt_fragment_idx[nt_tab]] = 0
            whitespace_widths[nt_fragment_idx[~nt_tab] - 1] = 100000
            penalty_widths[nt_fragment_idx[nt_tab]] = 0
            penalty_widths[nt_fragment_idx[~nt_tab] - 1] = -1

        return TextFragments(
            text=text,
            measure=self.measure,
            hyphen_width=self.hyphen_width,
            tab_width=self.tab_width,
            ch_widths=widths,
            ch_ws_mask=whitespace_mask,
            starts=start,
            ends=end,
            widths=fragment_widths,
            whitespace_widths=whitespace_widths,
            penalty_widths=penalty_widths,
        )


class TextFragments(TextFragmentsBase):
    """
    A fragment represents an unbreakable chunk of characters. Each fragment has a width and a whitespace width value. The
    latter represents the spacing between that and the next fragment. The penalty width is a special spacing that is
    only used when the fragment appears at the end of line, for example to reserve space for a hyphen.

    Wraps an input text to fit into a column of a given width.

    The column is of unbounded length. The width of the column does not have to homogeneous across the length of the
    column. All inputs that represent a height or width are assumed to be expressed in em units.
    """

    text: str

    ch_widths: FloatVector  # Width of each character in the text
    ch_ws_mask: IntVector  # Mask to indicate which characters are whitespace

    starts: IntVector  # Start indices of each fragment in the text
    ends: IntVector  # End indices of each fragment in the text

    hyphen_width: float  # Width of the hyphen character

    def __init__(
        self,
        text: str,
        measure: FontMeasure,
        hyphen_width: float,
        tab_width: float,
        ch_widths: FloatVector,
        ch_ws_mask: IntVector,
        starts: IntVector,
        ends: IntVector,
        widths: FloatVector,
        whitespace_widths: FloatVector,
        penalty_widths: FloatVector,
    ):
        self.text = text
        self.measure = measure
        self.tab_width = tab_width
        self.hyphen_width = hyphen_width
        self.ch_widths = ch_widths
        self.ch_ws_mask = ch_ws_mask
        self.starts = starts
        self.ends = ends

        super().__init__(widths, whitespace_widths, penalty_widths)

    def get_fragment_str(self, i: int) -> str:
        """Helper function to get the text representation of the i-th fragment."""
        return self.text[self.starts[i]: self.ends[i]]


class TextColumn:
    def __init__(
        self,
        fragments: TextFragments,
        column_width: int | float | list[float | int] | FloatVector | IntVector,
        fontsize: int | float,
    ):
        self.fragments = fragments
        self.column_width: FloatVector = self.vectorize_target_widths(column_width)
        self.fontsize = fontsize

    @cached_property
    def wrap(self) -> tuple[IntVector, IntVector, BoolVector, BoolVector]:
        """Wraps the text given a fontsize and a maximum line width.

        Returns a tuple of arrays containing the start and end indices of each line, and a boolean array indicating
        whether a hyphen is needed to break that line.
        """
        width = self.column_width / self.fontsize
        fragment_breaks: IntVector = np.array(wrap(self.fragments, width))
        hyphen_mask = self.fragments.penalty_widths[fragment_breaks[1:] - 1] > 0
        forced_mask = self.fragments.penalty_widths[fragment_breaks[1:] - 1] < 0
        line_starts = self.fragments.starts[fragment_breaks[:-1]]
        line_ends = self.fragments.ends[fragment_breaks[1:] - 1]
        line_starts[np.pad(forced_mask[:-1], (1, 0))] += 1  # Exclude the newline characters used for forced linebreaks
        return line_starts, line_ends, hyphen_mask, forced_mask

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
        _, _, _, forced_mask = self.wrap

        linewidths = np.diff(x[linebreaks], prepend=0)
        whitewidths = np.diff(ws[linebreaks], prepend=0)
        remainders = target_width - linewidths
        factors = remainders / whitewidths
        factors[np.isinf(factors)] = 0.0
        factors[forced_mask] = 0.0  # No justification in last line of paragraph with a forced line break

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
    ) -> FloatVector:
        """Ensure that the target linewidth(s) are in the correct array format."""
        if isinstance(targets, float | int):
            targets = [targets]

        targets_vector = np.array(targets, dtype=float)

        return targets_vector

    def to_bounding_boxes(
        self,
        justify: bool = False,
        line_spacing: float = 1.0,
    ) -> CharInfoVectors:
        """Calculate the bounding boxes of the text given a target width and fontsize.

        Returns a tuple containing the (hyphenated) text, and 4 arrays for the x and y coordinates and their respecting widths and heights.

        The target width can be a single float or a list of floats. If a list is provided, the text will be wrapped with different target widths.
        If the list is shorter than the number of lines, the last target width will be repeated.

        If justify is set to True, the text will be justified.

        line_spacing is the factor by which the line height is multiplied to increase interline spacing.

        All input and output sizes are expressed in em units.

        Returns a tuple of the hyphenated text, x, dx, y, dy.
        """
        assert isinstance(
            self.fragments.measure, FontMeasure
        ), "Calculating bboxes requires a FontMeasure to precisely measure text."

        # Modify text by inserting newline and hyphen characters at linebreak positions
        text, linebreaks, modified, targets_vector, charwidths = self.modify_text()

        # Determine Y-coordinates
        dy, y = self.calc_y(line_spacing, linebreaks, charwidths)

        # Determine X-coordinates
        dx, x = self.calc_x(justify, linebreaks, modified, targets_vector, charwidths)

        fontsize = self.fontsize
        return text, x[:-1] * fontsize, dx * fontsize, y * fontsize, dy * fontsize  # type: ignore[return-value]

    def modify_text(self):
        text = np.frombuffer(self.fragments.text.encode('utf-32-le'), dtype=np.uint32)
        widths = self.fragments.ch_widths
        if len(text) != len(widths): raise ValueError("Text and widths must have the same length. Something went wrong")

        line_starts, line_ends, hyphen_mask, _ = self.wrap

        # Add newline and hyphen character to the beginning of the text.
        # This enables quick reconstruction of text with newlines and hyphens inserted using vectorized operations.

        text = np.hstack([np.frombuffer('\n-'.encode('utf-32-le'), dtype=np.uint32), text])
        widths = np.hstack([[0.0, self.fragments.hyphen_width], widths])

        # Make sure the target line widths vector is the same length as the number of lines to prevent index errors
        pad = (0, max(0, len(line_starts) - len(self.column_width)))
        targets_vector = np.pad(self.column_width, pad, mode="edge", )[: len(line_starts)]

        # Create a character mapping representing the modified output text string. This vector maps the modified string
        # index positions to the input text string position. We initialize a mapping that matches the original text.
        modified = np.arange(len(text) - 2) + 2

        # Now remove whitespace characters at the end of each line
        mask = np.zeros(len(modified), dtype=np.int32)
        mask[line_ends[:-1]] += 1
        mask[line_starts[1:]] -= 1
        mask = (1 - mask.cumsum()).astype(np.bool)
        modified = modified[mask]

        # Insert newline characters
        linebreaks = line_ends - np.cumsum(np.pad(line_starts[1:] - line_ends[:-1], (1, 0)))
        modified = np.insert(modified, linebreaks[:-1], 0)
        linebreaks[:-1] += np.arange(len(linebreaks) - 1) + 1

        # Insert hyphens
        hyphbreaks = linebreaks[hyphen_mask] - 1
        modified = np.insert(modified, hyphbreaks, 1)
        linebreaks = (linebreaks + np.cumsum(hyphen_mask))

        # Reconstruct the modified text from the original
        text = text[modified].tobytes().decode('utf-32-le')
        widths = widths[modified]

        return text, linebreaks, modified, targets_vector, widths

    def calc_x(self, justify, linebreaks, modified, targets_vector, dx):
        linebreaks_ = linebreaks[:-1]
        x = np.pad(dx, (1, 0)).cumsum()

        # Optional text justification
        if justify:
            dx_ws = dx * self.fragments.ch_ws_mask[modified - 2]
            ws = np.pad(dx_ws, (1, 0)).cumsum()
            dx, x = self.justify(
                targets_vector / self.fontsize, dx, x, dx_ws, ws, linebreaks
            )

        resets = np.zeros_like(x)
        resets[linebreaks_] = np.diff(x[linebreaks_], prepend=0)
        x -= resets.cumsum()

        return dx, x

    def calc_y(self, line_spacing, linebreaks, widths):
        linebreaks = linebreaks[:-1]


        fm = self.fragments.measure
        extents = fm.vhb.hbfont.get_font_extents("ltr")
        line_gap = (extents.ascender - extents.descender) / fm.em
        y = np.zeros_like(widths)
        y[linebreaks] = -line_gap * line_spacing
        y = y.cumsum() + extents.descender / fm.em
        dy = np.full_like(y, line_gap)
        return dy, y

    def to_list(
        self,
    ) -> list[str]:
        """Breaks the text into lines given a target width and fontsize.

        Note that this method includes hyphenation if words are broken.

        Returns a list of strings, each representing a line of text no longer than the target width.
        """
        line_starts, line_ends, hyphen_mask, _ = self.wrap
        return [
            self.fragments.text[a:b].replace('\t', ' ' * int(self.fragments.tab_width)) + ('-' if h else '')
            for a, b, h in zip(line_starts, line_ends, hyphen_mask)
        ]