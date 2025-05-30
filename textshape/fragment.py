import re
from typing import Callable

import numpy as np

from .shape import FontMeasure, monospace_measure

from .types import FloatVector, Span, IntVector
from .wrap import TextFragmentsBase

re_words = re.compile(r"\S+")  # matches whole words

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


