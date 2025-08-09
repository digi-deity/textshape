
import numpy as np

from textshape.shape import FontMeasure
from textshape.fragment import TextFragments
from textshape.types import FloatVector, IntVector, BoolVector, CharInfoVectors
from textshape.wrap import wrap

class TextColumn:
    def __init__(
        self,
        fragments: TextFragments,
        column_width: int | float | list[float | int] | FloatVector | IntVector,
        fontsize: int | float,
        justify: bool = False,
    ):
        self.fragments = fragments
        self.column_width: FloatVector = self.vectorize_input(column_width)
        self.fontsize = fontsize
        self.justify = justify

        line_starts, line_ends, hyphen_mask, forced_mask = self.wrap()
        self.line_starts = line_starts
        self.line_ends = line_ends
        self.hyphen_mask = hyphen_mask
        self.forced_mask = forced_mask

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

    def apply_justification(
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
        factors[self.forced_mask] = 0.0  # No justification in last line of paragraph with a forced line break

        offsets = np.zeros_like(dx_ws)
        offsets[0] = factors[0]
        offsets[linebreaks[:-1]] = np.diff(factors)
        offsets = offsets.cumsum() * dx_ws

        dx = dx + offsets
        x = np.pad(dx, (1, 0)).cumsum()
        return dx, x

    def vectorize_input(
        self,
        targets: int | float | list[float | int] | FloatVector | IntVector,
        dtype: type = float,
    ) -> FloatVector:
        """Ensure that input number is converted to a vector of floats."""
        if isinstance(targets, float | int):
            targets = [targets]

        targets_vector = np.array(targets, dtype=dtype)

        return targets_vector

    def to_bounding_boxes(
        self,
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

        text, x, dx, y, dy, _ = self._to_bounding_boxes(line_spacing)
        return self._convert_and_scale(text, x, dx, y, dy)

    def _text_to_array(self, text: str) -> FloatVector:
        """Convert a string to a numpy array of unicode code points."""
        return np.frombuffer(text.encode('utf-32-le'), dtype=np.uint32)

    def _array_to_text(self, arr: FloatVector) -> str:
        """Convert a numpy array of unicode code points to a string."""
        return arr.tobytes().decode('utf-32-le')

    def _convert_and_scale(self, text, x, dx, y, dy, *args):
        """Convert the text to a string and scale the coordinates by the fontsize."""
        text = self._array_to_text(text)
        fontsize = self.fontsize
        return (
            text,
            x * fontsize,
            dx * fontsize,
            y * fontsize,
            dy * fontsize,
            *args,
        )

    def _to_bounding_boxes(
        self,
        line_spacing: float = 1.0,
    ):
        assert isinstance(
            self.fragments.measure, FontMeasure
        ), "Calculating bboxes requires a FontMeasure to precisely measure text."

        # Modify text by inserting newline and hyphen characters at linebreak positions
        text, linebreaks, modified, targets_vector, charwidths = self.modify_text()

        # Determine Y-coordinates
        dy, y = self.calc_y(line_spacing, linebreaks, charwidths)

        # Determine X-coordinates
        dx, x = self.calc_x(linebreaks, modified, targets_vector, charwidths)

        return text, x, dx, y, dy, linebreaks

    def modify_text(self):
        text = self._text_to_array(self.fragments.text)
        widths = self.fragments.ch_widths
        if len(text) != len(widths):
            raise ValueError("Text and widths must have the same length. Something went wrong")

        line_starts, line_ends, hyphen_mask = (self.line_starts, self.line_ends, self.hyphen_mask)

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
        text = text[modified]
        widths = widths[modified]

        return text, linebreaks, modified, targets_vector, widths

    def calc_x(self, linebreaks, modified, targets_vector, dx):
        linebreaks_ = linebreaks[:-1]
        x = np.pad(dx, (1, 0)).cumsum()

        # Optional text justification
        if self.justify:
            dx_ws = dx * self.fragments.ch_ws_mask[modified - 2]
            ws = np.pad(dx_ws, (1, 0)).cumsum()
            dx, x = self.apply_justification(
                targets_vector / self.fontsize, dx, x, dx_ws, ws, linebreaks
            )

        resets = np.zeros_like(x)
        resets[linebreaks_] = np.diff(x[linebreaks_], prepend=0)
        x -= resets.cumsum()

        return dx, x[:-1]

    def calc_y(self, line_spacing, linebreaks, widths):
        linebreaks = linebreaks[:-1]

        line_gap = self.fragments.measure.line_gap
        y = np.zeros_like(widths)
        y[0] = -line_gap
        y[linebreaks] -= line_gap * line_spacing
        y = y.cumsum()
        dy = np.full_like(y, line_gap)
        return dy, y

    def to_list(
        self,
    ) -> list[str]:
        """Breaks the text into lines given a target width and fontsize.

        Note that this method includes hyphenation if words are broken.

        Returns a list of strings, each representing a line of text no longer than the target width.
        """
        line_starts, line_ends, hyphen_mask = (self.line_starts, self.line_ends, self.hyphen_mask)
        return [
            self.fragments.text[a:b].replace('\t', ' ' * int(self.fragments.tab_width)) + ('-' if h else '')
            for a, b, h in zip(line_starts, line_ends, hyphen_mask)
        ]


class MultiColumn(TextColumn):
    def __init__(
            self,
            fragments: TextFragments,
            column_width: int | float | list[float | int] | FloatVector | IntVector,
            fontsize: int | float,
            justify=False,
    ):
        super().__init__(fragments, column_width, fontsize, justify)

    def column_splitting(
            self,
            linebreaks: IntVector,
            max_lines_per_column: IntVector,
    ) -> BoolVector:
        """Greedily break the text lines such that they fit in a column of max length column_height."""
        split_mask = np.zeros(len(linebreaks), dtype=bool)
        drop_mask = np.zeros(len(linebreaks), dtype=bool)
        double_linebreak = np.pad(np.diff(linebreaks) == 1, (0, 1))

        i = max_lines_per_column[0] - 1
        j = 0
        n = len(split_mask) - 1
        m = len(max_lines_per_column) - 1
        while i < n:
            k = 1
            while double_linebreak[i - k]:
                drop_mask[i - k] = True
                k += 1

            while double_linebreak[i]:
                drop_mask[i] = True
                i += 1

            split_mask[i] = True
            max_lines = max_lines_per_column[min(j, m)]
            i += max_lines
            j += 1

        return split_mask, drop_mask

    def to_bounding_boxes(
            self,
            line_spacing: float = 1.0,
            max_lines_per_column: int | list[int] | IntVector | None = None,
            reset_y: bool = True,
    ):
        max_lines_per_column = self.vectorize_input(max_lines_per_column, dtype=int)
        text, x, dx, y, dy, linebreaks = super()._to_bounding_boxes(line_spacing)
        split_mask, _drop_mask = self.column_splitting(linebreaks, max_lines_per_column)

        if split_mask.sum() == 0:
            return self._convert_and_scale(text, x, dx, y, dy, np.zeros(len(text), dtype=int))

        # Assign each character to a column
        cid = np.zeros(len(text), dtype=int)
        cid[linebreaks[split_mask]] = 1
        cid = cid.cumsum()


        # Reset y-coordinates for each column
        if reset_y:
            splits = np.pad(linebreaks[split_mask], (1, 0))
            y_offset = np.repeat(y[splits], np.diff(splits, append=len(y)))
            y = y - y_offset - self.fragments.measure.line_gap

        # Drop trailing empty lines from each column
        drop_mask = np.ones(len(text), dtype=bool)
        drop_mask[linebreaks[_drop_mask]] = False
        text = text[drop_mask]
        cid = cid[drop_mask]
        x = x[drop_mask]
        dx = dx[drop_mask]
        y = y[drop_mask]
        dy = dy[drop_mask]

        return self._convert_and_scale(text, x, dx, y, dy, cid)
