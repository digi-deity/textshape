import numpy as np

from .text import MultiColumn
from .types import Numeric


class Layout:
    """
    A class to lay out text columns across a page.

    Allows for multiple columns of text to be laid out across a page, with options for
    specifying the number of columns, their widths, and the spacing between, the page height
    and page width, page top/bottom and left/right margins.
    """

    def __init__(
        self,
        columns: int,
        column_spacing: Numeric,
        page_size: tuple[Numeric, Numeric],
        margins: float | tuple[Numeric, Numeric] | tuple[Numeric, Numeric, Numeric, Numeric]
    ):
        """
        Initialize the Layout with the given parameters.

        Args:
            columns: Number of columns to be created.
            column_spacing: Spacing between columns.
            page_size: Size of the page as (height, width).
            margins: Margins of the page.
                If a single float is provided, it applies to all margins.
                If a tuple of two floats is provided, it applies to top/bottom and left/right margins.
                If a tuple of four floats is provided, it applies to top, bottom, left, and right margins.
        """
        if isinstance(margins, (float, int)):
            margins_tuple = (margins, margins, margins, margins)
        elif isinstance(margins, tuple) and len(margins) == 2:
            margins_tuple = (margins[0], margins[0], margins[1], margins[1])
        elif isinstance(margins, tuple) and len(margins) == 4:
            margins_tuple = margins
        else:
            raise ValueError("Margins must be a numeric or a tuple of 2 or 4 numerics.")

        if columns < 1:
            raise ValueError("Number of columns must be at least 1.")

        self.columns = columns
        self.column_spacing = column_spacing
        self.page_width = page_size[0]
        self.page_height = page_size[1]
        self.page_top_margin = margins_tuple[0]
        self.page_bottom_margin = margins_tuple[1]
        self.page_left_margin = margins_tuple[2]
        self.page_right_margin = margins_tuple[3]

        column_widths = (self.page_width - self.page_left_margin - self.page_right_margin -
                        (self.columns - 1) * self.column_spacing) / self.columns
        column_widths = np.full(columns, column_widths, dtype=np.float32)

        if len(column_widths) != columns:
            raise ValueError(f"Column widths must have length {columns}.")

        self.column_widths = np.array(column_widths, dtype=np.float32)

    def max_lines_per_column(self, line_height: float) -> int:
        """
        Calculate the maximum number of lines that can fit in each column based on the page height,
        top and bottom margins, and the font size.

        Args:
            fontsize (float): The font size used for the text.

        Returns:
            int: The maximum number of lines per column.
        """
        usable_height = self.page_height - self.page_top_margin - self.page_bottom_margin
        return int(usable_height // line_height)

    def column_xy(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the x and y coordinates for each column based on the layout parameters.

        Returns:
            tuple[np.ndarray, np.ndarray]: Arrays of x and y coordinates for each column.
        """

        x = self.page_left_margin + np.pad(np.cumsum(self.column_widths[:-1] + self.column_spacing), (1, 0))
        y = np.full_like(x, self.page_top_margin)

        return x, y

    def to_bounding_boxes(
            self,
            multi_column: MultiColumn,
            line_spacing: float = 1.0,
    ) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Adjust the (x,y) coordinates of the text to fit within the specified columns in accordance
        with the layout parameters.
        """

        line_height = multi_column.fragments.measure.line_gap * multi_column.fontsize * line_spacing
        max_lines = self.max_lines_per_column(line_height)
        text, x, dx, x_orig, y, dy, y_orig, c = multi_column.to_bounding_boxes(
            max_lines_per_column=max_lines,
            line_spacing=line_spacing
        )

        x_move, y_move = self.column_xy()

        x += x_move[c % self.columns]
        x_orig += x_move[c % self.columns]
        y += y_move[c % self.columns]
        y_orig += y_move[c % self.columns]

        p = c // self.columns

        return (
            text,
            x,
            dx,
            x_orig,
            y,
            dy,
            y_orig,
            p, # page index
        )