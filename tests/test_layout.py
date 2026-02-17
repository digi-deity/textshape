import re

import numpy as np

from test_textwrap import DUMMY_FONT, TEXTS, DIR
from textshape import FontMeasure, TextFragmenter
from textshape.text import MultiColumn, TextColumn
from textshape.layout import Layout


def test_max_lines_per_column():
    """Test the calculation of maximum lines per column based on page height and font size. We don't move "columns" here,
    but we will color what would have been a separate column. This also shows that 'newlines' should be ignored if they
    occur on column boundaries.
    """
    text = '\t' + '\n\n\t'.join(TEXTS)

    fontsize = 12
    width = 31 * fontsize
    fm = FontMeasure(DUMMY_FONT)

    f = TextFragmenter(measure=fm)
    fragments = f(text)
    column = MultiColumn(fragments, column_width=width, fontsize=fontsize, justify=True)

    text, x, dx, x_orig, y, dy, y_orig, c = column.to_bounding_boxes(max_lines_per_column=3, reset_y=False)

    svg = fm.render_svg(text, x_orig, y_orig, fontsize=fontsize, canvas_width=width)

    boundaries = re.compile(r"\t|\b[^\s]")
    separators = np.array([x.span()[0] for x in boundaries.finditer(text)], dtype=int)
    bins = np.zeros(len(text))
    bins[separators] = 1
    bins[0] = 0
    bins = bins.cumsum(dtype=int)

    select_x = x[separators]
    select_dx = np.bincount(bins, weights=dx, minlength=bins[-1])
    select_y = y[separators]
    select_dy = dy[separators]
    select_c = c[separators]

    newlines = np.array([x.span()[0] for x in re.compile('\n').finditer(text)], dtype=int)

    colors = np.array(["red", "blue", "green", "orange", "purple", "cyan", "magenta"])

    def draw_rect(x, y, dx, dy, color="red"):
        return f'<rect width="{dx:.2f}" height="{dy:.2f}" x="{x:.2f}" y="{y:.2f}" style="stroke:{color}"/>'

    rects = [
        draw_rect(_x, _y, _dx, _dy, colors[_z % len(colors)])
        for _x, _y, _dx, _dy, _z in zip(select_x, select_y, select_dx, select_dy, select_c)
        if _dx > 0.0 or _dy > 0
    ] + [
        # Draw filled 1x1 rectangles for newlines
        draw_rect(x[n], y[n]+0.5*dy[n], 2.0, 2.0, colors[c[n] % len(colors)]) for n in newlines
    ]
    rects = "\n".join(rects)
    rects = f'<g style="stroke-width:1;" fill-opacity="0">\n{rects}\n</g>\n</svg>'

    svg = svg.replace("</svg>", rects)

    with open(DIR / "text-multi-column.svg", "w") as f:
        f.write(svg)

def test_layout():
    text = '\t' + '\n\n\t'.join(TEXTS)

    fontsize = 12
    fm = FontMeasure(DUMMY_FONT)

    p_height = 300
    p_width = 600
    margin = 50

    layout = Layout(columns=2, column_spacing=15, page_size=(p_width, p_height), margins=margin)

    f = TextFragmenter(measure=fm)
    fragments = f(text)
    column = MultiColumn(fragments, column_width=layout.column_widths, fontsize=fontsize, justify=True)

    text, _, dx, x_orig, _, dy, y_orig, p = layout.to_bounding_boxes(column)
    y_orig = y_orig + p_height * p  # Adjust y to start from the top of the page

    # Only keep values for the first page
    svg = fm.render_svg(text, x_orig, y_orig, fontsize=fontsize, canvas_width=p_width, canvas_height=(p[-1]+1) * p_height)

    def draw_rect(x, y, dx, dy, color="red"):
        return f'<rect width="{dx:.2f}" height="{dy:.2f}" x="{x:.2f}" y="{y:.2f}" style="stroke:{color}"/>'

    # Draw a boundary line between pages
    rects = [
        draw_rect(margin, z*p_height, p_width - 2*margin, 1, "black") for z in range(1,p.max() + 1)
    ]

    # Draw a boundary line for the page margins
    rects += [
        draw_rect(margin, p_height * p + margin, p_width - 2*margin, p_height - 2*margin, "blue")
        for p in range(0, p[-1] + 1)
    ]

    rects = "\n".join(rects)
    rects = f'<g style="stroke-width:1;" fill-opacity="0">\n{rects}\n</g>\n</svg>'

    svg = svg.replace("</svg>", rects)

    with open(DIR / "text-layout.svg", "w") as f:
        f.write(svg)