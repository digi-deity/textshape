import re

import numpy as np

from test_textwrap import DUMMY_FONT, TEXTS, DIR
from textshape import FontMeasure, TextFragmenter
from textshape.text import MultiColumn, TextColumn


def test_multi_column():
    text = '\t' + '\n\n\t'.join(TEXTS)

    fontsize = 12
    width = 31 * fontsize
    fm = FontMeasure(DUMMY_FONT)

    f = TextFragmenter(measure=fm)
    fragments = f(text)
    column = MultiColumn(fragments, column_width=width, fontsize=fontsize, max_lines_per_column=3)

    text, x, dx, y, dy, z = column.to_bounding_boxes(justify=True)
    svg = fm.render_svg(text, x, y, fontsize=fontsize, linewidth=width)

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
    select_z = z[separators]

    newlines = np.array([x.span()[0] for x in re.compile('\n').finditer(text)], dtype=int)

    colors = np.array(["red", "blue", "green", "orange", "purple", "cyan", "magenta"])

    def draw_rect(x, y, dx, dy, color="red"):
        return f'<rect width="{dx:.2f}" height="{dy:.2f}" x="{x:.2f}" y="{y:.2f}" style="stroke:{color}"/>'

    rects = [
        draw_rect(_x, _y, _dx, _dy, colors[_z % len(colors)])
        for _x, _y, _dx, _dy, _z in zip(select_x, select_y, select_dx, select_dy, select_z)
        if _dx > 0.0 or _dy > 0
    ] + [
        # Draw filled 1x1 rectangles for newlines
        draw_rect(x[n], y[n]+0.5*dy[n], 2.0, 2.0, colors[z[n] % len(colors)]) for n in newlines
    ]
    rects = "\n".join(rects)
    rects = f'<g style="stroke-width:1;" fill-opacity="0">\n{rects}\n</g>\n</svg>'

    svg = svg.replace("</svg>", rects)

    with open(DIR / "text-multi-column.svg", "w") as f:
        f.write(svg)