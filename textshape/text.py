from typing import Callable, TYPE_CHECKING
from collections import deque

import numpy as np

from textshape.fragment import Fragments, word_fragmenter
from textshape.shape import monospace_measure, FontMeasure
from textshape.types import FloatVector, Span, IntVector
from textshape.wrap import wrap

if TYPE_CHECKING:
    import uharfbuzz._harfbuzz.Buffer

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
        ravel = spans.ravel(order='F')
        fragment_widths = cumwidths[ravel[1:]] - cumwidths[ravel[:2*m-1]]
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

    def wrap(self, width: float, fontsize: float) -> tuple[IntVector, IntVector]:
        width = width / fontsize
        fragment_breaks = np.array(wrap(self.fragments, width))
        line_breaks = self.end[fragment_breaks[1:]-1]
        penalties = self.fragments.penalty_widths[fragment_breaks[1:]-1] > 0
        hyphens = line_breaks[penalties]
        return line_breaks, hyphens

    def text_lines(self, linewidth: int):
            fragments = self.fragments
            n = len(fragments)
            breakpoints, hyphens = self.wrap(linewidth)

            lines = []
            a = breakpoints[0]
            for i in range(1, len(breakpoints)):
                line = []
                b = breakpoints[i]
                for i in range(a, b):
                    line.append(self.get_fragment_str(i))
                    if not i + 1 == b:
                        line.append(' ' * round(fragments.whitespace_widths[i]))
                if fragments.penalty_widths[i] > 0.0 and i + 1 < n:
                    line.append('-')
                lines.append(''.join(line))
                a = b

            return lines

    def get_fragment_str(self, i: int) -> str:
        return self.text[self.start[i]:self.end[i]]

    def render_svg(self, linewidth: int):
        """Converts a buffer to SVG

        Args:
            buf (hb.Buffer): uharfbuzz ``hb.Buffer``

        Returns: An SVG string containing a rendering of the buffer
        """
        if not isinstance(self.measure, FontMeasure):
            raise TypeError("Text measure must be a FontMeasure if you want to render")

        buf = self.measure.shape(self.text)
        linebreaks, hyphens = self.wrap(linewidth)

        defs = {}
        paths = []

        lb = deque(linebreaks)
        hp = deque(hyphens)

        vhb = self.measure.vhb
        hbfont = vhb.hbfont

        font_extents = hbfont.get_font_extents("ltr")
        y_max = font_extents.ascender
        y_min = font_extents.descender
        x_min = x_max = 0

        line_gap = font_extents.line_gap or (font_extents.ascender - font_extents.descender)
        lb_next = None if len(lb) == 0 else lb.popleft()
        hp_next = None if len(hp) == 0 else hp.popleft()
        buf_hyphen = self.measure.shape('-')

        x_cursor = 0
        y_cursor = y_max

        i = 0
        n_glyphs = len(buf)
        while i < n_glyphs:
            info = buf.glyph_infos[i]
            pos = buf.glyph_positions[i]
            cluster = info.cluster

            if hp_next and cluster > hp_next:
                hp_next = hp.popleft() if hp else None
                info = buf_hyphen.glyph_infos[0]
                pos = buf_hyphen.glyph_positions[0]
                i -= 1
            elif lb_next and cluster > lb_next:
                lb_next = lb.popleft() if lb else None
                x_cursor = 0
                y_cursor = y_cursor - line_gap

            dx, dy = pos.x_offset, pos.y_offset
            p = vhb._glyph_to_svg(info.codepoint, x_cursor + dx, y_cursor + dy, defs)
            paths.append(p)

            if extents := hbfont.get_glyph_extents(info.codepoint):
                cur_x = x_cursor + dx
                cur_y = y_cursor + dy
                min_x = cur_x + min(extents.x_bearing, 0)
                min_y = cur_y + min(extents.height + extents.y_bearing, pos.y_advance)
                max_x = cur_x + max(extents.width + extents.x_bearing, pos.x_advance)
                max_y = cur_y + max(extents.y_bearing, 0)
                x_min = min(x_min, min_x)
                y_min = min(y_min, min_y)
                x_max = max(x_max, max_x)
                y_max = max(y_max, max_y)

            x_cursor += pos.x_advance
            y_cursor += pos.y_advance
            i += 1

        # Add a empty border
        x_min = x_min - line_gap
        y_min = y_min - line_gap
        x_max = x_max + line_gap
        y_max = y_max + line_gap
        svg = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x_min} {y_min} {x_max - x_min} {y_max - y_min}" transform="matrix(1 0 0 -1 0 0)">',
            f'<rect x="{x_min}" y="{y_min}" width="{x_max - x_min}" height="{y_max - y_min}" fill="#BBBBBB"/>',
            f'<rect x="{x_min + line_gap}" y="{y_min + line_gap}" width="{x_max - x_min - 2*line_gap}" height="{y_max - y_min - 2*line_gap}" fill="#FFFFFF"/>',
            "<defs>",
            *defs.values(),
            "</defs>",
            *paths,
            "</svg>",
            "",
        ]

        return "\n".join(svg)