import numpy as np
import uharfbuzz
from vharfbuzz import Vharfbuzz

from textshape.types import FloatVector
from typing import Optional
from collections import deque

class FontMeasure():
    def __init__(self, fontpath: str, features: Optional[dict] = None):
        self.fontpath = fontpath
        self.vhb = Vharfbuzz(fontpath)
        self.params = {'features': features or {}}

        # em is the unit of measurement for a font
        self.em = self.vhb.shape('\u2003', self.params).glyph_positions[0].x_advance
        
        font_extents = self.vhb.hbfont.get_font_extents("ltr")
        self.ascender = font_extents.ascender / self.em
        self.descender = font_extents.descender / self.em

    def __call__(self, text) -> FloatVector:
        return self.character_widths(text)

    def shape(self, text) -> uharfbuzz.Buffer:
        if not text:
            raise ValueError("No text provided")

        return self.vhb.shape(text, self.params)

    def character_widths(self, text, buf: Optional[uharfbuzz.Buffer] = None) -> FloatVector:
        """Maps the shaped glyphs back to input characters to determine the width of each character.
        Width is expressed in em units.

        Characters that merge into one glyph (e.g. letter + accent modifier) will share equal proportion of the glyph
        width.
        """
        buf = buf or self.shape(text)
        n = len(text)

        clusters = np.array([i.cluster for i in buf.glyph_infos])
        x_advances = np.array([p.x_advance for p in buf.glyph_positions])

        # Handle case where codepoint(s) decompose into more glyphs
        widths = np.bincount(clusters, weights=x_advances, minlength=n)

        # Handle case where codepoint(s) are merged into fewer glyphs
        diff = np.diff(clusters, append=n)
        jumps = diff>1
        merge_starts = clusters[jumps]
        merge_lengths = diff[jumps]

        for start, length in zip(merge_starts, merge_lengths):
            widths[start:start+length] = widths[start] / length

        # Alternatively, this loop can be replaced by fully vectorized functions.
        # However since cluster merges are rare, I think the loop will always be faster.
        #
        # widths[merge_starts] /= merge_lengths
        # idx = np.zeros(len(text))
        # idx[clusters] = clusters
        # idx = np.maximum.accumulate(idx, dtype=int)
        # widths = widths[idx]

        return widths / self.em

    def render_svg(self, text: str, x: FloatVector, y: FloatVector, fontsize: float, linewidth: float):
        """Converts a buffer to SVG

        Args:
            buf (hb.Buffer): uharfbuzz ``hb.Buffer``

        Returns: An SVG string containing a rendering of the buffer
        """

        defs = {}
        paths = []

        buf = self.shape(text)
        vhb = self.vhb


        x = x * self.em
        y = y * self.em

        x_cursor = 0
        y_cursor = 0

        i = 0
        n_glyphs = len(buf)
        prev_cluster = -1
        prev_x_advance = 0
        prev_y_advance = 0
        while i < n_glyphs:
            info = buf.glyph_infos[i]
            pos = buf.glyph_positions[i]
            cluster = info.cluster

            if cluster > prev_cluster:
                x_cursor = x[cluster]
                y_cursor = y[cluster]
            else:
                # Something interesting with clustering has happened. Advance according to harfbuzz suggestion.
                x_cursor += prev_x_advance
                y_cursor += prev_y_advance

            p = vhb._glyph_to_svg(info.codepoint, x_cursor + pos.x_offset, y_cursor + pos.y_offset, defs)
            paths.append(p)

            prev_y_advance = pos.y_advance
            prev_x_advance = pos.x_advance
            prev_cluster = cluster
            i += 1

        # Add a empty border and rescale
        s = fontsize / self.em
        font_extents = vhb.hbfont.get_font_extents("ltr")
        line_gap = (font_extents.line_gap or font_extents.ascender - font_extents.descender) * s

        x_min = 0
        x_max = linewidth
        y_max = font_extents.ascender * s
        y_min = (y[-1] + font_extents.descender) * s

        x_min = x_min - line_gap
        y_min = y_min - line_gap
        x_max = x_max + line_gap
        y_max = y_max + line_gap

        svg = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x_min} {y_min} {x_max - x_min} {y_max - y_min}" transform="matrix(1 0 0 -1 0 0)">',
            f'<rect x="{x_min}" y="{y_min}" width="{x_max - x_min}" height="{y_max - y_min}" fill="#BBBBBB"/>',
            f'<rect x="{x_min + line_gap}" y="{y_min + line_gap}" width="{x_max - x_min - 2*line_gap}" height="{y_max - y_min - 2*line_gap}" fill="#FFFFFF"/>',
            '<defs>',
            *defs.values(),
            "</defs>",
            f'<g transform="scale({s}, {s})">',
            *paths,
            "</g>",
            "</svg>",
            "",
        ]

        return "\n".join(svg)

def monospace_measure(s: str) -> FloatVector:
    return [1.0] * len(s)
