from test_textwrap import DUMMY_FONT, TEXTS

from textshape import FontMeasure, Fragments
from textshape.layout import MultiColumn

def test_wrap_force_newline_and_tabs():
    fontsize = 12
    width = 30 * fontsize
    linespacing = 1.0
    lineheight = fontsize * linespacing

    fm = FontMeasure(DUMMY_FONT)

    text = '\t' + '\n\n\t'.join(TEXTS)
    ft = Fragments(text, measure=fm, tab_width=2)

    MultiColumn(
        height=10*lineheight,
        width=width,
        text=ft,
    )