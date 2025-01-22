from textshape.shape import FontMeasure
import re

def test_multibyte_unicode():
    fontpath = '/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf'

    fm = FontMeasure(fontpath, {'liga': True, 'clig': True, 'kern': True})
    out = fm.character_widths('a\u031A! ffi. ffi')

def test_render_svg():
    fontpath = '/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf'

    fm = FontMeasure(fontpath)
    buf = fm.shape(text)
    svg = fm.render_svg(buf, linebreaks)

    with open('text.svg', 'w') as f:
        f.write(svg)