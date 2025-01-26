import re
import numpy as np

from hyperhyphen import Hyphenator
from textshape.shape import FontMeasure
from textshape.text import Text

TEXTS = [
        """
        Whether I shall turn out to be the hero of my own life, or whether that
        station will be held by anybody else, these pages must show. To begin my
        life with the beginning of my life, I record that I was born (as I have
        been informed and believe) on a Friday, at twelve o’clock at night.
        It was remarked that the clock began to strike, and I began to cry,
        simultaneously.
        """,
        """
        In consideration of the day and hour of my birth, it was declared by
        the nurse, and by some sage women in the neighbourhood who had taken a
        lively interest in me several months before there was any possibility
        of our becoming personally acquainted, first, that I was destined to be
        unlucky in life; and secondly, that I was privileged to see ghosts and
        spirits; both these gifts inevitably attaching, as they believed, to
        all unlucky infants of either gender, born towards the small hours on a
        Friday night.
        """,
        """
        I need say nothing here, on the first head, because nothing can show
        better than my history whether that prediction was verified or falsified
        by the result. On the second branch of the question, I will only remark,
        that unless I ran through that part of my inheritance while I was still
        a baby, I have not come into it yet. But I do not at all complain of
        having been kept out of this property; and if anybody else should be in
        the present enjoyment of it, he is heartily welcome to keep it.
        """,
        """
        I was born with a caul, which was advertised for sale, in the
        newspapers, at the low price of fifteen guineas. Whether sea-going
        people were short of money about that time, or were short of faith and
        preferred cork jackets, I don’t know; all I know is, that there was but
        one solitary bidding, and that was from an attorney connected with the
        bill-broking business, who offered two pounds in cash, and the balance
        in sherry, but declined to be guaranteed from drowning on any higher
        bargain. Consequently the advertisement was withdrawn at a dead
        loss--for as to sherry, my poor dear mother’s own sherry was in the
        market then--and ten years afterwards, the caul was put up in a raffle
        down in our part of the country, to fifty members at half-a-crown a
        head, the winner to spend five shillings. I was present myself, and I
        remember to have felt quite uncomfortable and confused, at a part of
        myself being disposed of in that way. The caul was won, I recollect, by
        an old lady with a hand-basket, who, very reluctantly, produced from it
        the stipulated five shillings, all in halfpence, and twopence halfpenny
        short--as it took an immense time and a great waste of arithmetic, to
        endeavour without any effect to prove to her. It is a fact which will
        be long remembered as remarkable down there, that she was never drowned,
        but died triumphantly in bed, at ninety-two. I have understood that it
        was, to the last, her proudest boast, that she never had been on the
        water in her life, except upon a bridge; and that over her tea (to which
        she was extremely partial) she, to the last, expressed her indignation
        at the impiety of mariners and others, who had the presumption to go
        ‘meandering’ about the world. It was in vain to represent to her
        that some conveniences, tea perhaps included, resulted from this
        objectionable practice. She always returned, with greater emphasis and
        with an instinctive knowledge of the strength of her objection, ‘Let us
        have no meandering.’
        """
    ]

TEXTS = [re.sub(r'\s+', ' ', x.strip()) for x in TEXTS]

def test_wrap_plaintext():
    h = Hyphenator(mode='spans')

    print('\n')
    for text in TEXTS:
        ft = Text(text, fragmenter=h)
        lines = ft.get_lines(30, 1)

        print('\n'.join([f'{len(l):02d}:  {l}' for l in lines]), end='\n\n')

        for line in lines:
            assert len(line) <= 30, f'This line is too long with {len(line)} characters: {line}'


def test_wrap_font_justified():
    h = Hyphenator(mode='spans')

    fontsize = 12
    width = 30*fontsize

    fm = FontMeasure('/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf')

    text = TEXTS[-1]
    ft = Text(text, fragmenter=h, measure=fm)
    text, x, dx, y, dy = ft.get_bboxes(width, fontsize, justify=True)
    svg = fm.render_svg(text, x, y, fontsize=fontsize, linewidth=width)
    with open('text-justified.svg', 'w') as f:
        f.write(svg)

def test_wrap_font():
    h = Hyphenator(mode='spans')

    fontsize = 12
    width = 30*fontsize

    fm = FontMeasure('/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf')

    text = TEXTS[-1]
    ft = Text(text, fragmenter=h, measure=fm)
    text, x, dx, y, dy = ft.get_bboxes(width, fontsize, justify=False)
    svg = fm.render_svg(text, x, y, fontsize=fontsize, linewidth=width)
    with open('text.svg', 'w') as f:
        f.write(svg)

def draw_rect(x, y, dx, dy):
    return f'<rect width="{dx:.2f}" height="{dy:.2f}" x="{x:.2f}" y="{y:.2f}"/>'

def test_wrap_font_selection():
    boundaries = re.compile(r'\b[^\s]')
    h = Hyphenator(mode='spans')

    fontsize = 12
    width = 30*fontsize

    fm = FontMeasure('/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf')

    text = TEXTS[-1]
    ft = Text(text, fragmenter=h, measure=fm)
    text, x, dx, y, dy = ft.get_bboxes(width, fontsize, justify=True, line_spacing=1.2)
    svg = fm.render_svg(text, x, y, fontsize=fontsize, linewidth=width)

    separators = np.array([x.span()[0] for x in boundaries.finditer(text)], dtype=int)
    bins = np.zeros(len(text))
    bins[separators] = 1
    bins[0] = 0
    bins = bins.cumsum(dtype=int)

    select_x = x[separators]
    select_dx = np.bincount(bins, weights=dx, minlength=bins[-1])
    select_y = y[separators]
    select_dy = dy[separators]

    rects = [
        draw_rect(_x, _y, _dx, _dy)
        for _x, _y, _dx, _dy in zip(select_x, select_y, select_dx, select_dy)
        if _dx > 0. or _dy > 0
    ]
    rects = '\n'.join(rects)
    rects = f'<g style="stroke-width:1;stroke:red;" fill-opacity="0">\n{rects}\n</g>\n</svg>'

    svg = svg.replace('</svg>', rects)

    with open('text-selection.svg', 'w') as f:
        f.write(svg)