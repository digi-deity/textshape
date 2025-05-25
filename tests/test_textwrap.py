from hashlib import md5
import re
import numpy as np
import pathlib

import pytest

from textshape.shape import FontMeasure
from textshape.fragment import TextFragmenter
from textshape import TextColumn

DIR = pathlib.Path(__file__).parent.resolve()
DUMMY_FONT = str(DIR / 'fonts/NotoSans-Regular.ttf')

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
        """,
]

TEXTS = [re.sub(r"\s+", " ", x.strip()) for x in TEXTS]
TEXTS_SPANS = {
    '764569e58f53ea8b6404f6fa7fc0247f': [(0, 5), (6, 12)],
    '4b8c1a80e64629618fceb9139c807c09': [(0, 7), (8, 9), (10, 15), (16, 20), (21, 24), (25, 27), (28, 30), (31, 34), (35, 39), (40, 42), (43, 45), (46, 49), (50, 55), (56, 58), (59, 66), (67, 71), (72, 79), (80, 84), (85, 87), (88, 92), (93, 95), (96, 103), (104, 109), (110, 115), (116, 121), (122, 126), (127, 132), (133, 135), (136, 141), (142, 144), (145, 149), (150, 154), (155, 158), (159, 164), (164, 168), (169, 171), (172, 174), (175, 180), (181, 182), (183, 189), (190, 194), (195, 196), (197, 200), (201, 205), (206, 209), (210, 211), (212, 216), (217, 221), (222, 230), (231, 234), (235, 243), (244, 246), (247, 248), (249, 256), (257, 259), (260, 266), (267, 274), (275, 277), (278, 284), (285, 287), (288, 291), (292, 300), (301, 305), (306, 309), (310, 315), (316, 321), (322, 324), (325, 332), (333, 336), (337, 338), (339, 344), (345, 347), (348, 352), (353, 358), (358, 360), (360, 362), (362, 365), (365, 368)],
    'e6522659fe4e1994f41410e161cfa17e': [(0, 2), (3, 9), (9, 11), (11, 12), (12, 16), (17, 19), (20, 23), (24, 27), (28, 31), (32, 36), (37, 39), (40, 42), (43, 49), (50, 52), (53, 56), (57, 65), (66, 68), (69, 72), (73, 79), (80, 83), (84, 86), (87, 91), (92, 96), (97, 102), (103, 105), (106, 109), (110, 115), (115, 119), (119, 123), (124, 127), (128, 131), (132, 137), (138, 139), (140, 146), (147, 152), (152, 155), (156, 158), (159, 161), (162, 169), (170, 176), (177, 183), (184, 189), (190, 193), (194, 197), (198, 203), (203, 206), (206, 209), (210, 212), (213, 216), (217, 222), (222, 225), (226, 232), (232, 236), (237, 245), (245, 248), (249, 255), (256, 260), (261, 262), (263, 266), (267, 275), (276, 278), (279, 281), (282, 289), (290, 292), (293, 298), (299, 302), (303, 309), (309, 312), (313, 317), (318, 319), (320, 323), (324, 328), (328, 329), (329, 334), (335, 337), (338, 341), (342, 348), (349, 352), (353, 357), (357, 361), (362, 366), (367, 372), (373, 378), (379, 389), (390, 396), (396, 400), (401, 403), (404, 408), (409, 418), (419, 421), (422, 425), (426, 433), (434, 441), (442, 444), (445, 451), (452, 459), (460, 464), (465, 472), (473, 476), (477, 482), (483, 488), (489, 491), (492, 493), (494, 500), (501, 507)],
    'feb705fb2b69c952134ac06bb774d4cf': [(0, 1), (2, 6), (7, 10), (11, 15), (15, 18), (19, 24), (25, 27), (28, 31), (32, 37), (38, 43), (44, 51), (52, 56), (56, 59), (60, 63), (64, 68), (69, 75), (76, 80), (81, 83), (84, 91), (92, 99), (100, 104), (105, 111), (111, 115), (116, 119), (120, 124), (124, 128), (129, 131), (132, 137), (137, 141), (142, 144), (145, 148), (149, 156), (157, 159), (160, 163), (164, 170), (171, 177), (178, 180), (181, 184), (185, 189), (189, 194), (195, 196), (197, 201), (202, 206), (207, 214), (215, 219), (220, 226), (227, 228), (229, 232), (233, 240), (241, 245), (246, 250), (251, 253), (254, 256), (257, 262), (262, 263), (263, 268), (269, 274), (275, 276), (277, 280), (281, 286), (287, 288), (289, 294), (295, 296), (297, 301), (302, 305), (306, 310), (311, 315), (316, 318), (319, 323), (324, 327), (328, 329), (330, 332), (333, 336), (337, 339), (340, 343), (344, 352), (353, 355), (356, 362), (363, 367), (368, 372), (373, 376), (377, 379), (380, 384), (385, 389), (389, 391), (391, 394), (395, 398), (399, 401), (402, 409), (410, 414), (415, 421), (422, 424), (425, 427), (428, 431), (432, 439), (440, 445), (445, 449), (450, 452), (453, 456), (457, 459), (460, 462), (463, 471), (472, 479), (480, 482), (483, 487), (488, 491)],
    'fd57da8da14c36eeb53d7789f81da094': [(0, 1), (2, 5), (6, 10), (11, 15), (16, 17), (18, 23), (24, 29), (30, 33), (34, 39), (39, 44), (45, 48), (49, 54), (55, 57), (58, 61), (62, 66), (66, 68), (68, 73), (74, 76), (77, 80), (81, 84), (85, 90), (91, 93), (94, 101), (102, 110), (111, 118), (119, 125), (125, 128), (129, 135), (136, 140), (141, 146), (147, 149), (150, 155), (156, 161), (162, 166), (167, 172), (173, 175), (176, 180), (181, 186), (187, 189), (190, 195), (196, 199), (200, 209), (210, 214), (215, 219), (219, 223), (224, 225), (226, 231), (232, 237), (238, 241), (242, 243), (244, 248), (249, 252), (253, 257), (258, 263), (264, 267), (268, 271), (272, 275), (276, 280), (280, 284), (285, 293), (294, 297), (298, 302), (303, 306), (307, 311), (312, 314), (315, 320), (320, 323), (324, 333), (334, 338), (339, 342), (343, 355), (356, 360), (360, 365), (366, 369), (370, 377), (378, 381), (382, 388), (389, 391), (392, 397), (398, 401), (402, 405), (406, 413), (414, 416), (417, 421), (421, 424), (425, 428), (429, 437), (438, 440), (441, 443), (444, 448), (448, 450), (450, 454), (455, 459), (460, 465), (465, 468), (469, 471), (472, 475), (476, 482), (483, 491), (492, 497), (497, 504), (505, 508), (509, 514), (514, 518), (518, 522), (523, 526), (527, 531), (531, 536), (537, 539), (540, 541), (542, 546), (547, 556), (557, 559), (560, 562), (563, 567), (567, 570), (571, 573), (574, 578), (579, 583), (584, 592), (593, 596), (597, 603), (604, 607), (608, 610), (611, 614), (615, 621), (622, 631), (632, 635), (636, 641), (642, 647), (647, 653), (654, 657), (658, 662), (663, 666), (667, 670), (671, 673), (674, 676), (677, 678), (679, 685), (686, 690), (691, 693), (694, 697), (698, 702), (703, 705), (706, 709), (710, 714), (714, 718), (719, 721), (722, 727), (728, 735), (736, 738), (739, 751), (752, 753), (754, 759), (760, 763), (764, 770), (771, 773), (774, 779), (780, 784), (785, 795), (796, 797), (798, 801), (802, 809), (810, 817), (818, 821), (822, 823), (824, 829), (829, 832), (833, 835), (836, 840), (841, 845), (846, 851), (852, 857), (857, 861), (861, 865), (866, 869), (870, 879), (880, 882), (883, 884), (885, 889), (890, 892), (893, 899), (900, 905), (906, 914), (915, 917), (918, 920), (921, 925), (926, 930), (931, 934), (935, 939), (940, 943), (944, 948), (949, 950), (951, 956), (956, 961), (962, 964), (965, 967), (968, 971), (972, 976), (977, 981), (982, 983), (984, 992), (992, 996), (997, 1001), (1002, 1006), (1007, 1012), (1012, 1016), (1016, 1019), (1020, 1028), (1029, 1033), (1034, 1036), (1037, 1040), (1041, 1045), (1045, 1046), (1046, 1051), (1052, 1056), (1057, 1067), (1068, 1071), (1072, 1074), (1075, 1079), (1079, 1085), (1086, 1089), (1090, 1098), (1099, 1103), (1103, 1108), (1109, 1118), (1119, 1121), (1122, 1126), (1127, 1129), (1130, 1137), (1138, 1142), (1143, 1146), (1147, 1148), (1149, 1154), (1155, 1160), (1161, 1163), (1164, 1169), (1169, 1175), (1176, 1178), (1179, 1185), (1185, 1188), (1189, 1193), (1193, 1196), (1197, 1200), (1201, 1207), (1208, 1210), (1211, 1216), (1217, 1219), (1220, 1224), (1225, 1227), (1228, 1230), (1231, 1232), (1233, 1237), (1238, 1243), (1244, 1248), (1249, 1251), (1252, 1256), (1257, 1262), (1262, 1267), (1268, 1270), (1271, 1277), (1277, 1281), (1282, 1286), (1287, 1293), (1294, 1298), (1299, 1302), (1303, 1306), (1307, 1312), (1313, 1321), (1322, 1325), (1326, 1330), (1331, 1343), (1344, 1346), (1347, 1351), (1352, 1354), (1355, 1366), (1367, 1368), (1369, 1373), (1374, 1379), (1379, 1384), (1385, 1389), (1390, 1392), (1393, 1397), (1398, 1400), (1401, 1404), (1405, 1410), (1411, 1414), (1415, 1420), (1420, 1423), (1424, 1430), (1431, 1435), (1436, 1439), (1440, 1445), (1446, 1449), (1450, 1454), (1455, 1457), (1458, 1461), (1462, 1467), (1468, 1470), (1471, 1474), (1475, 1480), (1481, 1487), (1488, 1492), (1493, 1494), (1495, 1502), (1503, 1506), (1507, 1511), (1512, 1516), (1517, 1520), (1521, 1524), (1525, 1528), (1529, 1534), (1535, 1538), (1539, 1542), (1543, 1552), (1553, 1561), (1562, 1566), (1567, 1569), (1570, 1573), (1574, 1579), (1580, 1589), (1590, 1593), (1594, 1599), (1599, 1601), (1601, 1605), (1606, 1608), (1609, 1612), (1613, 1617), (1617, 1620), (1621, 1623), (1624, 1632), (1633, 1636), (1637, 1644), (1645, 1648), (1649, 1652), (1653, 1656), (1657, 1664), (1664, 1668), (1669, 1671), (1672, 1674), (1675, 1680), (1680, 1683), (1683, 1687), (1688, 1693), (1694, 1697), (1698, 1704), (1705, 1707), (1708, 1711), (1712, 1714), (1715, 1719), (1720, 1722), (1723, 1728), (1728, 1732), (1733, 1735), (1736, 1739), (1740, 1744), (1745, 1749), (1750, 1755), (1755, 1763), (1764, 1767), (1768, 1775), (1776, 1782), (1782, 1785), (1786, 1794), (1795, 1799), (1800, 1804), (1805, 1810), (1810, 1814), (1814, 1818), (1819, 1823), (1823, 1828), (1829, 1832), (1833, 1839), (1840, 1849), (1850, 1854), (1855, 1862), (1863, 1868), (1868, 1871), (1872, 1875), (1876, 1880), (1881, 1883), (1884, 1891), (1891, 1895), (1896, 1901), (1901, 1905), (1906, 1908), (1909, 1912), (1913, 1921), (1922, 1924), (1925, 1928), (1929, 1934), (1934, 1939), (1940, 1944), (1945, 1947), (1948, 1952), (1953, 1955), (1956, 1960), (1960, 1963), (1963, 1968)]
}

def dummy_splitter(text):
    """Split the test text into words and spaces using hardcoded splits."""

    try:
        return TEXTS_SPANS[md5(text.encode()).hexdigest()]
    except KeyError as e:
        raise NotImplementedError("Pre-fragmented test not found in case.") from e


def test_wrap_plaintext():
    f = TextFragmenter(splitter=dummy_splitter)

    print("\n")
    for text in TEXTS:
        fragments = f(text)
        column = TextColumn(fragments, column_width=30, fontsize=1)
        lines = column.to_list()

        print("\n".join([f"{len(l):02d}:  {l}" for l in lines]), end="\n\n")

        for line in lines:
            assert (
                len(line) <= 30
            ), f"This line is too long with {len(line)} characters: {line}"

def test_wrap_plaintext_force_newline_and_tabs():
    text = '\t' + '\n\n\t'.join(TEXTS)  # Create tab indents and force double linebreak between paragraphs

    f = TextFragmenter()
    fragments = f(text)
    column = TextColumn(fragments, column_width=30, fontsize=1)
    lines = column.to_list()

    print('\n')
    print("\n".join([f"{len(l):02d}:  {l}" for l in lines]), end="\n\n")

    force_count = 0
    tab_count = 0
    for line in lines:
        assert (
            len(line) <= 30
        ), f"This line is too long with {len(line)} characters: {line}"

        with pytest.raises(ValueError):
            assert line.index('\n'), "A newline character leaked into the output text"

        tab_count += line.count('    ')
        if len(line) == 0:
            force_count += 1

    assert tab_count == 4, "Expected 4 tabs"
    assert force_count == 3, "Expected 3 empty lines due to forced linebreaks in between paragraphs"

def test_oneliner():
    text = "Hello world."

    fontsize = 12
    width = 30 * fontsize
    fm = FontMeasure(DUMMY_FONT)

    f = TextFragmenter(measure=fm, splitter=dummy_splitter)
    fragments = f(text)
    column = TextColumn(fragments, column_width=width, fontsize=fontsize)
    text, x, dx, y, dy = column.to_bounding_boxes()

    svg = fm.render_svg(text, x, y, fontsize=fontsize, linewidth=width)
    with open(DIR / "text-oneliner.svg", "w") as f:
        f.write(svg)

def test_wrap_font():
    text = TEXTS[-1]

    fontsize = 12
    width = 30 * fontsize
    fm = FontMeasure(DUMMY_FONT)

    f = TextFragmenter(measure=fm, splitter=dummy_splitter)
    fragments = f(text)
    column = TextColumn(fragments, column_width=width, fontsize=fontsize)

    text, x, dx, y, dy = column.to_bounding_boxes(justify=False)
    svg = fm.render_svg(text, x, y, fontsize=fontsize, linewidth=width)
    with open(DIR / "text.svg", "w") as f:
        f.write(svg)


def test_wrap_font_justified():
    text = TEXTS[-1]

    fontsize = 12
    width = 30 * fontsize
    fm = FontMeasure(DUMMY_FONT)

    f = TextFragmenter(measure=fm, splitter=dummy_splitter)
    fragments = f(text)
    column = TextColumn(fragments, column_width=width, fontsize=fontsize)

    text, x, dx, y, dy = column.to_bounding_boxes(justify=True)
    svg = fm.render_svg(text, x, y, fontsize=fontsize, linewidth=width)
    with open(DIR / "text-justified.svg", "w") as f:
        f.write(svg)

def test_heterogeneous_widths():
    text = TEXTS[-1]

    fontsize = 12
    width = np.hstack([np.arange(10, 47, 2), np.arange(10, 46, 2)[::-1]]) * fontsize
    fm = FontMeasure(DUMMY_FONT)

    f = TextFragmenter(measure=fm, splitter=dummy_splitter)
    fragments = f(text)
    column = TextColumn(fragments, column_width=width, fontsize=fontsize)

    text, x, dx, y, dy = column.to_bounding_boxes(justify=True)
    svg = fm.render_svg(text, x, y, fontsize=fontsize, linewidth=46 * fontsize)
    with open(DIR / "text-heterogeneous.svg", "w") as f:
        f.write(svg)

def test_wrap_force_newline_and_tabs():
    text = '\t' + '\n\n\t'.join(TEXTS)

    fontsize = 12
    width = 30 * fontsize
    fm = FontMeasure(DUMMY_FONT)

    f = TextFragmenter(measure=fm, tab_width=1)
    fragments = f(text)
    column = TextColumn(fragments, column_width=width, fontsize=fontsize)

    text, x, dx, y, dy = column.to_bounding_boxes(justify=True)
    svg = fm.render_svg(text, x, y, fontsize=fontsize, linewidth=width)
    with open(DIR / "text-force-newline-and-tabs.svg", "w") as f:
        f.write(svg)


def draw_rect(x, y, dx, dy):
    return f'<rect width="{dx:.2f}" height="{dy:.2f}" x="{x:.2f}" y="{y:.2f}"/>'


def test_wrap_font_selection():
    text = TEXTS[-1]

    fontsize = 12
    width = 30 * fontsize
    fm = FontMeasure(DUMMY_FONT)

    f = TextFragmenter(measure=fm, splitter=dummy_splitter)
    fragments = f(text)
    column = TextColumn(fragments, column_width=width, fontsize=fontsize)

    text, x, dx, y, dy = column.to_bounding_boxes(justify=True, line_spacing=1.2)
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

    rects = [
        draw_rect(_x, _y, _dx, _dy)
        for _x, _y, _dx, _dy in zip(select_x, select_y, select_dx, select_dy)
        if _dx > 0.0 or _dy > 0
    ]
    rects = "\n".join(rects)
    rects = f'<g style="stroke-width:1;stroke:red;" fill-opacity="0">\n{rects}\n</g>\n</svg>'

    svg = svg.replace("</svg>", rects)

    with open(DIR / "text-selection.svg", "w") as f:
        f.write(svg)
