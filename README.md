# TextShape

A high-performance Python library for shaping text into columns using advanced line breaking algorithms and precise character positioning with HarfBuzz. TextShape is built with performance in mind, vectorizing most operations with NumPy for efficient text layout operations.

## Features

- **Advanced Text Wrapping**: Intelligent line breaking with support for hyphenation
- **Multi-Column Page Layout**: Support for multiple columns on a page with customizable spacing for margins
- **Font-Aware Rendering**: Precise character positioning using HarfBuzz
- **Justification**: Full text justification support
- **Performance Optimized**: Vectorized operations with NumPy for speed
- **SVG Output**: Generate SVG visualizations of text

## Installation

Install TextShape using pip:

```bash
pip install textshape
```

### Requirements

- Python 3.9+
- NumPy >= 2.0.0
- vharfbuzz

Optional dependency for hyphenation:

- hyperhyphen

## Quick Start

```python
from textshape import FontMeasure, TextFragmenter, TextColumn

# Load a font
font_path = "path/to/your/font.ttf"
font_measure = FontMeasure(font_path)

# Create text fragments
text = "Your text here..."
fragmenter = TextFragmenter(measure=font_measure)
fragments = fragmenter(text)

# Create a text column
column = TextColumn(
    fragments=fragments,
    column_width=300,  # Width in points
    fontsize=12,
    justify=True
)

# Get bounding boxes for rendering
text, x, dx, x_origin, y, dy, y_origin = column.to_bounding_boxes()
```

## Usage Examples

### Basic Text Wrapping

```python
from textshape import FontMeasure, TextFragmenter, TextColumn

# Setup
font_measure = FontMeasure("fonts/NotoSans-Regular.ttf")
fragmenter = TextFragmenter(measure=font_measure)

text = """Whether I shall turn out to be the hero of my own life, or whether that 
station will be held by anybody else, these pages must show."""

# Fragment and wrap text
fragments = fragmenter(text)
column = TextColumn(
    fragments=fragments,
    column_width=31 * 12,  # 31 characters at 12pt
    fontsize=12,
    justify=False
)

# Get wrapped lines as strings
lines = column.to_list()
for i, line in enumerate(lines):
    print(f"{i+1:02d}: {line}")
```

### Multi-Column Layout

```python
from textshape import FontMeasure, TextFragmenter, MultiColumn

# Setup for multi-column layout
font_measure = FontMeasure("fonts/NotoSans-Regular.ttf")
fragmenter = TextFragmenter(measure=font_measure)

# Long text content
text = "Your long text content here..."
fragments = fragmenter(text)

# Create multi-column layout
multi_column = MultiColumn(
    fragments=fragments,
    column_width=250,
    fontsize=12,
    justify=True
)

# Get bounding boxes with column information
text, x, dx, x_origin, y, dy, y_origin, column_id = multi_column.to_bounding_boxes(
    max_lines_per_column=20,
    line_spacing=1.2
)
```

### Page Layout with Margins

```python
from textshape import FontMeasure, TextFragmenter, MultiColumn, Layout

# Setup
font_measure = FontMeasure("fonts/NotoSans-Regular.ttf")
fragmenter = TextFragmenter(measure=font_measure)

text = "Your document text..."
fragments = fragmenter(text)

# Create page layout
layout = Layout(
    columns=2,
    column_spacing=15,
    page_size=(600, 800),  # width, height
    margins=50  # uniform margins
)

# Create multi-column text
multi_column = MultiColumn(
    fragments=fragments,
    column_width=layout.column_widths,
    fontsize=12,
    justify=True
)

# Get positioned text for the layout
text, x, dx, x_origin, y, dy, y_origin, page = layout.to_bounding_boxes(multi_column)
```

### SVG Rendering

```python
# Generate SVG output
svg_content = font_measure.render_svg(
    text=text,
    x_origin=x_origin,
    y_origin=y_origin,
    fontsize=12,
    canvas_width=600,
    canvas_height=800
)

# Save to file
with open("output.svg", "w") as f:
    f.write(svg_content)
```

**Note on Coordinate System**: TextShape uses the SVG/HTML coordinate system where the origin (0, 0) is at the top-left corner, and the y-axis increases downward. All coordinates returned by `to_bounding_boxes()` follow this convention.

### Line breaking with hyphenation

```python
import re
from hyperhyphen import Hyphenator

hyph = Hyphenator(mode="spans", language="en_US")

# Use custom splitter
fragmenter = TextFragmenter(
    measure=font_measure,
    splitter=hyph,
)
```

## API Reference

### Core Classes

#### `FontMeasure`
Handles font loading and character measurement using HarfBuzz.

```python
FontMeasure(fontpath: str, features: Optional[dict] = None)
```

#### `TextFragmenter`
Breaks a string of text into atomic fragments. A fragment cannot be split further, and lines can only be broken at fragment boundaries.
Fragments can be full words, or parts of words if allowing for hyphenation.

```python
TextFragmenter(
    measure: Optional[Callable] = None,
    splitter: Optional[Callable] = None,
    tab_width: float | int = 4
)
```

#### `TextColumn`
Wraps text fragments into a single column.

```python
TextColumn(
    fragments: TextFragments,
    column_width: int | float | list[float],
    fontsize: int | float,
    justify: bool = False
)
```

#### `MultiColumn`
Extends TextColumn to support multiple columns.

```python
MultiColumn(
    fragments: TextFragments,
    column_width: int | float | list[float],
    fontsize: int | float,
    justify: bool = False
)
```

#### `Layout`
Manages page layout with multiple columns and margins.

```python
Layout(
    columns: int,
    column_spacing: float,
    page_size: tuple[float, float],
    margins: float | tuple[float, ...] 
)
```

## Advanced Features

### Text Justification
Enable full justification to align text to both left and right margins:

```python
column = TextColumn(fragments, column_width=300, fontsize=12, justify=True)
```

### Variable Column Widths
Support different widths for each line:

```python
import numpy as np

# Varying column widths
widths = np.linspace(200, 400, num_lines)
column = TextColumn(fragments, column_width=widths, fontsize=12)
```

### Custom Line Spacing
Control spacing between lines:

```python
text, x, dx, x_origin, y, dy, y_origin = column.to_bounding_boxes(line_spacing=1.5)
```

## Performance Tips

1. **Reuse FontMeasure objects** - Font loading is expensive
2. **Use vectorized operations** - The library is optimized for batch processing
3. **Cache fragments** - TextFragmenter results can be reused for different layouts
4. **Choose appropriate column widths** - Very narrow columns increase computation time

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

## Acknowledgments

- Built on top of [HarfBuzz](https://harfbuzz.github.io/) for text shaping
- Uses [NumPy](https://numpy.org/) for high-performance array operations
- Inspired by advanced typesetting systems