# ArtCanvas API
The original default canvas is 1920×1200, white background, black brush.
Colors are floats from 0 to 1; coordinates are pixels; angles are radians.
The runtime supplies Cairo-compatible CFFI bindings for its `import cairo`.

```python
from artcanvas import ArtCanvas
with ArtCanvas(1920, 1200, "output.png") as canvas:
    canvas.set_color(0, 0, 1)
    canvas.draw_circle(960, 600, 200, fill=True)
```

Methods (use these exact argument orders):

```python
ArtCanvas(width=1920, height=1200, filename="output.png")
move_brush_to(x, y)
draw_line_to(x, y)
draw_arc(xc, yc, radius, start_angle, end_angle)
draw_rectangle(x, y, width, height, fill=False)
draw_circle(x, y, radius, fill=False)
set_color(r, g, b, a=1.0)
set_line_width(width)
draw_text(x, y, text, font_size=16, font_family="Sans")
fill_background(r, g, b)  # resets brush color to black
draw_polygon(points, fill=False)  # [(x,y), ...], at least 3 points
draw_bezier_curve(x1, y1, cx1, cy1, cx2, cy2, x2, y2)
set_gradient(x1, y1, x2, y2, stops)
set_radial_gradient(cx1, cy1, radius1, cx2, cy2, radius2, stops)
set_dash_pattern(dashes, offset=0)
draw_rounded_rectangle(x, y, width, height, radius, fill=False)
save()
```

Gradient stops are `(offset, r, g, b, a)` tuples. `save()` finishes the surface;
use it once at the end OR use a context manager that saves on exit.
Save as `output.png` in the current directory.
