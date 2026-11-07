# ArtCanvas API v2.1
The default canvas is 1920×1200, white background, black brush.
Colors are floats from 0 to 1; coordinates are pixels; angles are radians.
ArtCanvas wraps the installed Cairo bindings; agents should use `from artcanvas import ArtCanvas`.
Dimensions must be integers from 1 to 4096. Episode environments require their
configured dimensions (1920×1200 by default). Use `math` and seeded `random` as needed.

```python
from artcanvas import ArtCanvas
with ArtCanvas(1920, 1200, "output.png") as canvas:
    canvas.set_color(0, 0, 1)
    canvas.draw_circle(960, 600, 200, fill=True)
```

Core methods (use these exact argument orders):

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
# Composable paths (segments are drawn only by stroke/fill):
new_path()
move_to(x, y)
line_to(x, y)
curve_to(cx1, cy1, cx2, cy2, x, y)
close_path()
stroke()
fill()
# Transforms and brush state:
save_state()
restore_state()
translate(x, y)
rotate(radians)
scale(x, y)
save()
```

Gradient stops are `(offset, r, g, b, a)` tuples. `save()` finishes the surface;
repeated saves are harmless. A context manager saves only on successful exit.
`save_state()`/`restore_state()` push/pop Cairo brush settings and transforms;
these do not write a file. Keep state pushes and pops balanced.
Save as `output.png` in the current directory.

## Detail primitives (v2.1)

All new methods are additive; existing v2 programs continue to work. Angles for
`set_pattern` are **degrees**; angles for `rotate` and `draw_arc` are **radians**.

```python
# Ellipse radii, not width/height; ellipse_path appends without drawing:
draw_ellipse(x, y, radius_x, radius_y, fill=False)
ellipse_path(x, y, radius_x, radius_y)

# Current path and clip:
clip()                  # intersect clip with current path; consumes path
reset_clip()            # reset clip; save_state/restore_state also manage clips
fill_preserve()         # keep path so it can be stroked too
stroke_preserve()
set_line_cap(cap)       # "butt", "round", "square"
set_line_join(join)     # "miter", "round", "bevel"

# Repeating, seeded patterns:
set_pattern(kind, spacing=8, angle=0, line_width=1, opacity=1,
            color=None, seed=0)
fill_pattern(kind, spacing=8, angle=0, line_width=1, opacity=1,
             color=None, seed=0)

# Reusable geometry:
path = create_path()    # capture the current composable path
draw_path(path, fill=False, stroke=False)

# Transparency, compositing, and optional Gaussian blur:
begin_layer(opacity=1, blend_mode="over")
end_layer(blur_radius=0)
# Or, automatically match begin/end:
with canvas.layer(opacity=0.7, blend_mode="multiply", blur_radius=3):
    ...
```

**Patterns:** `kind` is `"hatching"`, `"crosshatch"`, `"dots"`,
`"stipple"`, or `"noise"`. `spacing` and `line_width` are in pixels
(positive, spacing 1–512); `angle` is degrees, `opacity` is 0–1.
`color` is optional `(r,g,b)` or `(r,g,b,a)` (floats 0–1).
Without a color, the current solid brush color is used; gradients fall back
to black. `seed` controls the randomized stipple/noise placement. Calls
are deterministic for identical arguments. `set_pattern` changes the brush
source and can be used with any `fill()`, `stroke()`, or shape method.
`fill_pattern` paints across the **current clip** and restores the previous
brush source; clip to a shape before calling it.

**Paths:** `draw_path(path)` strokes by default. Use `fill=True` to fill,
or `fill=True, stroke=True` to draw both. `create_path` keeps the original
current path unchanged. Surround reused drawings with `save_state()`,
`translate`/`rotate`/`scale`, and `restore_state()`.

**Layers:** `blend_mode` is `"over"`, `"multiply"`, `"screen"`,
`"overlay"`, `"add"`, `"darken"`, or `"lighten"`. Layers nest,
are composed at `end_layer`, and support `blur_radius` 0–100 pixels
(0 means no blur). Balance begin/end before saving. A context-managed
`layer()` automatically ends the layer. Blur is rasterized in the Cairo
image backend and requires Pillow (already installed in the runtime).

### Detailed shape example

```python
from artcanvas import ArtCanvas

with ArtCanvas(1920, 1200, "output.png") as c:
    c.fill_background(1, 1, 1)

    # Save an organic leaf silhouette.
    c.new_path()
    c.move_to(400, 600)
    c.curve_to(500, 350, 850, 300, 1000, 600)
    c.curve_to(850, 900, 500, 850, 400, 600)
    c.close_path()
    leaf = c.create_path()

    # Fill and texture only inside the silhouette.
    c.save_state()
    c.new_path()
    c.ctx.append_path(leaf)  # alternatively reconstruct the path with move_to/curve_to
    c.clip()
    c.set_gradient(400, 400, 1000, 800, [
        (0, 0.08, 0.25, 0.05, 1),
        (1, 0.35, 0.75, 0.15, 1),
    ])
    c.draw_rectangle(400, 300, 600, 600, fill=True)
    c.fill_pattern("hatching", spacing=5, angle=30, line_width=0.7,
                   opacity=0.25, color=(0.05, 0.25, 0.03))
    c.restore_state()

    c.set_color(0.03, 0.2, 0.05)
    c.set_line_width(3)
    c.draw_path(leaf)

    # A softly blurred, semi-transparent highlight.
    with c.layer(opacity=0.35, blend_mode="screen", blur_radius=12):
        c.set_color(1, 1, 0.5)
        c.draw_ellipse(680, 460, 130, 30, fill=True)
```
