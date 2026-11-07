import math
import random
from contextlib import contextmanager
import cairo
from PIL import Image, ImageFilter

__version__ = "2.1.0"


class ArtCanvas:
    def __init__(self, width=1920, height=1200, filename="output.png"):
        if not all(isinstance(n, int) and not isinstance(n, bool) and 1 <= n <= 4096 for n in (width, height)):
            raise ValueError("Canvas dimensions must be integers between 1 and 4096")
        self._finished = False
        self._layers = []
        # Initialize canvas
        self.width = width
        self.height = height
        self.filename = filename
        
        # Initialize surface and context (moved from __enter__)
        self.surface = cairo.ImageSurface(cairo.FORMAT_RGB24, self.width, self.height)
        self.ctx = cairo.Context(self.surface)
        
        # Set default white background
        self.ctx.set_source_rgb(1, 1, 1)
        self.ctx.paint()
        self.ctx.set_line_width(1)
        self.ctx.set_source_rgb(0, 0, 0)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.save()
        elif not self._finished:
            self.surface.finish()
            self._finished = True

    def move_brush_to(self, x, y):
        self.ctx.move_to(x, y)

    def draw_line_to(self, x, y):
        self.ctx.line_to(x, y)
        self.ctx.stroke()  # Add this line to make the path visible
        self.ctx.move_to(x, y)  # Move to end point for next line

    def draw_arc(self, xc, yc, radius, start_angle, end_angle):
        # move to start point the circumference of the arc
        self.ctx.move_to(xc + radius * math.cos(start_angle), yc + radius * math.sin(start_angle))
        self.ctx.arc(xc, yc, radius, start_angle, end_angle)
        self.ctx.stroke()  # Add this line to make the arc visible

    def draw_rectangle(self, x, y, width, height, fill=False):
        """Draw a rectangle with top-left corner at (x,y)"""
        self.ctx.rectangle(x, y, width, height)
        if fill:
            self.ctx.fill()
        else:
            self.ctx.stroke()

    def draw_circle(self, x, y, radius, fill=False):
        """Draw a circle centered at (x,y)"""
        self.ctx.arc(x, y, radius, 0, 2 * math.pi)
        if fill:
            self.ctx.fill()
        else:
            self.ctx.stroke()

    def set_color(self, r, g, b, a=1.0):
        """Set drawing color (RGB values between 0 and 1)"""
        self.ctx.set_source_rgba(r, g, b, a)

    def set_line_width(self, width):
        """Set the width of lines being drawn"""
        self.ctx.set_line_width(width)

    def draw_text(self, x, y, text, font_size=16, font_family="Sans"):
        """Draw text at position (x,y)"""
        self.ctx.select_font_face(font_family)
        self.ctx.set_font_size(font_size)
        self.ctx.move_to(x, y)
        self.ctx.show_text(text)

    def fill_background(self, r, g, b):
        """Fill the entire canvas with a solid color"""
        self.ctx.set_source_rgb(r, g, b)
        self.ctx.paint()
        self.ctx.set_source_rgb(0, 0, 0)  # Reset to black

    def draw_polygon(self, points, fill=False):
        """Draw a polygon from a list of (x,y) points"""
        if not points or len(points) < 3:
            return
        self.ctx.move_to(*points[0])
        for x, y in points[1:]:
            self.ctx.line_to(x, y)
        self.ctx.close_path()
        if fill:
            self.ctx.fill()
        else:
            self.ctx.stroke()

    def draw_bezier_curve(self, x1, y1, cx1, cy1, cx2, cy2, x2, y2):
        """Draw a cubic bezier curve from (x1,y1) to (x2,y2) with control points (cx1,cy1) and (cx2,cy2)"""
        self.ctx.move_to(x1, y1)
        self.ctx.curve_to(cx1, cy1, cx2, cy2, x2, y2)
        self.ctx.stroke()

    def set_gradient(self, x1, y1, x2, y2, stops):
        """Create a linear gradient. stops should be a list of (offset, r, g, b, a) tuples"""
        gradient = cairo.LinearGradient(x1, y1, x2, y2)
        for offset, r, g, b, a in stops:
            gradient.add_color_stop_rgba(offset, r, g, b, a)
        self.ctx.set_source(gradient)

    def set_radial_gradient(self, cx1, cy1, radius1, cx2, cy2, radius2, stops):
        """Create a radial gradient. stops should be a list of (offset, r, g, b, a) tuples"""
        gradient = cairo.RadialGradient(cx1, cy1, radius1, cx2, cy2, radius2)
        for offset, r, g, b, a in stops:
            gradient.add_color_stop_rgba(offset, r, g, b, a)
        self.ctx.set_source(gradient)

    def set_dash_pattern(self, dashes, offset=0):
        """Set a dash pattern for lines. dashes is a list of dash lengths"""
        self.ctx.set_dash(dashes, offset)

    def draw_rounded_rectangle(self, x, y, width, height, radius, fill=False):
        """Draw a rectangle with rounded corners"""
        self.ctx.new_path()
        self.ctx.arc(x + radius, y + radius, radius, math.pi, 3 * math.pi / 2)
        self.ctx.arc(x + width - radius, y + radius, radius, 3 * math.pi / 2, 0)
        self.ctx.arc(x + width - radius, y + height - radius, radius, 0, math.pi / 2)
        self.ctx.arc(x + radius, y + height - radius, radius, math.pi / 2, math.pi)
        self.ctx.close_path()
        if fill:
            self.ctx.fill()
        else:
            self.ctx.stroke()

    def save(self):
        """Save the image to file and cleanup"""
        if not self._finished:
            if self._layers:
                raise RuntimeError("unclosed ArtCanvas layers")
            self.surface.write_to_png(self.filename)
            self.surface.finish()
            self._finished = True

    def new_path(self):
        self.ctx.new_path()

    def move_to(self, x, y):
        self.ctx.move_to(x, y)

    def line_to(self, x, y):
        """Append a segment without immediately stroking it."""
        self.ctx.line_to(x, y)

    def curve_to(self, cx1, cy1, cx2, cy2, x, y):
        self.ctx.curve_to(cx1, cy1, cx2, cy2, x, y)

    def close_path(self):
        self.ctx.close_path()

    def stroke(self):
        self.ctx.stroke()

    def fill(self):
        self.ctx.fill()

    def save_state(self):
        """Push brush settings and the current transform (does not save a file)."""
        self.ctx.save()

    def restore_state(self):
        self.ctx.restore()

    def translate(self, x, y):
        self.ctx.translate(x, y)

    def rotate(self, radians):
        self.ctx.rotate(radians)

    def scale(self, x, y):
        self.ctx.scale(x, y)

    # Detailed shapes and compound paths
    def ellipse_path(self, x, y, radius_x, radius_y):
        """Append an ellipse centered at (x, y) to the current path."""
        if not all(math.isfinite(v) for v in (x, y, radius_x, radius_y)):
            raise ValueError("Ellipse coordinates and radii must be finite")
        if radius_x <= 0 or radius_y <= 0:
            raise ValueError("Ellipse radii must be positive")
        self.ctx.save()
        try:
            self.ctx.translate(x, y)
            self.ctx.scale(radius_x, radius_y)
            self.ctx.arc(0, 0, 1, 0, 2 * math.pi)
        finally:
            self.ctx.restore()

    def draw_ellipse(self, x, y, radius_x, radius_y, fill=False):
        """Draw an ellipse centered at (x, y)."""
        self.ctx.new_path()
        self.ellipse_path(x, y, radius_x, radius_y)
        if fill:
            self.ctx.fill()
        else:
            self.ctx.stroke()

    def clip(self):
        """Intersect the current clip with the current path; consumes that path."""
        self.ctx.clip()

    def reset_clip(self):
        """Reset the clip to the full canvas (or current group's extent)."""
        self.ctx.reset_clip()

    def fill_preserve(self):
        """Fill the current path without discarding it."""
        self.ctx.fill_preserve()

    def stroke_preserve(self):
        """Stroke the current path without discarding it."""
        self.ctx.stroke_preserve()

    def set_line_cap(self, cap):
        """Set cap to 'butt', 'round', or 'square'."""
        caps = {
            "butt": cairo.LINE_CAP_BUTT,
            "round": cairo.LINE_CAP_ROUND,
            "square": cairo.LINE_CAP_SQUARE,
        }
        if cap not in caps:
            raise ValueError("line cap must be 'butt', 'round', or 'square'")
        self.ctx.set_line_cap(caps[cap])

    def set_line_join(self, join):
        """Set join to 'miter', 'round', or 'bevel'."""
        joins = {
            "miter": cairo.LINE_JOIN_MITER,
            "round": cairo.LINE_JOIN_ROUND,
            "bevel": cairo.LINE_JOIN_BEVEL,
        }
        if join not in joins:
            raise ValueError("line join must be 'miter', 'round', or 'bevel'")
        self.ctx.set_line_join(joins[join])

    # Deterministic, seamlessly tiled texture sources.
    def _pattern_source(self, kind, spacing, angle, line_width, opacity, color, seed):
        kinds = ("hatching", "crosshatch", "dots", "stipple", "noise")
        if kind not in kinds:
            raise ValueError("pattern kind must be one of: " + ", ".join(kinds))
        if not all(isinstance(v, (int, float)) and math.isfinite(v)
                   for v in (spacing, angle, line_width, opacity)):
            raise ValueError("pattern parameters must be finite numbers")
        if not 1 <= spacing <= 512 or line_width <= 0 or not 0 <= opacity <= 1:
            raise ValueError("spacing must be 1..512, line_width > 0, opacity 0..1")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed must be an integer")
        if color is None:
            source = self.ctx.get_source()
            try:
                rgba = source.get_rgba()
            except AttributeError:
                rgba = (0, 0, 0, 1)
        else:
            if len(color) not in (3, 4):
                raise ValueError("color must contain 3 or 4 RGB(A) components")
            rgba = tuple(color) + ((1,) if len(color) == 3 else ())
        if not all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1
                   for v in rgba):
            raise ValueError("color components must be finite values in 0..1")
        r, g, b, a = rgba
        a *= opacity

        step = max(1, round(spacing))
        size = step if kind in ("hatching", "crosshatch", "dots") else step * 8
        tile = cairo.ImageSurface(cairo.FORMAT_ARGB32, size, size)
        brush = cairo.Context(tile)
        brush.set_line_cap(cairo.LINE_CAP_ROUND)
        brush.set_source_rgba(r, g, b, a)
        brush.set_line_width(line_width)
        if kind in ("hatching", "crosshatch"):
            brush.move_to(0, step / 2)
            brush.line_to(step, step / 2)
            if kind == "crosshatch":
                brush.move_to(step / 2, 0)
                brush.line_to(step / 2, step)
            brush.stroke()
        elif kind == "dots":
            brush.arc(step / 2, step / 2, min(line_width, step) / 2, 0, 2 * math.pi)
            brush.fill()
        else:
            rng = random.Random(seed)
            for row in range(8):
                for col in range(8):
                    if kind == "stipple" and rng.random() < 0.15:
                        continue
                    x = (col + 0.5 + rng.uniform(-0.35, 0.35)) * step
                    y = (row + 0.5 + rng.uniform(-0.35, 0.35)) * step
                    size_factor = rng.uniform(0.35, 0.9) if kind == "noise" else rng.uniform(0.55, 1.2)
                    brush.set_source_rgba(r, g, b, a * rng.uniform(0.25, 1.0))
                    brush.arc(x, y, min(step * 0.45, line_width * size_factor) / 2,
                              0, 2 * math.pi)
                    brush.fill()
        pattern = cairo.SurfacePattern(tile)
        pattern.set_extend(cairo.EXTEND_REPEAT)
        pattern.set_filter(cairo.FILTER_BILINEAR)
        # The source matrix maps user coordinates to pattern coordinates.
        matrix = cairo.Matrix()
        matrix.rotate(-math.radians(angle))
        pattern.set_matrix(matrix)
        return pattern

    def set_pattern(self, kind, spacing=8, angle=0, line_width=1,
                    opacity=1, color=None, seed=0):
        """Set a repeating texture as the current source. Angle is in degrees.

        Color defaults to the current solid brush color (or black for gradients).
        Use set_pattern() followed by fill()/stroke()/draw_*().
        """
        self.ctx.set_source(self._pattern_source(
            kind, spacing, angle, line_width, opacity, color, seed))

    def fill_pattern(self, kind, spacing=8, angle=0, line_width=1,
                     opacity=1, color=None, seed=0):
        """Paint a texture over the current clip, preserving the current source.

        Call clip() on a shape first to restrict the pattern to that shape.
        """
        self.ctx.save()
        try:
            self.set_pattern(kind, spacing, angle, line_width, opacity, color, seed)
            self.ctx.paint()
        finally:
            self.ctx.restore()

    # Shape reuse: define a path once, draw it under different transforms.
    def create_path(self):
        """Return an immutable snapshot of the current path."""
        return self.ctx.copy_path()

    def append_path(self, path):
        """Append a saved path to the current composable path without drawing."""
        self.ctx.append_path(path)

    def draw_path(self, path, fill=False, stroke=False):
        """Draw a saved path, optionally filling and outlining it.

        Defaults to a stroke when neither fill nor stroke is requested.
        """
        self.ctx.new_path()
        self.ctx.append_path(path)
        if fill and stroke:
            self.ctx.fill_preserve()
            self.ctx.stroke()
        elif fill:
            self.ctx.fill()
        else:
            self.ctx.stroke()

    # Transparent layers, blend modes, and optional soft-focus effects.
    _BLEND_OPERATORS = {
        "over": cairo.OPERATOR_OVER,
        "multiply": cairo.OPERATOR_MULTIPLY,
        "screen": cairo.OPERATOR_SCREEN,
        "overlay": cairo.OPERATOR_OVERLAY,
        "add": cairo.OPERATOR_ADD,
        "darken": cairo.OPERATOR_DARKEN,
        "lighten": cairo.OPERATOR_LIGHTEN,
    }

    def begin_layer(self, opacity=1, blend_mode="over"):
        """Open a transparent drawing layer, to be composited by end_layer()."""
        if not isinstance(opacity, (float, int)) or not math.isfinite(opacity) or not 0 <= opacity <= 1:
            raise ValueError("layer opacity must be in 0..1")
        if blend_mode not in self._BLEND_OPERATORS:
            raise ValueError("unsupported blend mode: " + str(blend_mode))
        self.ctx.push_group()
        self._layers.append((opacity, blend_mode))

    def end_layer(self, blur_radius=0):
        """Composite the latest layer, optionally Gaussian-blurred in pixels."""
        if not self._layers:
            raise RuntimeError("end_layer() without begin_layer()")
        if (not isinstance(blur_radius, (int, float)) or not math.isfinite(blur_radius)
                or not 0 <= blur_radius <= 100):
            raise ValueError("blur_radius must be in 0..100 pixels")
        opacity, mode = self._layers[-1]
        source = self.ctx.pop_group()
        self._layers.pop()
        if blur_radius:
            # Cairo image surfaces store premultiplied BGRA on little-endian
            # systems. Pillow RGBa retains premultiplication while blurring,
            # avoiding dark fringes on translucent shapes.
            import sys
            if sys.byteorder != "little":
                raise RuntimeError("blurred layers require a little-endian runtime")
            surface = source.get_surface()
            width, height = surface.get_width(), surface.get_height()
            image = Image.frombytes(
                "RGBa", (width, height), bytes(surface.get_data()),
                "raw", "BGRa", surface.get_stride(), 1)
            pixels = bytearray(image.filter(
                ImageFilter.GaussianBlur(blur_radius)).tobytes("raw", "BGRa"))
            blurred = cairo.ImageSurface.create_for_data(
                pixels, cairo.FORMAT_ARGB32, width, height, width * 4)
            blurred.set_device_offset(*surface.get_device_offset())
            replacement = cairo.SurfacePattern(blurred)
            replacement.set_matrix(source.get_matrix())
            source = replacement

        self.ctx.save()
        try:
            self.ctx.set_operator(self._BLEND_OPERATORS[mode])
            self.ctx.set_source(source)
            self.ctx.paint_with_alpha(opacity)
        finally:
            self.ctx.restore()

    @contextmanager
    def layer(self, opacity=1, blend_mode="over", blur_radius=0):
        """Context-managed layer: with canvas.layer(0.7, 'multiply'): ..."""
        self.begin_layer(opacity, blend_mode)
        try:
            yield self
        finally:
            self.end_layer(blur_radius)
