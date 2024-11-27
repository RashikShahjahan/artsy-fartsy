import math
import cairo

class ArtCanvas:
    def __init__(self, width=1920, height=1200, filename="output.png"):
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
        # Save the image before cleanup
        if self.surface:
            self.surface.write_to_png(self.filename)
            self.surface.finish()

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
        if self.surface:
            self.surface.write_to_png(self.filename)
            self.surface.finish()
