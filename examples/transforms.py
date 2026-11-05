"""Seeded shapes, gradients, composable paths, and local transforms."""
import math
import random
import cairo

surface = cairo.ImageSurface(cairo.FORMAT_RGB24, 1920, 1200)
ctx = cairo.Context(surface)
gradient = cairo.LinearGradient(0, 0, 1920, 1200)
gradient.add_color_stop_rgb(0, 0.04, 0.08, 0.2)
gradient.add_color_stop_rgb(1, 0.2, 0.4, 0.6)
ctx.set_source(gradient)
ctx.paint()
for i in range(12):
    ctx.save()
    ctx.translate(160 + (i % 6) * 320, 350 + (i // 6) * 500)
    ctx.rotate(random.uniform(-math.pi / 4, math.pi / 4))
    ctx.set_source_rgb(1, 0.7 + random.random() * 0.3, 0.3)
    ctx.new_path()
    ctx.move_to(0, -100)
    ctx.line_to(90, 80)
    ctx.curve_to(40, 120, -40, 120, -90, 80)
    ctx.close_path()
    ctx.fill()
    ctx.restore()
surface.write_to_png("output.png")
