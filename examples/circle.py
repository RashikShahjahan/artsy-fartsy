import math
import cairo

surface = cairo.ImageSurface(cairo.FORMAT_RGB24, 1920, 1200)
ctx = cairo.Context(surface)
ctx.set_source_rgb(1, 1, 1)
ctx.paint()
ctx.set_source_rgb(0, 0, 1)
ctx.arc(960, 600, 200, 0, 2 * math.pi)
ctx.fill()
surface.write_to_png("output.png")
