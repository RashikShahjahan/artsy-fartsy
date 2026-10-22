from artcanvas import ArtCanvas

with ArtCanvas() as canvas:
    canvas.set_color(0, 0, 1)
    canvas.draw_circle(960, 600, 200, fill=True)
