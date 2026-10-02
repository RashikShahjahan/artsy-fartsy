from artcanvas import ArtCanvas

with ArtCanvas(256, 256, filename="output.png") as canvas:
    canvas.fill_background(0.95, 0.97, 1.0)
    canvas.set_color(0.2, 0.4, 0.8)
    canvas.draw_circle(128, 128, 64, fill=True)
