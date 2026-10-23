"""Seeded shapes, gradients, composable paths, and local transforms."""
import math
import random
from artcanvas import ArtCanvas

with ArtCanvas() as canvas:
    canvas.set_gradient(0, 0, 1920, 1200, [(0, 0.04, 0.08, 0.2, 1), (1, 0.2, 0.4, 0.6, 1)])
    canvas.draw_rectangle(0, 0, 1920, 1200, fill=True)
    for i in range(12):
        canvas.save_state()
        canvas.translate(160 + (i % 6) * 320, 350 + (i // 6) * 500)
        canvas.rotate(random.uniform(-math.pi / 4, math.pi / 4))
        canvas.set_color(1, 0.7 + random.random() * 0.3, 0.3)
        canvas.new_path()
        canvas.move_to(0, -100)
        canvas.line_to(90, 80)
        canvas.curve_to(40, 120, -40, 120, -90, 80)
        canvas.close_path()
        canvas.fill()
        canvas.restore_state()
