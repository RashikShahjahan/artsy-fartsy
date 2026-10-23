"""Deterministic, natural computational drawing repair curriculum."""

from __future__ import annotations

import hashlib
from typing import Any

REPAIR_CURRICULUM_VERSION = "artsy-algorithmic-natural-repairs-v3"

TASKS: list[dict[str, Any]] = [{'family': 'landscapes',
  'name': 'voronoi-mosaic',
  'skills': ['half-plane clipping', 'degenerate geometry', 'gradients'],
  'instructions': ['Draw a stained-glass Voronoi mosaic inside the rectangle from (180,160) to (1740,980)',
                   'Use 24 sites sampled with the supplied random seed, then replace the final two sites '
                   'with duplicates of the first two',
                   'Merge coincident sites so the final diagram has exactly 22 cells',
                   'Construct cells by intersecting perpendicular-bisector half-planes with the frame, using '
                   'Python geometry rather than external libraries',
                   'Handle vertical bisectors, parallel edges, boundary vertices, and empty intermediate '
                   'polygons consistently',
                   'Fill each cell with a radial gradient centered at its site using alternating teal, '
                   'violet, and gold palettes',
                   'Draw each shared cell edge once with a dark 5-pixel stroke and a small white dot at each '
                   'unique site',
                   'Keep every cell inside the frame and leave no cracks between adjacent cells']},
 {'family': 'triangles',
  'name': 'delaunay-network',
  'skills': ['triangulation', 'circumcircles', 'edge deduplication'],
  'instructions': ['Draw a Delaunay triangulation of 28 sites inside a wide rectangular frame',
                   'Include the corners of a square centered at (960,560), four sites on the horizontal line '
                   'y=560, and twenty seeded random sites',
                   'Handle the cocircular square and collinear subsets deterministically without duplicate '
                   'triangles',
                   'Implement an incremental triangulation and remove any temporary super-triangle geometry '
                   'from the final drawing',
                   "Fill triangles by their centroid's vertical position with a continuous blue-to-orange "
                   'palette',
                   'Draw each unique interior edge once and give the convex hull a thicker outline',
                   'Draw a small white circle at every original site and pale dashed circumcircles for '
                   'exactly three interior triangles',
                   'Clip all visible circumcircle strokes to the frame by computing their visible arc '
                   'intervals']},
 {'family': 'circles',
  'name': 'apollonian-packing',
  'skills': ['Descartes circles', 'complex coordinates', 'recursive deduplication'],
  'instructions': ['Draw an Apollonian circle packing inside a large enclosing circle centered at (960,550)',
                   'Start from an enclosing circle of radius 430 and three mutually tangent inner circles',
                   'Compute tangent-circle centers and curvatures with the Descartes relation rather than '
                   'approximating positions by eye',
                   'Recursively fill curvilinear gaps through five generations, stopping for circles smaller '
                   'than four pixels',
                   'Deduplicate circles reached through different recursion paths and preserve the signed '
                   'curvature of the enclosing circle',
                   'Give alternate recursion generations teal, gold, and violet outlines without filling the '
                   'gaps',
                   'Use line widths proportional to radius with a visible lower bound, and give the '
                   'enclosing circle a dark outline',
                   'Keep the packing symmetric and ensure circles are tangent without overlaps crossing the '
                   'enclosure']},
 {'family': 'circles',
  'name': 'involute-gears',
  'skills': ['involute curves', 'polar transforms', 'tangent layout'],
  'instructions': ['Draw a row of three meshing involute gears with 12, 20, and 28 teeth',
                   'Use module 12, a 20-degree pressure angle, and center distances determined by the sum of '
                   'pitch radii',
                   'Sample both involute flanks of each tooth and join them to root and tip arcs with closed '
                   'polygonal outlines',
                   'Handle radii below the base circle without invalid square roots or discontinuous tooth '
                   'roots',
                   'Rotate adjacent gears by the correct phase so their teeth interleave at the pitch '
                   'circles',
                   'Fill the three gears with orange, teal, and violet diagonal gradients and leave white '
                   'circular axle holes',
                   'Give each gear six radial spokes, six small bolts, and a thin dashed pitch-circle '
                   'overlay',
                   'Keep all gears inside the drawing area and avoid drawing duplicate tooth-boundary '
                   'segments']},
 {'family': 'lines',
  'name': 'variable-width-ribbon',
  'skills': ['curve offsets', 'arc-length sampling', 'join geometry'],
  'instructions': ['Draw a smooth variable-width ribbon through waypoints (220,700), (430,300), (730,700), '
                   '(730,700), (1110,280), (1450,760), and (1700,440)',
                   'Merge consecutive duplicate waypoints before fitting a centripetal Catmull-Rom spline',
                   'Resample the centerline by approximate arc length and vary ribbon width smoothly between '
                   '35 and 100 pixels',
                   'Compute both ribbon boundaries from local normals, maintaining a consistent normal '
                   'orientation',
                   'Use bevel joins when a miter would exceed three ribbon widths and round caps at the '
                   'endpoints',
                   'Fill the ribbon with a teal-to-purple gradient and draw a thin white dashed centerline',
                   'Place exactly 24 gold markers at equal arc-length intervals along the ribbon',
                   "Keep the ribbon's two boundaries continuous through the repeated waypoint and the "
                   'sharpest bends']},
 {'family': 'lines',
  'name': 'marching-squares',
  'skills': ['contour extraction', 'saddle cases', 'polyline stitching'],
  'instructions': ['Draw a contour map of f(x,y)=sin(3*x)*cos(2*y)+0.35*sin(5*(x+y)) over a 64-by-40 '
                   'sampling grid',
                   'Use eight equally spaced contour levels spanning the sampled value range',
                   'Implement marching squares with interpolation along cell edges and a consistent '
                   'saddle-cell disambiguation rule',
                   'Handle samples exactly on a contour level and equal-valued endpoints consistently',
                   'Stitch neighboring contour segments into continuous polylines without duplicate edges',
                   'Color successive contour levels from navy through teal to gold, with every second level '
                   'thicker',
                   'Draw a pale translucent checkerboard of grid cells underneath the contours and a dark '
                   'rectangular frame',
                   'Keep all contour paths inside the frame and preserve closed contours without spurious '
                   'connecting strokes']},
 {'family': 'rows',
  'name': 'torus-knot-tube',
  'skills': ['3D frames', 'surface normals', 'depth sorting'],
  'instructions': ['Draw a shaded three-dimensional tube following the closed torus-knot centerline '
                   '((2+cos(3*t))*cos(2*t),(2+cos(3*t))*sin(2*t),sin(3*t))',
                   'Sample 180 centerline positions and twelve vertices around each tube cross-section',
                   'Construct smoothly transported local frames so adjacent cross-sections do not flip '
                   'orientation',
                   'Close both the centerline seam and each circular cross-section without cracks',
                   'Rotate the mesh around two axes and project it with a finite-distance perspective camera',
                   'Compute face normals and shade faces with a fixed directional light, clamping color '
                   'channels to the documented range',
                   "Sort faces by camera depth before drawing so the knot's overpasses occlude its "
                   'underpasses',
                   'Use a continuous teal-to-violet material palette and a pale offset shadow underneath the '
                   'knot']},
 {'family': 'rectangles',
  'name': 'isometric-city',
  'skills': ['3D projection', 'occlusion sorting', 'polygon clipping'],
  'instructions': ['Draw an isometric city on a four-by-four grid with sixteen buildings of seeded varying '
                   'heights',
                   'Project all building corners from three-dimensional coordinates using one shared '
                   'projection function',
                   'Draw and depth-sort the visible roof and wall faces so nearer buildings correctly hide '
                   'farther ones',
                   'Give each exposed wall exactly three columns of windows and as many complete rows as fit '
                   'its height',
                   'Clip window polygons to their wall boundaries using two-dimensional convex-polygon '
                   'clipping',
                   'Use a different shading value for each of the three face orientations while preserving a '
                   'teal-and-orange palette',
                   'Place four small cubic rooftop structures on the four tallest buildings and draw their '
                   'visible faces consistently',
                   "Add a paved base with grid-aligned roads and keep all objects within the base's "
                   'projected footprint']},
 {'family': 'triangles',
  'name': 'penrose-rosette',
  'skills': ['recursive substitution', 'shared vertices', 'edge deduplication'],
  'instructions': ['Draw a five-fold Penrose-style rosette beginning with ten triangular sectors around a '
                   'central point',
                   'Apply a documented golden-ratio Robinson-triangle subdivision through six levels',
                   'Track triangle type and orientation so neighboring subdivisions share exactly the same '
                   'boundary vertices',
                   'Merge numerically equivalent vertices and draw shared edges once',
                   'Color the two triangle types with alternating gold and violet gradients',
                   'Clip the subdivided patch to a regular decagonal frame by computing polygon '
                   'intersections',
                   'Skip empty or zero-area clipped pieces while retaining valid triangles touching the '
                   'frame',
                   'Keep the final rosette five-fold symmetric and give the decagonal boundary a dark '
                   'outline']},
 {'family': 'flowers',
  'name': 'closed-spline-flower',
  'skills': ['centripetal splines', 'periodic boundaries', 'curvature'],
  'instructions': ['Draw a closed asymmetric flower outline interpolating twelve alternating outer and inner '
                   'control points around an ellipse',
                   'Include a repeated first point at the end of the control list and a consecutive repeated '
                   'point near the top',
                   'Remove redundant points and fit a periodic centripetal Catmull-Rom spline with a smooth '
                   'closing seam',
                   'Sample the outline at equal approximate arc-length intervals and compute local tangent '
                   'and normal directions',
                   'Draw forty short gold normal ticks outside the flower without reversing their '
                   'orientation at the seam',
                   'Fill the flower with a radial violet-to-teal gradient and a thin dark outline',
                   'Draw a second smooth curve inset from the outline and place twelve white beads equally '
                   'along it',
                   'Keep all inset geometry inside the outer outline, including at its narrowest concave '
                   'bends']},
 {'family': 'triangles',
  'name': 'clipped-fractal',
  'skills': ['recursive geometry', 'convex clipping', 'degenerate polygons'],
  'instructions': ['Draw a depth-seven Sierpinski triangle pattern centered behind a regular 24-sided '
                   'circular-looking window',
                   'Compute recursive triangle vertices and retain alternating terminal triangles in teal '
                   'and gold',
                   'Clip every retained triangle to the window using a general convex polygon clipper',
                   'Handle triangles whose vertices lie exactly on the window boundary and intersections '
                   'producing fewer than three distinct vertices',
                   'Discard zero-area pieces and merge consecutive equal polygon vertices before drawing',
                   'Add a second, smaller rotated fractal layer in translucent violet with its own clipped '
                   'pieces',
                   'Draw the window boundary as a dark outline and add exactly 24 evenly spaced white '
                   'perimeter dots',
                   'Keep all fractal fills inside the window and avoid cracks along adjacent clipped edges']},
 {'family': 'lines',
  'name': 'line-arrangement-glass',
  'skills': ['segment intersections', 'planar graph faces', 'angular sorting'],
  'instructions': ['Draw a stained-glass panel formed by the arrangement of eighteen seeded line segments '
                   'clipped to a rectangular frame',
                   'Include three vertical segments, three horizontal segments, two parallel diagonals, and '
                   'ten additional seeded segments',
                   'Compute intersections robustly, merging shared endpoints and handling parallel and '
                   'collinear cases',
                   'Split segments at intersections and construct a planar half-edge graph',
                   "Walk the graph's bounded faces in angular order, excluding the unbounded exterior face",
                   'Fill every bounded face with a centroid-based teal, violet, or gold gradient',
                   'Draw each shared segment once with a dark border and a small white dot at every unique '
                   'intersection',
                   'Keep all faces inside the frame and avoid self-intersecting face loops or overlapping '
                   'duplicate fills']},
 {'family': 'rows',
  'name': 'lissajous-weave',
  'skills': ['curve intersections', 'over-under ordering', 'interval clipping'],
  'instructions': ['Draw two thick closed Lissajous ribbons, one with frequency pair (3,4) and the other '
                   'with pair (4,5)',
                   'Use different phase offsets and sample each centerline at 720 positions',
                   'Find approximate crossings between nonadjacent polyline segments and merge repeated '
                   'detections of the same crossing',
                   'Alternate over-under ordering at successive crossings to create a woven appearance',
                   'Cut short gaps in underpassing strokes using arc-length intervals rather than drawing '
                   'opaque rectangular patches',
                   'Keep gaps continuous when they cross the periodic start/end seam of either ribbon',
                   'Give one ribbon teal shading and the other violet shading with thin white highlights '
                   'along their visible upper edges',
                   'Add exactly sixteen gold beads equally spaced along the visible part of the outer '
                   'ribbon']},
 {'family': 'circles',
  'name': 'hyperbolic-disk',
  'skills': ['circle inversion', 'geodesic arcs', 'recursive deduplication'],
  'instructions': ['Draw a Poincare-disk-inspired rosette from a central regular hyperbolic heptagon and '
                   'three rings of reflected neighboring cells',
                   'Represent geodesic edges as circular arcs orthogonal to the disk boundary, with a '
                   'straight-line special case through the disk center',
                   'Construct reflected vertices by circle inversion and keep them strictly inside the disk',
                   'Deduplicate cells reached by different reflection sequences using rounded vertex '
                   'coordinates',
                   'Choose arc directions consistently so each cell closes along the shorter interior '
                   'geodesics',
                   'Fill alternate rings in teal, violet, and gold, and draw their boundaries with dark '
                   'strokes',
                   'Scale outline widths with cell size so the outer cells remain legible without filling '
                   'the entire boundary band',
                   'Draw one crisp enclosing circle and leave the exterior background untouched']},
 {'family': 'landscapes',
  'name': 'triangulated-terrain',
  'skills': ['height-field mesh', 'level-set clipping', 'perspective shading'],
  'instructions': ['Draw a three-dimensional island terrain generated from a 40-by-30 sampled sum of three '
                   'radial hills and a sinusoidal ridge',
                   'Triangulate every grid cell with alternating diagonals and project the terrain with a '
                   'perspective camera',
                   'Clip each triangle at sea level, producing separate land and water polygons where '
                   'necessary',
                   'Handle vertices exactly at sea level and faces with zero-area projected pieces',
                   'Shade land faces from their surface normals and elevation, using sand, green, and pale '
                   'stone bands',
                   'Sort all visible faces by camera depth and draw a translucent blue sea plane around the '
                   'island',
                   'Extract and draw four continuous elevation contours across triangle boundaries without '
                   'duplicate segments',
                   'Add a dark projected rectangular base under the terrain and keep the entire island '
                   'inside its footprint']},
 {'family': 'columns',
  'name': 'woven-offset-maze',
  'skills': ['graph traversal', 'offset joins', 'curve stitching'],
  'instructions': ['Draw a twelve-by-eight rectangular maze generated by a seeded depth-first spanning-tree '
                   'traversal',
                   'Remove shared walls consistently and leave exactly one entrance and one exit on opposite '
                   'sides',
                   'Solve the maze using the generated cell graph and draw its route as one continuous '
                   'rounded ribbon',
                   'Construct the route ribbon from offset polylines with round caps and bounded miter joins',
                   'Handle consecutive collinear segments and a route touching the maze boundary without '
                   'division by zero or protruding joins',
                   'Fill the route with a teal-to-gold gradient and draw all remaining maze wall segments '
                   'once in dark violet',
                   'Place one small white dot in every visited solution cell and a larger gold disk at each '
                   'endpoint',
                   'Keep the solution ribbon inside passage boundaries and retain visible separation from '
                   'every maze wall']}]

def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()

def generate_repair_prompts(seed: int = 42, count: int = 128) -> list[dict[str, Any]]:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 or seed > 9_007_199_254_740_991:
        raise ValueError("seed must be a nonnegative safe integer")
    if isinstance(count, bool) or not isinstance(count, int) or not 16 <= count <= 512:
        raise ValueError("repair prompt pool count must be from 16 to 512")
    specs = []
    for i in range(count):
        task = TASKS[i % len(TASKS)]
        variant = i // len(TASKS)
        instructions = [*task["instructions"],
            f"Use {['teal, violet, and gold', 'navy, orange, and cream', 'emerald, magenta, and pale gold', 'blue, coral, and ivory'][variant % 4]} as the overall palette while preserving explicitly requested special colors",
            f"Add exactly {12 + variant * 4} alternating small diamonds and circles in a separate bottom legend band, without text",
            "Use a pale sky-blue background",
            "Leave at least 70 pixels clear along every canvas edge",
            "Keep all requested components visible with no text, labels, or unrequested objects"]
        specs.append({"id": f"repair-{task['name']}-{_sha256(f'{REPAIR_CURRICULUM_VERSION}:{seed}:{i}')[:8]}",
            "family": task["family"], "split": "train",
            "prompt": "Create this computational drawing with the documented ArtCanvas API and Python math/random only. " + ". ".join(instructions) + ".",
            "expectedOutcome": "drawing",
            "seed": int(_sha256(f"{seed}:algorithmic:{i}")[:8], 16),
            "repairTask": {"complexity": "advanced", "skills": list(task["skills"])}})
    return specs
