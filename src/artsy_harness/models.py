"""Model calls return evidence; the local Python runner executes programs."""

import json
from dataclasses import dataclass
from urllib.error import HTTPError
from urllib.request import Request, urlopen

DEFAULT_MODEL = "qwen3.5:9b-q4_K_M"
PROMPT_VERSION = "artcanvas-v2"
PROMPT_TEMPLATE = """Write a Python program that draws the requested image using ArtCanvas.
Return only Python source, without Markdown fences or explanations.
Import ArtCanvas from artcanvas. Use a 256 by 256 canvas and save output.png in the current directory.
Available methods: set_color(r,g,b,a=1), set_line_width(width),
fill_background(r,g,b), draw_circle(x,y,radius,fill=False),
draw_rectangle(x,y,width,height,fill=False), draw_polygon(points,fill=False),
move_brush_to(x,y), draw_line_to(x,y), draw_arc(xc,yc,radius,start_angle,end_angle),
draw_text(x,y,text,font_size=16,font_family="Sans"), save().
Coordinates are pixels, colors are between 0 and 1, angles are radians.
Use `with ArtCanvas(256, 256, "output.png") as canvas:` to save on exit.

Drawing request:
"""


@dataclass
class ModelResponse:
    raw: bytes
    http_status: int
    text: str | None = None
    model: str | None = None
    digest: str | None = None
    error: str | None = None


class Ollama:
    provider = "ollama"

    def __init__(self, *, base_url="http://localhost:11434", model=DEFAULT_MODEL,
                 options=None, timeout=600, think=True):
        self.endpoint = base_url.rstrip("/") + "/api/generate"
        self.model = model
        self.options = dict(options) if options is not None else {
            "seed": 0, "temperature": 0.2, "top_p": 0.9,
            "num_predict": 8192, "num_ctx": 16384,
        }
        self.timeout = timeout
        self.think = think

    def request(self, prompt):
        return {"model": self.model, "prompt": prompt, "stream": False,
                "options": self.options, "think": self.think}

    def generate(self, payload):
        request = Request(self.endpoint, data=json.dumps(payload).encode("utf-8"),
                          headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = ModelResponse(response.read(), response.status)
        except HTTPError as error:
            with error:
                return ModelResponse(error.read(), error.code, error=str(error))
        try:
            body = json.loads(result.raw)
            if not isinstance(body, dict):
                raise ValueError("Expected a JSON object")
            result.model = body.get("model")
            result.digest = body.get("digest") or body.get("model_digest")
            if body.get("error"):
                raise ValueError(str(body["error"]))
            if body.get("done") is not True or not isinstance(body.get("response"), str):
                raise ValueError("Expected a completed, non-streaming response with text")
            result.text = body["response"]
        except (ValueError, UnicodeError) as error:
            result.error = str(error)
        return result
