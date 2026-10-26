FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash which libcairo2 fonts-dejavu-core && rm -rf /var/lib/apt/lists/*
COPY runtime/requirements.txt /opt/artsy/requirements.txt
RUN pip install --no-cache-dir -r /opt/artsy/requirements.txt Pillow==12.0.0
COPY runtime/ /opt/artsy/runtime/
COPY evaluation/render.py /opt/artsy/render.py
COPY training/score_rl.py /opt/artsy/score_rl.py
RUN python -c "import site; from pathlib import Path; Path(site.getsitepackages()[0], 'artsy.pth').write_text('/opt/artsy/runtime\\n')"
WORKDIR /workspace
