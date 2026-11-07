FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash which file patch diffutils ripgrep build-essential pkg-config libcairo2-dev fonts-dejavu-core && rm -rf /var/lib/apt/lists/*
COPY runtime/requirements.txt /opt/artsy/requirements.txt
RUN pip install --no-cache-dir -r /opt/artsy/requirements.txt
RUN python -c "from importlib.metadata import distribution; from pathlib import Path; Path('/opt/artsy/pycairo-api.pyi').symlink_to(distribution('pycairo').locate_file('cairo/__init__.pyi'))"
COPY runtime/artcanvas.py /opt/artsy/runtime/artcanvas.py
COPY docs/artcanvas.md /opt/artsy/docs/artcanvas.md
ENV PYTHONPATH=/opt/artsy/runtime
RUN python -c "from artcanvas import ArtCanvas; assert ArtCanvas.__name__ == 'ArtCanvas'"
COPY evaluation/render.py /opt/artsy/render.py
COPY evaluation/agent_commands.py /opt/artsy/agent_commands.py
COPY drawing/mini_protocol.py /opt/artsy/mini_protocol.py
RUN chmod 755 /opt/artsy/agent_commands.py && ln -s /opt/artsy/agent_commands.py /usr/local/bin/artsy-preview && ln -s /opt/artsy/agent_commands.py /usr/local/bin/artsy-submit
COPY drawing/render_contract.py /opt/artsy/render_contract.py
COPY training/score_rl.py /opt/artsy/score_rl.py
WORKDIR /workspace

RUN useradd --create-home --uid 1000 artsy && chown artsy:artsy /workspace
USER artsy
