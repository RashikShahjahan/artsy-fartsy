# Build from repository root. Both VMs use this image; only the agent receives credentials.
FROM python:3.14-slim
RUN apt-get update && apt-get install -y --no-install-recommends bash sudo build-essential pkg-config libcairo2-dev fonts-dejavu-core ripgrep && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/artsy/app
COPY runtime/requirements.txt runtime/requirements.txt
COPY inference/requirements-tinker.txt inference/requirements-tinker.txt
COPY drawing/requirements.txt drawing/requirements.txt
# Inference is hosted by Tinker; CPU wheels avoid shipping unused CUDA libraries.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torchvision==0.29.1
RUN pip install --no-cache-dir -r runtime/requirements.txt -r inference/requirements-tinker.txt -r drawing/requirements.txt
COPY drawing drawing
COPY inference inference
COPY runtime runtime
COPY docs/artcanvas.md docs/artcanvas.md
COPY evaluation/render.py /opt/artsy/render.py
COPY evaluation/agent_commands.py /opt/artsy/agent_commands.py
COPY drawing/mini_protocol.py /opt/artsy/mini_protocol.py
COPY drawing/render_contract.py /opt/artsy/render_contract.py
COPY evaluation/__init__.py evaluation/__init__.py
COPY evaluation/render.py evaluation/render.py
COPY client/sandbox/agent.py /opt/artsy/app/agent.py
COPY client/sandbox/action.py /opt/artsy/app/action.py
COPY client/sandbox/snapshot.py /opt/artsy/app/snapshot.py
RUN chmod +x /opt/artsy/agent_commands.py && ln -s /opt/artsy/agent_commands.py /usr/local/bin/artsy-preview && ln -s /opt/artsy/agent_commands.py /usr/local/bin/artsy-submit
ENV PYTHONPATH=/opt/artsy/app:/opt/artsy/app/runtime
ENV MSWEA_GLOBAL_CONFIG_DIR=/tmp/mini-swe-agent
RUN mkdir -p /workspace /vercel/sandbox/job && chmod 777 /workspace /vercel/sandbox/job
RUN useradd --create-home --shell /bin/bash ubuntu && echo 'ubuntu ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/ubuntu
USER ubuntu
WORKDIR /workspace
