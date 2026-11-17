#!/usr/bin/env bash
# Run as root in a Python 3.14 Vercel Sandbox after copying runtime.tar.gz.
set -euo pipefail
apt-get update -qq
apt-get install -y -qq --no-install-recommends build-essential pkg-config libcairo2-dev fonts-dejavu-core ripgrep
rm -rf /var/lib/apt/lists/*
mkdir -p /opt/artsy/app /workspace /vercel/sandbox/job
tar -xzf /vercel/sandbox/runtime.tar.gz -C /opt/artsy/app
cd /opt/artsy/app
python3 -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torchvision==0.29.1
python3 -m pip install --no-cache-dir -r runtime/requirements.txt -r inference/requirements-tinker.txt -r drawing/requirements.txt
cp evaluation/render.py evaluation/agent_commands.py drawing/mini_protocol.py drawing/render_contract.py /opt/artsy/
cp client/sandbox/agent.py client/sandbox/action.py client/sandbox/snapshot.py /opt/artsy/app/
chmod +x /opt/artsy/agent_commands.py
ln -sf /opt/artsy/agent_commands.py /usr/local/bin/artsy-preview
ln -sf /opt/artsy/agent_commands.py /usr/local/bin/artsy-submit
ln -sf /usr/local/bin/python3 /usr/local/bin/python
chmod 777 /workspace /vercel/sandbox/job
PYTHONPATH=/opt/artsy/app:/opt/artsy/app/runtime MSWEA_GLOBAL_CONFIG_DIR=/tmp/mini-swe-agent python3 - <<'PY'
from pathlib import Path
from drawing.mini_agent import run_agent
from inference.tinker_backend import TinkerBackend
from evaluation.render import render
result = render('from artcanvas import ArtCanvas\nwith ArtCanvas(1920,1200,"output.png"):\n    pass\n', 42, Path('/tmp/runtime-smoke'))
assert result['ok'], result
print('Remote runtime imports and Cairo rendering passed')
PY
rm -rf /tmp/mini-swe-agent /tmp/runtime-smoke
