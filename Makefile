PYTHON ?= .venv/bin/python
EVAL_PYTHON ?= $(if $(wildcard .eval-venv/bin/python),.eval-venv/bin/python,$(PYTHON))
ARGS ?=

.PHONY: client client-build client-check client-tts-setup ui tts-setup serve render draw eval eval-base agent-image dataset rlvr rlhf grpo test test-ui test-eval test-js requirements-sync requirements-check syntax check test-all
client:
	npm --prefix client run dev -- $(ARGS)

client-build:
	npm --prefix client run build

client-check:
	npm --prefix client run typecheck
	npm --prefix client test
	$(PYTHON) -m unittest client.sandbox.test_agent -v
	$(PYTHON) -m unittest discover -s client/backend -t . -p 'test_*.py' -v

client-tts-setup:
	UV_CACHE_DIR="$(CURDIR)/.cache/uv" uv venv --allow-existing --python 3.12 .tts-venv
	UV_CACHE_DIR="$(CURDIR)/.cache/uv" uv pip install --python .tts-venv/bin/python kokoro-onnx==0.4.9
	$(PYTHON) -m client.backend.setup_tts

ui:
	$(PYTHON) -m ui.start $(ARGS)

tts-setup:
	UV_CACHE_DIR="$(CURDIR)/.cache/uv" uv venv --allow-existing --python 3.12 .tts-venv
	UV_CACHE_DIR="$(CURDIR)/.cache/uv" uv pip install --python .tts-venv/bin/python -r ui/requirements-tts.txt
	$(PYTHON) -m ui.setup_tts

serve:
	$(PYTHON) -m uvicorn inference.server:app --host 127.0.0.1 --port 8080 $(ARGS)
render:
	$(PYTHON) -m drawing.run $(ARGS)

draw:
	$(EVAL_PYTHON) -m evaluation.draw $(ARGS)

eval eval-base:
	$(EVAL_PYTHON) -m evaluation.evaluate $(ARGS)

agent-image:
	docker build -f evaluation/Dockerfile -t artsy-editing-agent:latest .
	$(PYTHON) -c 'from drawing.mini_dataset import inspect_image; print(inspect_image())'

dataset:
	$(PYTHON) -m drawing.generate_dataset $(ARGS)

rlvr:
	$(PYTHON) -m training.rlvr $(ARGS)

rlhf:
	$(PYTHON) -m training.rlhf $(ARGS)

grpo:
	$(PYTHON) -m training.grpo $(ARGS)

test:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py' -v $(ARGS)

test-eval:
	$(EVAL_PYTHON) -m unittest discover -s evaluation -p 'test_*.py' -v $(ARGS)

test-ui:
	$(PYTHON) -m unittest discover -s ui -p 'test_*.py' -v $(ARGS)

test-js:
	node --test ui/test_*.cjs

requirements-sync:
	$(PYTHON) scripts/sync_requirements.py

requirements-check:
	$(PYTHON) scripts/sync_requirements.py --check

syntax:
	$(PYTHON) -m compileall -q drawing inference evaluation runtime training ui client/backend tests scripts
	@for file in ui/*.js ui/*.cjs; do node --check "$$file" || exit; done

check: requirements-check syntax test test-ui test-eval test-js

test-all: check
