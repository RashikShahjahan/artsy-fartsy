PYTHON ?= .venv/bin/python
EVAL_PYTHON ?= $(if $(wildcard .eval-venv/bin/python),.eval-venv/bin/python,$(PYTHON))
ARGS ?=

.PHONY: ui serve render draw eval eval-base dataset grpo test test-eval check
ui:
	$(PYTHON) -m ui.server $(ARGS)

serve:
	$(PYTHON) -m uvicorn inference.server:app --host 127.0.0.1 --port 8080 $(ARGS)
render:
	$(PYTHON) -m drawing.run $(ARGS)

draw:
	$(EVAL_PYTHON) -m evaluation.draw $(ARGS)

eval eval-base:
	$(EVAL_PYTHON) -m evaluation.evaluate $(ARGS)

dataset:
	$(PYTHON) -m drawing.generate_dataset $(ARGS)

grpo:
	$(PYTHON) -m training.grpo $(ARGS)

test:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py' -v $(ARGS)

test-eval:
	$(EVAL_PYTHON) -m unittest discover -s evaluation -p 'test_*.py' -v $(ARGS)

check:
	$(PYTHON) -m compileall -q drawing inference evaluation training tests $(ARGS)
