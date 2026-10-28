PYTHON ?= .venv/bin/python
EVAL_PYTHON ?= .eval-venv/bin/python
ARGS ?=

.PHONY: serve render draw eval eval-base dataset dpo grpo test test-eval check
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

dpo:
	$(PYTHON) -m training.dpo $(ARGS)

grpo:
	$(PYTHON) -m training.grpo $(ARGS)

test:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py' -v $(ARGS)

test-eval:
	$(EVAL_PYTHON) -m unittest discover -s evaluation -p 'test_*.py' -v $(ARGS)

check:
	$(PYTHON) -m compileall -q drawing inference evaluation training tests $(ARGS)
