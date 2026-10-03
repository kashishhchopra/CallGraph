PY ?= .venv/bin/python
.PHONY: setup download manifests preprocess process train-intent train-emotion train-summary eval report serve test lint up

setup:
	python3.11 -m venv .venv && .venv/bin/pip install -e ".[dev]" && $(PY) -m spacy download en_core_web_sm
	docker compose up -d falkordb
	ollama pull llama3.2:3b

download:
	$(PY) scripts/dataset_cli.py download

manifests:
	$(PY) scripts/dataset_cli.py manifests

preprocess:
	$(PY) scripts/preprocess_all.py

process:
	$(PY) scripts/process.py unidata
	$(PY) scripts/process.py axondata
	$(PY) scripts/process.py spokenwoz

train-intent:
	$(PY) scripts/train_intent.py all

train-emotion:
	$(PY) scripts/train_emotion.py all && $(PY) scripts/tune_fusion.py

train-summary:
	$(PY) scripts/train_summary.py

eval:
	$(PY) -m eval.run && $(PY) -m eval.report

report:
	$(PY) -m eval.report

serve:
	docker compose up -d falkordb
	PYTHONPATH=. $(PY) -m uvicorn api.main:app --host 0.0.0.0 --port 8010

up:
	docker compose up --build

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src api eval scripts tests && $(PY) -m black --check src api eval scripts tests
