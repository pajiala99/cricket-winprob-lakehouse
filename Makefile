PY ?= .venv/bin/python

.PHONY: setup pipeline incremental live-score live-feed dashboard test lint clean

setup:            ## create the virtualenv and install the project
	uv venv --python 3.11 .venv
	uv pip install --python $(PY) -e ".[dev]"

pipeline:         ## clean full build: simulate -> bronze -> silver -> gold -> model -> impact
	$(PY) -m cricklake.orchestrate --fresh

incremental:      ## ingest the next data drop (new matches + revisions) with the current model
	$(PY) -m cricklake.orchestrate --incremental

live-score:       ## terminal 1: Structured Streaming scorer
	$(PY) -m cricklake.streaming.live_scoring

live-feed:        ## terminal 2: publish a brand-new match ball by ball
	$(PY) -m cricklake.streaming.live_feed --interval 0.5

dashboard:        ## Streamlit dashboard on http://localhost:8501
	$(PY) -m streamlit run dashboard/app.py

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests dashboard orchestration

clean:
	rm -rf data/lakehouse data/checkpoints data/landing/packed data/landing/rejected data/models data/stream
