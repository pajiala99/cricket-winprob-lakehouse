PY ?= .venv/bin/python
REAL := CRICKLAKE_CONFIG=conf/real.yaml
AIRFLOW_ENV := AIRFLOW_HOME=$(CURDIR)/.airflow AIRFLOW__CORE__DAGS_FOLDER=$(CURDIR)/orchestration/airflow/dags \
	AIRFLOW__CORE__LOAD_EXAMPLES=False CRICKLAKE_HOME=$(CURDIR)

.PHONY: setup pipeline incremental live-score live-feed dashboard test lint clean \
	real-download real-pipeline real-dashboard real-live-feed real-live-score airflow-setup airflow-test

setup:            ## create the virtualenv and install the project
	uv venv --python 3.11 .venv
	uv pip install --python $(PY) -e ".[dev]"

pipeline:         ## clean full build on simulated data: simulate -> bronze -> silver -> gold -> model -> impact
	$(PY) -m cricklake.orchestrate --fresh

incremental:      ## ingest the next data drop (new matches + revisions) with the current model
	$(PY) -m cricklake.orchestrate --incremental

live-score:       ## terminal 1: Structured Streaming scorer
	$(PY) -m cricklake.streaming.live_scoring

live-feed:        ## terminal 2: publish a brand-new simulated match ball by ball
	$(PY) -m cricklake.streaming.live_feed --interval 0.5

dashboard:        ## Streamlit dashboard on http://localhost:8501
	$(PY) -m streamlit run dashboard/app.py

real-download:    ## real Cricsheet T20 data: IPL, T20Is, BBL, WBBL, PSL, CPL, SA20, ILT20, MLC
	$(REAL) $(PY) -m cricklake.ingest.download_real

real-pipeline:    ## build the real-data lakehouse under data/real/
	$(REAL) $(PY) -m cricklake.orchestrate --fresh --skip simulate validate_wpa --pattern "cricsheet/*.zip"

real-dashboard:
	$(REAL) $(PY) -m streamlit run dashboard/app.py

real-live-score:  ## terminal 1: live scorer on the real-data lakehouse
	$(REAL) $(PY) -m cricklake.streaming.live_scoring

real-live-feed:   ## terminal 2: replay the latest real final ball by ball
	$(REAL) $(PY) -m cricklake.streaming.live_feed --replay latest-final --interval 0.5

airflow-setup:    ## separate Airflow 3 environment (pinned by Airflow's official constraints)
	uv venv --python 3.11 .venv-airflow
	uv pip install --python .venv-airflow/bin/python "apache-airflow==3.3.1" apache-airflow-providers-standard \
		-c https://raw.githubusercontent.com/apache/airflow/constraints-3.3.1/constraints-3.11.txt
	$(AIRFLOW_ENV) .venv-airflow/bin/airflow db migrate

airflow-test:     ## run the daily DAG once, end to end, without a scheduler
	$(AIRFLOW_ENV) .venv-airflow/bin/airflow dags test cricket_lakehouse_daily

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests dashboard orchestration

clean:
	rm -rf data/lakehouse data/checkpoints data/landing/packed data/landing/rejected data/models data/stream
