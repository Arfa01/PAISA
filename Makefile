.PHONY: setup sample psx psx-expanded quality compare api test clean

setup:
	python3 -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt && pip install -e .

sample:
	. .venv/bin/activate && python scripts/run_v0_pipeline.py --provider sample --symbols OGDC HBL MCB --start 2021-01-01 --end 2023-12-31 --train --quality-report

psx:
	. .venv/bin/activate && python scripts/run_v0_pipeline.py --provider psx-dps --symbols OGDC HBL MCB --start 2021-01-01 --train --quality-report

psx-expanded:
	. .venv/bin/activate && python scripts/run_v0_pipeline.py --provider psx-dps --symbols-from-csv data/metadata/stock_universe.csv --max-priority 2 --start 2021-01-01 --train --quality-report

quality:
	. .venv/bin/activate && python scripts/generate_data_quality_report.py

compare:
	. .venv/bin/activate && python scripts/run_model_comparison.py

api:
	. .venv/bin/activate && PYTHONPATH=src python -m uvicorn paisa.api.main:app --reload --host 127.0.0.1 --port 8000

test:
	. .venv/bin/activate && pytest

clean:
	rm -rf data/raw/psx data/processed/*.csv data/processed/*.json models/baseline_* models/model_comparison_* .pytest_cache
