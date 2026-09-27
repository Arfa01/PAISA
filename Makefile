.PHONY: setup sample psx train-ogdc-rf api test clean

setup:
	python3 -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt && pip install -e .

sample:
	. .venv/bin/activate && python scripts/run_v0_pipeline.py --provider sample --symbols OGDC HBL MCB --start 2021-01-01 --end 2023-12-31 --train

psx:
	. .venv/bin/activate && python scripts/run_v0_pipeline.py --provider psx-dps --symbols OGDC HBL MCB --start 2021-01-01 --train

train-ogdc-rf:
	. .venv/bin/activate && python scripts/train_ogdc_rf_variants.py --symbol OGDC --start 2021-01-01 --provider psx-dps --output-dir artifacts/model_training --random-state 42

api:
	. .venv/bin/activate && uvicorn paisa.api.main:app --reload --host 127.0.0.1 --port 8000

test:
	. .venv/bin/activate && pytest

clean:
	rm -rf data/raw/psx data/processed/*.csv data/processed/*.json models/baseline_* .pytest_cache
