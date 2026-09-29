.PHONY: setup sample psx psx-expanded quality compare train-ogdc-rf api test clean

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

train-ogdc-rf:
	. .venv/bin/activate && python scripts/train_ogdc_rf_variants.py --symbol OGDC --start 2021-01-01 --provider psx-dps --output-dir artifacts/model_training --random-state 42

api:
	. .venv/bin/activate && python scripts/serve_api.py

test:
	. .venv/bin/activate && pytest

clean:
	rm -rf data/raw/psx data/processed/*.csv data/processed/*.json models/baseline_* models/model_comparison_* artifacts/model_training artifacts/paisa_TRN-*.zip .pytest_cache
