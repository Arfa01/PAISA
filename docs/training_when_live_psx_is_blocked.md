# Training when live PSX providers are blocked

The default OGDC training command fetches daily data through the existing PAISA
pipeline:

```bash
python scripts/train_ogdc_models.py
```

The PSX Data Portal endpoint is public but undocumented and can temporarily
return `403`, `404`, or a network error. A failed real-data acquisition stops
the run. The model trainer never silently substitutes sample or synthetic rows.

All outputs are academic experiments and are not financial advice.

## Use a previously prepared real-data file

If a prior successful real-provider or manual-CSV run created the standard
cleaned file, train from that immutable input instead of making another network
request:

```text
data/processed/psx_prices.csv
```

Run:

```bash
python scripts/train_ogdc_models.py \
  --prices data/processed/psx_prices.csv \
  --symbol OGDC \
  --start 2021-01-01 \
  --output-dir artifacts/model_training \
  --random-state 42
```

The trainer selects OGDC rows, validates the standard daily price schema,
records the supplied source provenance, rebuilds the reusable technical
features and next-session target, and stores an input fingerprint in the run
manifest. Inspect the CSV's `source` values before training: the current trainer
records but does not automatically reject a source string containing `sample`
or `synthetic`.

Choose a new `--run-id` if supplying one explicitly. Reusing a run or model
destination is an error; do not delete an earlier run to replace it.

## Preparing the file from manual real PSX data

Place the real downloaded CSV at:

```text
data/external/manual_psx/OGDC.csv
```

Then use the existing v0 pipeline to standardize it:

```bash
python scripts/run_v0_pipeline.py \
  --provider manual-csv \
  --manual-csv-dir data/external/manual_psx \
  --symbols OGDC \
  --start 2021-01-01
```

After `data/processed/psx_prices.csv` has been inspected, run the unified
training command with `--prices` as shown above. Do not commit the processed
file, frozen input, model binaries, results, or ZIP archive.

## Optional provider

`pypsx-toolkit` remains isolated from the default environment because its
upstream package can install but fail to import on some platforms. To try it:

```bash
python -m pip install -r requirements-provider-pypsx.txt
python -c "import pypsx_toolkit; print(pypsx_toolkit.__file__)"
python scripts/train_ogdc_models.py --provider pypsx
```

A failure of this optional provider does not invalidate the default CPU
environment or offline test suite.

## Verify the result

After training:

```bash
pytest
```

Inspect the unique directory printed by the CLI. Its per-model test prediction
files are contract-compatible with `scripts/run_backtest.py`; the backtest must
use an experiment configuration with the same `model_id` and evaluation dates.
