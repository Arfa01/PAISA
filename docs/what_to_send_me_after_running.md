# What to send back if something fails

If PSX data access fails on your Mac, send these exact things:

1. The full command you ran.
2. The full terminal error text.
3. Your Python version:

```bash
python --version
```

4. Package install result:

```bash
pip show pypsx-toolkit
```

5. Whether this command works:

```bash
python scripts/run_v0_pipeline.py --provider sample --symbols OGDC HBL MCB --start 2021-01-01 --end 2023-12-31 --train
```

If the sample command works but PSX fails, the code structure is fine and only the data-source adapter needs adjustment.
