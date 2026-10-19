# Data sources for PAISA v0

## v0 source choice

The v0 implementation is PSX-first.

Primary provider in code:

```text
psx-dps
```

Endpoint:

```text
https://dps.psx.com.pk/timeseries/eod/{SYMBOL}
```

Observed row shape:

```text
[unix_timestamp, close_price, volume, open_price]
```

This source is enough for the first baseline because PAISA can create daily returns, lag features, moving averages, volatility, RSI, MACD, and next-day labels from open, close, and volume.

## Limitation

The PSX DPS time-series endpoint may not provide high/low. v0 stores high/low as missing when unavailable rather than fabricating data.

## Optional provider

`pypsx-toolkit` is an optional provider for full OHLCV and is deliberately not
part of the default requirements. Install and verify it separately:

```bash
pip install -r requirements-provider-pypsx.txt
python -c "import pypsx_toolkit"
```

Use it only if both commands succeed. Its upstream package can report a
successful installation but fail to import on some platforms. This does not
block the primary `psx-dps`, `manual-csv`, or `sample` workflows.

## Manual CSV fallback

For FYP continuity, keep a fallback path where one CSV per symbol can be manually placed in `data/external/manual_psx/`.
