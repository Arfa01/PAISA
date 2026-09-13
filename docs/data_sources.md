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

`pypsx-toolkit` is included as an optional provider for full OHLCV. Use it only if it installs and runs successfully in your environment.

## Manual CSV fallback

For FYP continuity, keep a fallback path where one CSV per symbol can be manually placed in `data/external/manual_psx/`.
