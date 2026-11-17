# GTA VI launch-day Twitch forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/samuelwhitaker96/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/samuelwhitaker96/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Entry for make.fun's [`gta6-twitch-ccu-forecast-01` bounty](https://make.fun/bounty/gta6-twitch-ccu-forecast-01): forecast how many
people watch Grand Theft Auto VI live on Twitch in each hour of its release
day, 2026-11-19 (UTC), after the first.

## What it does

`main.py` (Python 3.13, standard library only, no network) reads every
`<id>.in` case file in the input directory (`$MAKEFUN_INPUT_DIR`, else
`/input`) and writes one `<id>.out` forecast per case to the output directory
(`$MAKEFUN_OUTPUT_DIR`, else `/output`): 23 rows, hours 2–24, each five
quantiles (0.05, 0.25, 0.5, 0.75, 0.95) of that hour's concurrent viewers.

## Model

The forecast is anchored on hour 1's print and built from the case's
comparables — past big game launches' hourly Twitch counts — expressed as
log-ratios to their own hour-1 print, `log(v[h] / v[1])`. Two pools are
blended in log space:

- **(A) all comparables** (except the target's own pre-release history):
  weighted by tier (comparable 1.0, early-access contaminated 0.25, weak
  0.15), by unlock type (non-midnight-local launches ×0.35), and by scale —
  a Gaussian kernel on log hour-1 viewers (σ 0.8), so launches of a similar
  size count most. Big launches ramp less steeply from hour 1 than small
  ones, and the kernel captures that.
- **(B) rolling midnight-local, console-only launches** — the shape GTA VI
  itself has (PS5/Xbox Series, midnight local rolling, no PC at launch) —
  with a wider scale kernel (σ 1.5). This keeps the forecast ramping into
  the evening even when hour 1 is enormous, where pool (A) alone would be
  dominated by PC front-loaded launches whose hour 1 already holds most of
  the world.

Per hour, each pool gives a weighted median and a spread (IQR / 1.349). The
blended median is the mean of the two medians; the blended spread is the
larger of the two spreads and half the distance between the medians, with a
floor growing like √(`h` − 1) and a cap of 1.0 log units. Quantiles are
`p1 · exp(median + z_τ · spread)`.

The shape this recovers is the unlock geography: hour 1 holds the UK /
Europe / Oceania midnight audience; the US unlocks at 05:00/08:00 UTC
(hours 6 and 9); the peak lands in hours 17–22, when the US afternoon
overlaps the European evening.

## Validation

- **Practice cases** (the kit's four past launches, scored against their
  real SullyGnome hourly counts): mean pinball loss **0.202** on the log
  scale, against **0.387** for the house baseline (flat at hour 1's print)
  — Wolverine 0.357, Red Dead Redemption 2 0.123, God of War Ragnarök
  0.149, Spider-Man 2 0.182. 90% coverage 0.87.
- **Synthetic final cases** (hour-1 prints from 100 to 5,000,000 viewers,
  plus no-comparables and no-known-hour-1 fallbacks): all outputs valid
  (23 rows, monotone quantiles, every value in (0, 10⁹], files < 2 KB),
  byte-identical across two runs, ~0.03 s per case.

## Run locally

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```
