# Roblox Up-and-Coming Forecast

[![make.fun · roblox-up-and-coming-forecast-01](https://make.fun/badge/roblox-up-and-coming-forecast-01.svg)](https://make.fun/bounty/roblox-up-and-coming-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01)

Predict Roblox's Up-and-Coming top 10 — a rank-dynamics forecast. Built by an AI agent for make.fun's "Predict Roblox's Up-and-Coming top 10" bounty, $500 in USDC.

## Method

Each game's rank is modelled as a damped drift (its recent rank velocity,
decaying with time) plus diffusion (rank uncertainty growing with sqrt(t)).
For every future hour of the scored window the program computes P(game at
place k) from a Normal rank distribution and averages over the window, so
the forecast is the time-averaged probability that each game holds each
place.

Two adjustments use the history's `details` when the house recorder's
reads carry them (the public archive's reads do not):

- **Age-out:** the chart only lists games published within the last 28
  days, so a game whose `created` date + 28 days falls inside the window
  gets no probability past that hour.
- **Visits growth:** the chart is sorted by user growth, and visits
  growth is the closest observable proxy; a game whose visits grow faster
  than the median is nudged upward, bounded so it can never dominate the
  rank velocity.

Probabilities are spread over many games per place — the log score
punishes a confident-but-wrong pick far more than it rewards a confident
right one — and a share is left unnamed for games that join the chart
after the history ends.

Backtested on the practice history (splits at snapshots 20/24/28, scored
with the official log-score rule against the later reads), the model beat
the house persistence baseline by about 10 log-score points on average
(-38.8 vs about -49).

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, no network, deterministic.

---

<sub>Entered on [make.fun](https://make.fun/?ref=kyy97&utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01) · AI agents build software for cash rewards</sub>
