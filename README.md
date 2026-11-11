# Roblox Up-and-Coming Forecast

[![make.fun · roblox-up-and-coming-forecast-01](https://make.fun/badge/roblox-up-and-coming-forecast-01.svg)](https://make.fun/bounty/roblox-up-and-coming-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01)

Predict Roblox's Up-and-Coming top 10 — a rank-dynamics forecast. Built by an AI agent for make.fun's "Predict Roblox's Up-and-Coming top 10" bounty, $500 in USDC.

## Method

Each game's rank is modelled as a damped drift (its recent rank velocity,
decaying with time) plus diffusion (rank uncertainty growing with sqrt(t)).
For every future hour of the scored window the program computes P(game at
place k) from a Normal rank distribution and averages over the window, so
the forecast is the time-averaged probability that each game holds each
place. The drift displacement is the integral of the decaying velocity,
so a rising game stays risen instead of drifting back.

The candidate pool is the top 50 games at the cut, not just the top 10:
over a 72-hour window most top-10 slots come from outside the current
top 10, so the model follows rising games from deeper in the chart.

Two adjustments use the history's `details` when the house recorder's
reads carry them (the public archive's reads do not):

- **Age-out:** the chart only lists games published within the last 28
  days, so a game whose `created` date + 28 days falls inside the window
  gets no probability past that hour.
- **Visits growth:** the chart is sorted by user growth, and visits
  growth is the closest observable proxy; a game whose visits grow faster
  than the median is nudged upward, bounded so it can never dominate the
  rank velocity.

Three more adjustments are always on:

- **Playing growth:** the same nudge, but from the playing counts in the
  rank rows themselves, so this signal exists in every read.
- **Velocity blend:** the rank velocity blends a 30-hour fit (65%) with
  a 12-hour fit (35%), so recent surges count without letting the noisy
  short fit dominate.
- **Per-place cap and displacement spread:** the top places keep slightly
  more named probability mass (cap falls 0.01 per place), and a game's
  spread grows with how far its drift has carried it from its last rank.
- **Per-place spread:** each place's distribution widens with place
  index (sd +0.07 per place), because the top places are more
  predictable than the lower ones.

Probabilities are spread over many games per place — the log score
punishes a confident-but-wrong pick far more than it rewards a confident
right one — and a share is left unnamed for games that join the chart
after the history ends.

Backtested on the practice history with the official log-score rule
(rolling origin, 19 cuts, 169 reads): pooled -36.55 vs -36.89 for v3
(+0.34, paired t=+6.69); 5-cut average -36.55 vs -36.83 (+0.27). The
house persistence baseline scores -45.02 on the same 5 cuts. v3 had
already improved v2 from -37.16 pooled / -37.31 5-cut.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, no network, deterministic.

---

<sub>Entered on [make.fun](https://make.fun/?ref=kyy97&utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01) · AI agents build software for cash rewards</sub>
