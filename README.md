# Roblox Up-and-Coming Forecast

[![Bounty](https://img.shields.io/badge/bounty-$500%20USDC-blue)](https://make.fun/bounties/roblox-up-and-coming-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01)

This program forecasts the top ten of the Roblox Up-and-Coming chart. The bounty pays **$500 USDC**.

## Method

**Rank dynamics.** Each game's rank follows a damped drift plus diffusion. The drift comes from the game's recent rank velocity, fading exponentially with a 12-hour half-life. The diffusion is rank uncertainty that widens with the square root of time. For every future grid time in the scored window, the program computes, for each game and each place, the probability that the game holds that place under a Normal rank distribution, then averages over the grid times.

**Candidate pool.** The top 50 games at the cut are used as candidates, not just the top 10, because most top-ten slots over 72 hours come from outside the current top ten.

**Adjustments.**
- **Age-out:** games exit the chart 28 days after creation.
- **Visits growth nudge:** velocity is adjusted by the deviation of a game's visits growth from the median visits growth across candidates, scaled by 20.0.
- **Playing growth nudge:** same idea on the playing counts from the rank rows, scaled by 12.5.
- **Velocity blend:** 65% from a 30-hour linear fit, 35% from a 12-hour fit, clamped to ±2.0 ranks/hour.
- **Per-place cap:** 0.80 at place 1, falling by 0.01 per place, clamped to [0.30, 1.00].
- **Displacement spread:** extra spread of 0.05 per rank of absolute displacement.
- **Per-place spread:** extra 0.13 per place index.
- **Coarse time grid:** averaging every 14 hours across the window.
- **Uniform velocity weights:** all points in the 30-hour window weighted equally.

**Probability spreading.** Probabilities are spread over many games per place because the log score punishes a confident wrong pick far more than it rewards a confident right one. A share of probability mass is left unnamed to cover games that enter the chart after the recorded history runs out.

## Backtest

- Rolling pooled: **−36.38** vs −36.55 for the previous version
- 5-cut average: **−36.23** vs −36.55
- 13 of 19 cuts improving
- House persistence baseline on the same 5 cuts: **−45.02**

## Run

```bash
mkdir -p output
MAKEFUN_INPUT_DIR=/input MAKEFUN_OUTPUT_DIR=/output python3 main.py
```

Runs on Python 3.13, uses only the standard library, needs no network, and is deterministic.

---

Entered on [make.fun](https://make.fun?ref=kyy97&utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01).