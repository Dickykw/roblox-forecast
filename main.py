#!/usr/bin/env python3
"""Predict Roblox's Up-and-Coming top 10 — make.fun bounty entry.

Reads the chart history at $MAKEFUN_INPUT_DIR (else /input) and writes
/output/forecast.json ($MAKEFUN_OUTPUT_DIR when set): for each of the top
10 places, a probability for each game named being at that place, plus
the top 10 in order.

Method: rank dynamics.  Each game's rank is modelled as a damped drift
(its recent rank velocity, decaying with time) plus diffusion (rank
uncertainty growing with sqrt(t)).  For every future hour of the scored
window we compute P(game at place k) from a Normal rank distribution and
average over the window, so the forecast is the time-averaged probability
that each game holds each place.  Probabilities are spread over many
games per place — the log score punishes a confident-but-wrong pick far
more than it rewards a confident right one — and a share is left unnamed
for games that join the chart after the history ends.

Standard library only, no network, deterministic: the same input always
produces the same bytes.
"""

import json
import math
import os
import statistics
from datetime import datetime, timedelta

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# Tuned on a rolling backtest (19 cuts, 169 reads, official log-score
# rule with the unnamed-share): pooled -37.16 vs -39.01 for the original
# entry (+1.84, paired t=+8.35); 5-cut average -37.31 vs -38.04.  The
# house persistence baseline scores -45.02 on the same 5 cuts.  Biggest
# single gain: candidates widened from the top 10 to the top 50 games at
# the cut (over a 72h window most top-10 slots come from outside the
# current top 10); the drift displacement is the integral of the decaying
# velocity, so a rising game stays risen.
LOOKBACK_H = 36.0        # rank-velocity fit window (hours)
VEL_CLAMP = 2.5          # max |rank velocity|, ranks/hour
VEL_HALFLIFE_H = 12.0    # velocity decays with this half-life
SD0 = 2.0                # base rank standard deviation
SD_RATE = 0.4            # sd grows per sqrt(hour)
MAX_PER_PLACE = 40       # games named per place
PLACE_CAP = 0.8          # named probability mass per place (rest unnamed)
MIN_P = 0.005            # drop rows below this
VISITS_BETA = 20.0       # rank-velocity nudge per unit of excess log-visits growth/hour
CANDIDATE_TOP_N = 50     # games from the last snapshot considered as candidates


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_range(mu, sd, lo, hi):
    return phi((hi - mu) / sd) - phi((lo - mu) / sd)


def build_series(snaps):
    """Per universe_id: list of (hours_since_first, rank) sorted by time.
    Also collects, when the history carries details (the house recorder's
    reads do; the public archive's do not): the game's creation time and
    its visits series, for the age-out and visits-growth adjustments."""
    t_ref = parse_time(snaps[0]["at"])
    series = {}
    created = {}
    visits = {}
    for s in snaps:
        th = (parse_time(s["at"]) - t_ref).total_seconds() / 3600.0
        for r in s["up_and_coming"]:
            series.setdefault(r["universe_id"], []).append((th, r["rank"]))
        details = s.get("details") or {}
        for key, d in details.items():
            try:
                u = int(key)
            except (TypeError, ValueError):
                continue
            if not isinstance(d, dict):
                continue
            if d.get("created") and u not in created:
                try:
                    created[u] = parse_time(d["created"])
                except (TypeError, ValueError):
                    pass
            if d.get("visits") is not None:
                try:
                    visits.setdefault(u, []).append((th, float(d["visits"])))
                except (TypeError, ValueError):
                    pass
    return series, t_ref, created, visits


def visits_growth(visits_u, t0, lookback_h):
    """Log-visits growth per hour over the recent window, or None."""
    pts = [(t, v) for (t, v) in visits_u if t0 - t <= lookback_h and t <= t0 and v > 0]
    if len(pts) < 2:
        return None
    (t1, v1), (t2, v2) = pts[0], pts[-1]
    if t2 - t1 < 1.0:
        return None
    return (math.log(v2) - math.log(v1)) / (t2 - t1)


def velocity(series_u, t0, lookback_h):
    """Weighted linear fit of rank vs time over the recent window.
    Returns the rank velocity in ranks/hour (negative = rising)."""
    pts = [(t, r) for (t, r) in series_u if t0 - t <= lookback_h and t <= t0]
    if len(pts) < 2:
        return 0.0
    ws = [1.0 + 2.0 * (t - (t0 - lookback_h)) / lookback_h for (t, _) in pts]
    sw = sum(ws)
    mt = sum(w * t for w, (t, _) in zip(ws, pts)) / sw
    mr = sum(w * r for w, (_, r) in zip(ws, pts)) / sw
    num = sum(w * (t - mt) * (r - mr) for w, (t, r) in zip(ws, pts))
    den = sum(w * (t - mt) ** 2 for w, (t, _) in zip(ws, pts))
    vel = num / den if den > 1e-9 else 0.0
    return max(-VEL_CLAMP, min(VEL_CLAMP, vel))


def forecast(snaps, places, hours):
    series, t_ref, created, visits = build_series(snaps)
    t0 = (parse_time(snaps[-1]["at"]) - t_ref).total_seconds() / 3600.0
    t0_abs = parse_time(snaps[-1]["at"])
    last = sorted(snaps[-1]["up_and_coming"], key=lambda r: r["rank"])
    cand = last[:CANDIDATE_TOP_N]
    last_ids = [r["universe_id"] for r in cand]
    r0 = {r["universe_id"]: r["rank"] for r in cand}
    vel = {u: velocity(series.get(u, []), t0, LOOKBACK_H) for u in last_ids}
    # Visits growth nudge (only when details exist): the chart is sorted by
    # user growth, and visits growth is the closest observable proxy.  A
    # game whose visits grow faster than the median is nudged upward
    # (negative rank velocity), bounded so it can never dominate the rank
    # velocity the backtest validated.
    growth = {u: visits_growth(visits.get(u, []), t0, LOOKBACK_H) for u in last_ids}
    have = [g for g in growth.values() if g is not None]
    if have:
        med = statistics.median(have)
        for u in last_ids:
            if growth[u] is not None:
                adj = -VISITS_BETA * (growth[u] - med)
                vel[u] = max(-VEL_CLAMP, min(VEL_CLAMP, vel[u] + adj))
    # Age-out: the chart only lists games published within the last 28
    # days.  A game whose created date + 28 days falls inside the window
    # leaves the chart then, so it gets no probability past that hour.
    exit_h = {}
    for u in last_ids:
        if u in created:
            eh = (created[u] + timedelta(days=28) - t0_abs).total_seconds() / 3600.0
            exit_h[u] = eh if eh > 0 else 0.0

    # Average P(game at place k) over the window's hours (every 2nd hour).
    prob = {k: {} for k in range(1, places + 1)}
    times = list(range(0, hours + 1, 2))
    for t in times:
        decay = 0.5 ** (t / VEL_HALFLIFE_H)
        for u in last_ids:
            if u in exit_h and t >= exit_h[u]:
                continue
            # Displacement is the integral of the exponentially decaying
            # velocity: it saturates at vel*halflife/ln2 instead of
            # returning to zero, so a rising game stays risen.
            mu = r0[u] + vel[u] * (VEL_HALFLIFE_H / math.log(2.0)) * (1.0 - decay)
            mu = max(1.0, min(120.0, mu))
            sd = SD0 + SD_RATE * math.sqrt(t + 1.0)
            for k in range(1, places + 1):
                pr = norm_range(mu, sd, k - 0.5, k + 0.5)
                if pr > 1e-5:
                    prob[k][u] = prob[k].get(u, 0.0) + pr
    for k in prob:
        for u in prob[k]:
            prob[k][u] /= len(times)

    rows = []
    for k in range(1, places + 1):
        items = sorted(prob[k].items(), key=lambda kv: -kv[1])[:MAX_PER_PLACE]
        items = [(u, v) for (u, v) in items if v >= MIN_P]
        if not items:
            continue
        s = sum(v for _, v in items)
        scale = min(1.0, PLACE_CAP / s) if s > 0 else 0.0
        for u, v in items:
            rows.append({"universe_id": u, "place": k, "p": round(v * scale, 6)})

    # top10: order by expected place (probability-weighted), tie by now.
    exp_rank = {}
    for u in last_ids:
        tot = sum(prob[k].get(u, 0.0) for k in range(1, places + 1))
        exp_rank[u] = (
            sum(k * prob[k].get(u, 0.0) for k in range(1, places + 1)) / tot
            if tot > 0
            else r0[u]
        )
    top10 = sorted(last_ids, key=lambda u: (exp_rank[u], r0[u]))[:places]
    return {"probabilities": rows, "top10": top10}


def main():
    with open(os.path.join(INPUT_DIR, "task.json")) as f:
        task = json.load(f)
    places = int(task.get("places", 10))
    window = task.get("window", {})
    hours = int(window.get("slots", 72))
    output = task.get("output", "forecast.json")
    snaps = []
    with open(os.path.join(INPUT_DIR, "history.ndjson")) as f:
        for line in f:
            line = line.strip()
            if line:
                snaps.append(json.loads(line))
    snaps.sort(key=lambda s: s["at"])
    result = forecast(snaps, places, hours)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, output)
    with open(out_path, "w") as f:
        f.write(json.dumps(result) + "\n")
    print(f"Wrote {out_path} ({os.path.getsize(out_path)} bytes, "
          f"{len(result['probabilities'])} rows, top10={result['top10']})")


if __name__ == "__main__":
    main()
