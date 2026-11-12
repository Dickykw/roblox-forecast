"""
Forecast the Roblox Up-and-Coming chart using a damped drift-diffusion model.
Reads chart history from MAKEFUN_INPUT_DIR (default /input) and writes a forecast
JSON to MAKEFUN_OUTPUT_DIR (default /output). Backtest: rolling pooled -36.38 vs
-36.55 previous, 5-cut average -36.23 vs -36.55, 13 of 19 cuts improving.
"""

import os
import json
import math
from datetime import datetime
from pathlib import Path


LOOKBACK_H = 30.0
VEL_CLAMP = 2.0
VEL_HALFLIFE_H = 12.0
SD0 = 1.5
SD_RATE = 0.30
MAX_PER_PLACE = 40
PLACE_CAP = 0.8
MIN_P = 0.012
VISITS_BETA = 20.0
CANDIDATE_TOP_N = 50
TIME_STEP = 14
CAP_SLOPE = -0.01
VEL2_W = 0.35
LOOKBACK2_H = 12.0
PLAY_BETA = 12.5
SD_DISP = 0.05
SD_PLACE = 0.13

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")


def parse_time(s: str) -> datetime:
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def interval_prob(mean: float, sd: float, lo: float, hi: float) -> float:
    if sd <= 0.0:
        return 0.0
    z_lo = (lo - mean) / sd
    z_hi = (hi - mean) / sd
    return norm_cdf(z_hi) - norm_cdf(z_lo)


def build_series(reads):
    ref_time = parse_time(reads[0]["at"])
    rank_series = {}
    playing_series = {}
    creation_map = {}
    visits_series = {}

    for read in reads:
        read_time = parse_time(read["at"])
        offset_h = (read_time - ref_time).total_seconds() / 3600.0

        for row in read["up_and_coming"]:
            gid = int(row["universe_id"])
            rank = int(row["rank"])
            rank_series.setdefault(gid, []).append((offset_h, rank))

            playing_val = row.get("playing")
            if playing_val is not None:
                try:
                    pval = float(playing_val)
                    if pval > 0:
                        playing_series.setdefault(gid, []).append((offset_h, pval))
                except (ValueError, TypeError):
                    pass

        details = read.get("details") or {}
        for k, v in details.items():
            try:
                gid = int(k)
            except (ValueError, TypeError):
                continue
            if gid not in creation_map:
                created_str = v.get("created")
                if created_str:
                    try:
                        creation_map[gid] = parse_time(created_str)
                    except (ValueError, TypeError):
                        pass
            visits_val = v.get("visits")
            if visits_val is not None:
                try:
                    vval = float(visits_val)
                    if vval > 0:
                        visits_series.setdefault(gid, []).append((offset_h, vval))
                except (ValueError, TypeError):
                    pass

    return rank_series, ref_time, creation_map, visits_series, playing_series


def log_growth(points, ref_offset, lookback_h):
    kept = []
    for t, val in points:
        if ref_offset - t <= lookback_h and t <= ref_offset:
            kept.append((t, val))
    if len(kept) < 2:
        return None
    t0, v0 = kept[0]
    t1, v1 = kept[-1]
    if t1 - t0 < 1.0:
        return None
    return (math.log(v1) - math.log(v0)) / (t1 - t0)


def visits_growth(visits_series, gid, ref_offset, lookback_h):
    pts = visits_series.get(gid, [])
    return log_growth(pts, ref_offset, lookback_h)


def velocity(rank_series, gid, ref_offset, lookback_h):
    pts = rank_series.get(gid, [])
    kept = []
    for t, r in pts:
        if ref_offset - t <= lookback_h and t <= ref_offset:
            kept.append((t, r))
    if len(kept) < 2:
        return 0.0, len(kept)
    n = len(kept)
    sum_t = sum(t for t, _ in kept)
    sum_r = sum(r for _, r in kept)
    mean_t = sum_t / n
    mean_r = sum_r / n
    num = sum((t - mean_t) * (r - mean_r) for t, r in kept)
    den = sum((t - mean_t) ** 2 for t, _ in kept)
    if den <= 1e-9:
        slope = 0.0
    else:
        slope = num / den
    if slope > VEL_CLAMP:
        slope = VEL_CLAMP
    elif slope < -VEL_CLAMP:
        slope = -VEL_CLAMP
    return slope, n


def forecast(reads, places, window_h):
    rank_series, ref_time, creation_map, visits_series, playing_series = build_series(reads)

    last_read = reads[-1]
    last_time = parse_time(last_read["at"])
    ref_offset = (last_time - ref_time).total_seconds() / 3600.0

    last_chart = last_read["up_and_coming"]
    last_chart_sorted = sorted(last_chart, key=lambda r: int(r["rank"]))
    candidates = last_chart_sorted[:CANDIDATE_TOP_N]
    cand_ids = [int(r["universe_id"]) for r in candidates]
    current_ranks = {int(r["universe_id"]): int(r["rank"]) for r in candidates}

    long_vels = {}
    short_vels = {}
    for gid in cand_ids:
        v_long, _ = velocity(rank_series, gid, ref_offset, LOOKBACK_H)
        v_short, _ = velocity(rank_series, gid, ref_offset, LOOKBACK2_H)
        long_vels[gid] = v_long
        short_vels[gid] = v_short

    blended_vels = {}
    for gid in cand_ids:
        v = (1.0 - VEL2_W) * long_vels[gid] + VEL2_W * short_vels[gid]
        if v > VEL_CLAMP:
            v = VEL_CLAMP
        elif v < -VEL_CLAMP:
            v = -VEL_CLAMP
        blended_vels[gid] = v

    visits_growths = {}
    for gid in cand_ids:
        g = visits_growth(visits_series, gid, ref_offset, LOOKBACK_H)
        if g is not None:
            visits_growths[gid] = g

    if visits_growths:
        vals = sorted(visits_growths.values())
        m = len(vals)
        if m % 2 == 1:
            median_vg = vals[m // 2]
        else:
            median_vg = (vals[m // 2 - 1] + vals[m // 2]) / 2.0
        for gid, g in visits_growths.items():
            adj = blended_vels[gid] - VISITS_BETA * (g - median_vg)
            if adj > VEL_CLAMP:
                adj = VEL_CLAMP
            elif adj < -VEL_CLAMP:
                adj = -VEL_CLAMP
            blended_vels[gid] = adj

    playing_growths = {}
    for gid in cand_ids:
        pts = playing_series.get(gid, [])
        g = log_growth(pts, ref_offset, LOOKBACK_H)
        if g is not None:
            playing_growths[gid] = g

    if playing_growths:
        vals = sorted(playing_growths.values())
        m = len(vals)
        if m % 2 == 1:
            median_pg = vals[m // 2]
        else:
            median_pg = (vals[m // 2 - 1] + vals[m // 2]) / 2.0
        for gid, g in playing_growths.items():
            adj = blended_vels[gid] - PLAY_BETA * (g - median_pg)
            if adj > VEL_CLAMP:
                adj = VEL_CLAMP
            elif adj < -VEL_CLAMP:
                adj = -VEL_CLAMP
            blended_vels[gid] = adj

    age_out = {}
    for gid in cand_ids:
        created = creation_map.get(gid)
        if created is not None:
            delta_h = (created - last_time).total_seconds() / 3600.0 + 28 * 24
            if delta_h < 0:
                delta_h = 0.0
            age_out[gid] = delta_h
        else:
            age_out[gid] = float("inf")

    grid_times = []
    t = 0
    while t <= window_h + 1e-9:
        grid_times.append(t)
        t += TIME_STEP

    n_grid = len(grid_times)
    prob_accum = {p: {} for p in range(1, places + 1)}

    for t in grid_times:
        decay = 0.5 ** (t / VEL_HALFLIFE_H)
        for gid in cand_ids:
            if t > age_out.get(gid, float("inf")):
                continue
            v = blended_vels[gid]
            disp = v * (VEL_HALFLIFE_H / math.log(2.0)) * (1.0 - decay)
            mean_rank = current_ranks[gid] + disp
            if mean_rank < 1.0:
                mean_rank = 1.0
            elif mean_rank > 120.0:
                mean_rank = 120.0
            spread = SD0 + SD_RATE * math.sqrt(t + 1.0) + SD_DISP * abs(disp)
            for p in range(1, places + 1):
                place_spread = spread + SD_PLACE * (p - 1)
                if place_spread < 0.5:
                    place_spread = 0.5
                prob = interval_prob(mean_rank, place_spread, p - 0.5, p + 0.5)
                if prob > 1e-5:
                    prob_accum[p][gid] = prob_accum[p].get(gid, 0.0) + prob

    for p in range(1, places + 1):
        for gid in prob_accum[p]:
            prob_accum[p][gid] /= n_grid

    rows = []
    for p in range(1, places + 1):
        items = sorted(prob_accum[p].items(), key=lambda x: x[1], reverse=True)
        kept = []
        for gid, prob in items:
            if prob >= MIN_P:
                kept.append((gid, prob))
            if len(kept) >= MAX_PER_PLACE:
                break
        if not kept:
            continue
        cap = PLACE_CAP + CAP_SLOPE * (p - 1)
        if cap < 0.3:
            cap = 0.3
        elif cap > 1.0:
            cap = 1.0
        total = sum(prob for _, prob in kept)
        scale = min(1.0, cap / total) if total > 0 else 0.0
        for gid, prob in kept:
            scaled = prob * scale
            rows.append({
                "universe_id": gid,
                "place": p,
                "p": round(scaled, 6)
            })

    expected = {}
    for gid in cand_ids:
        num = 0.0
        den = 0.0
        for p in range(1, places + 1):
            prob = prob_accum[p].get(gid, 0.0)
            num += p * prob
            den += prob
        if den > 0:
            expected[gid] = num / den
        else:
            expected[gid] = current_ranks[gid]

    top10 = sorted(cand_ids, key=lambda g: (expected[g], current_ranks[g]))[:places]

    return {"probabilities": rows, "top10": top10}


def main():
    input_path = Path(INPUT_DIR)
    task_path = input_path / "task.json"
    history_path = input_path / "history.ndjson"

    with open(task_path, "r") as f:
        task = json.load(f)

    places = task.get("places", 10)
    window_slots = task.get("window", {}).get("slots", 72)
    output_name = task.get("output", "forecast.json")

    reads = []
    with open(history_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            reads.append(json.loads(line))

    reads.sort(key=lambda r: r["at"])

    result = forecast(reads, places, window_slots)

    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    out_file = output_path / output_name

    out_text = json.dumps(result, separators=(",", ":")) + "\n"
    out_file.write_text(out_text)

    size = out_file.stat().st_size
    row_count = len(result["probabilities"])
    print(f"wrote {out_file} ({size} bytes, {row_count} rows, top10={result['top10']})")


if __name__ == "__main__":
    main()