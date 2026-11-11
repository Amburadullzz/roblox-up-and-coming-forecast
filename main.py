"""Forecast Roblox Up-and-Coming top 10 places for the 72h after the history ends.

Reads $MAKEFUN_INPUT_DIR/history.ndjson (default /input) and writes
$MAKEFUN_OUTPUT_DIR/forecast.json (default /output). Python standard library
only, no network, deterministic: same history in, same bytes out.

Model: every pair of reads 1..72h apart in the history gives the transition of
a game from its rank then to a place now, keyed by rank group and by rank
momentum over the ~8h before. The five dt buckets are weighted by the hours
they cover in a 72h window, so the one forecast suits all 72 hourly reads.
Every game on the last chart is named at every place; the mass left unnamed is
the estimated share of places held by games published after the last read,
which the scorer spreads over exactly those games.
"""
import json
import math
import os
from collections import defaultdict

BUCKETS = [(1, 6), (6, 12), (12, 24), (24, 48), (48, 72.0001)]
BUCKET_W = [5.0, 6.0, 12.0, 24.0, 24.0]
LOOKBACK_H = 8.0
MOM_EDGES = [-1.0, 0.0, 0.5, 2.0]
NGRP = 19
NMOM = 6
SHRINK_GRP = 20000.0
SHRINK_CELL = 3.0
SHRINK_NEW = 150.0
MAX_ROWS_PER_PLACE = 100


def parse_at(s):
    """ISO 8601 UTC -> seconds since epoch (float). Avoids any local timezone."""
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1]
    if "." in s:
        head, frac = s.split(".", 1)
        frac = (frac + "000000")[:6]
    else:
        head, frac = s, "000000"
    day, clock = head.split("T")
    y, mo, d = [int(x) for x in day.split("-")]
    hh, mm, ss = [int(x) for x in clock.split(":")]
    return _days_from_civil(y, mo, d) * 86400.0 + hh * 3600.0 + mm * 60.0 + ss + int(frac) / 1e6


def _days_from_civil(y, m, d):
    y -= m <= 2
    era = (y if y >= 0 else y - 399) // 400
    yoe = y - era * 400
    doy = (153 * (m + (-3 if m > 2 else 9)) + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


def rank_group(j):
    if j <= 10:
        return j
    if j <= 12:
        return 11
    if j <= 15:
        return 12
    if j <= 20:
        return 13
    if j <= 25:
        return 14
    if j <= 35:
        return 15
    if j <= 50:
        return 16
    if j <= 70:
        return 17
    return 18


def mom_bucket(m):
    if m is None:
        return 0
    b = 1
    for e in MOM_EDGES:
        if m >= e:
            b += 1
    return b


def bucket_of(h):
    for x, (lo, hi) in enumerate(BUCKETS):
        if lo <= h < hi:
            return x
    return None


def load(reads):
    out = []
    for r in sorted(reads, key=lambda r: r["at"]):
        out.append((parse_at(r["at"]),
                    {g["universe_id"]: g["rank"] for g in r["up_and_coming"]},
                    r))
    return out


def features_at(cm, i):
    t_now, ra, _ = cm[i]
    lb = None
    for j in range(i - 1, -1, -1):
        dh = (t_now - cm[j][0]) / 3600.0
        if dh < 1.0:
            continue
        if dh > 14.0:
            break
        if lb is None or abs(dh - LOOKBACK_H) < abs(lb[0] - LOOKBACK_H):
            lb = (dh, j)
    prev = cm[lb[1]][1] if lb else {}
    dh = lb[0] if lb else 1.0
    feats = {}
    for u, r in ra.items():
        if u in prev:
            mom = mom_bucket((prev[u] - r) / dh)
        else:
            mom = mom_bucket(None)
        feats[u] = (rank_group(r), mom)
    return feats


def transitions(cm):
    n = len(cm)
    nb = len(BUCKETS)
    counts = [[defaultdict(lambda: [0.0] * 11) for _ in range(2)] for _ in range(nb)]
    totals = [[defaultdict(float) for _ in range(2)] for _ in range(nb)]
    newc = [[0.0] * 11 for _ in range(nb)]
    tot = [[0.0] * 11 for _ in range(nb)]
    for i in range(n):
        feats = features_at(cm, i)
        ra = cm[i][1]
        for b_i in range(i + 1, n):
            h = (cm[b_i][0] - cm[i][0]) / 3600.0
            bi = bucket_of(h)
            if bi is None:
                continue
            rb = cm[b_i][1]
            for u, r in ra.items():
                key1 = (feats[u][0],)
                key2 = feats[u]
                k = rb.get(u)
                totals[bi][0][key1] += 1.0
                totals[bi][1][key2] += 1.0
                if k is not None and k <= 10:
                    counts[bi][0][key1][k] += 1.0
                    counts[bi][1][key2][k] += 1.0
            chart = cm[b_i][2]["up_and_coming"]
            for k in range(1, 11):
                if k > len(chart):
                    break
                tot[bi][k] += 1.0
                if chart[k - 1]["universe_id"] not in ra:
                    newc[bi][k] += 1.0
    P = []
    NEW = []
    for bi in range(nb):
        gsum = [0.0] * 11
        for key, nk in totals[bi][0].items():
            c = counts[bi][0][key]
            for k in range(1, 11):
                gsum[k] += c[k]
        gtop = sum(gsum[1:]) + 1e-9
        base = [(gsum[k] + 1.0) / (gtop + 10.0) for k in range(11)]
        base[0] = 0.0
        lvl1 = {}
        for key, nk in totals[bi][0].items():
            c = counts[bi][0][key]
            lvl1[key] = [(c[k] + SHRINK_GRP * base[k]) / (nk + SHRINK_GRP) for k in range(11)]
        lvl2 = {}
        for key, nk in totals[bi][1].items():
            c = counts[bi][1][key]
            prior = lvl1[key[:1]]
            lvl2[key] = [(c[k] + SHRINK_CELL * prior[k]) / (nk + SHRINK_CELL) for k in range(11)]
        P.append(lvl2)
        pooled = (sum(newc[bi][1:]) + 1e-9) / (sum(tot[bi][1:]) + 1e-9)
        NEW.append([0.0] + [(newc[bi][k] + SHRINK_NEW * pooled) / (tot[bi][k] + SHRINK_NEW)
                            for k in range(1, 11)])
    return P, NEW


def forecast(reads):
    cm = load(reads)
    P, NEW = transitions(cm)
    i = len(cm) - 1
    feats = features_at(cm, i)
    last = cm[i][2]
    wsum = sum(BUCKET_W)
    newm = [sum(BUCKET_W[bi] * NEW[bi][k] for bi in range(len(BUCKETS))) / wsum
            for k in range(11)]
    probs = []
    for k in range(1, 11):
        rows = {}
        for g in last["up_and_coming"]:
            u = g["universe_id"]
            f = feats[u]
            v = 0.0
            for bi in range(len(BUCKETS)):
                cell = P[bi].get(f)
                if cell is None:
                    cell = [0.0] * 11
                v += BUCKET_W[bi] * cell[k]
            rows[u] = v / wsum
        items = sorted(rows.items(), key=lambda kv: (-kv[1], kv[0]))[:MAX_ROWS_PER_PLACE]
        s = sum(v for _, v in items)
        target = min(0.999, max(0.0, 1.0 - newm[k]))
        if s > 0:
            f2 = target / s
            for u, v in items:
                probs.append({"universe_id": u, "place": k, "p": round(v * f2, 8)})
    top10 = [g["universe_id"] for g in last["up_and_coming"][:10]]
    return {"probabilities": probs, "top10": top10}


def main():
    inp = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    reads = []
    path = os.path.join(inp, "history.ndjson")
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("up_and_coming"):
                reads.append(r)
    reads.sort(key=lambda r: r["at"])
    fc = forecast(reads)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "forecast.json"), "w", encoding="utf-8") as fh:
        json.dump(fc, fh, separators=(",", ":"), sort_keys=True)


if __name__ == "__main__":
    main()
