"""Feature-based hierarchical rank-transition forecaster (v4).

Cells keyed by (rank group, momentum bucket, tenure bucket) are estimated from
every pair of reads 1..72h apart in the history, shrunk toward coarser levels,
and averaged over the hours of the 72h window.
"""
import json, math, datetime as dt
from collections import defaultdict

BUCKETS = [(1, 6), (6, 12), (12, 24), (24, 48), (48, 72.0001)]
BUCKET_W = [8.0, 8.0, 12.0, 20.0, 24.0]
LOOKBACK_H = 6.0
MOM_EDGES = [-0.5, 0.0, 0.5, 2.0]
TEN_EDGES = [6.0, 24.0, 72.0]
NMOM = 6
NTEN = 5
NGRP = 19
SHRINK = {1: 400.0, 2: 6.0, 3: 5.0}
FEATURES = ("grp", "mom", "ten")
SHRINK_NEW = 150.0
RECENCY = None
TEMP = 0.85
NAMED_SCALE = 0.9


def T(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


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


def ten_bucket(h):
    b = 1
    for e in TEN_EDGES:
        if h >= e:
            b += 1
    return b


def mom24_bucket(m):
    if m is None:
        return 0
    b = 1
    for e in (-0.05, 0.0, 0.05):
        if m >= e:
            b += 1
    return b


def grw_bucket(g):
    if g is None:
        return 0
    b = 1
    for e in (0.0, 0.02, 0.05):
        if g >= e:
            b += 1
    return b


def bucket_of(h):
    for x, (lo, hi) in enumerate(BUCKETS):
        if lo <= h < hi:
            return x
    return None


def load(reads):
    """sorted (t, rank map, read dict)."""
    out = []
    for r in sorted(reads, key=lambda r: r["at"]):
        out.append((T(r["at"]), {g["universe_id"]: g["rank"] for g in r["up_and_coming"]}, r))
    return out


def features_at(cm, i):
    """per-game feature tuple for read i."""
    t_now, ra, _ = cm[i]
    lb = None
    for j in range(i - 1, -1, -1):
        dh = (t_now - cm[j][0]).total_seconds() / 3600.0
        if dh < 1.0:
            continue
        if dh > 14.0:
            break
        if lb is None or abs(dh - LOOKBACK_H) < abs(lb[0] - LOOKBACK_H):
            lb = (dh, j)
    prev = cm[lb[1]][1] if lb else {}
    dh = lb[0] if lb else 1.0
    first = {}
    for j in range(i + 1):
        for u in cm[j][1]:
            if u not in first:
                first[u] = cm[j][0]
    prev_read = cm[lb[1]][2] if lb else None
    play_prev = {}
    play_now = {}
    if prev_read:
        for g in prev_read["up_and_coming"]:
            if g.get("playing"):
                play_prev[g["universe_id"]] = g["playing"]
    for g in cm[i][2]["up_and_coming"]:
        if g.get("playing"):
            play_now[g["universe_id"]] = g["playing"]
    lb24 = None
    for j in range(i - 1, -1, -1):
        dh2 = (t_now - cm[j][0]).total_seconds() / 3600.0
        if dh2 < 1.0:
            continue
        if dh2 > 40.0:
            break
        if lb24 is None or abs(dh2 - 24.0) < abs(lb24[0] - 24.0):
            lb24 = (dh2, j)
    prev24 = cm[lb24[1]][1] if lb24 else {}
    dh24 = lb24[0] if lb24 else 1.0
    feats = {}
    for u, r in ra.items():
        grp = rank_group(r)
        mom = mom_bucket(((prev[u] - r) / dh) if u in prev else None)
        ten = ten_bucket((t_now - first[u]).total_seconds() / 3600.0)
        m24 = mom24_bucket(((prev24[u] - r) / dh24) if u in prev24 else None)
        if play_now.get(u) and play_prev.get(u):
            grw = grw_bucket(math.log(play_now[u] / play_prev[u]) / dh)
        else:
            grw = grw_bucket(None)
        feats[u] = {"grp": grp, "mom": mom, "ten": ten, "m24": m24, "grw": grw}
    return feats


def _key(feat, upto):
    k = []
    for name in FEATURES[:upto]:
        k.append(feat[name])
    return tuple(k)


def transitions(reads):
    cm = load(reads)
    n = len(cm)
    nb = len(BUCKETS)
    counts = [[defaultdict(lambda: [0.0] * 11) for _ in range(len(FEATURES) + 1)] for _ in range(nb)]
    totals = [[defaultdict(float) for _ in range(len(FEATURES) + 1)] for _ in range(nb)]
    newc = [[0.0] * 11 for _ in range(nb)]
    tot = [[0.0] * 11 for _ in range(nb)]
    for i in range(n):
        feats = features_at(cm, i)
        ra = cm[i][1]
        for b_i in range(i + 1, n):
            h = (cm[b_i][0] - cm[i][0]).total_seconds() / 3600.0
            bi = bucket_of(h)
            if bi is None:
                continue
            rb = cm[b_i][1]
            w = 1.0
            if RECENCY:
                w = math.exp(-(cm[-1][0] - cm[i][0]).total_seconds() / 86400.0 / RECENCY)
            for u, r in ra.items():
                f = feats[u]
                k = rb.get(u)
                for lvl in range(1, len(FEATURES) + 1):
                    key = _key(f, lvl)
                    totals[bi][lvl][key] += w
                    if k is not None and k <= 10:
                        counts[bi][lvl][key][k] += w
            for k in range(1, 11):
                tot[bi][k] += w
                if cm[b_i][2]["up_and_coming"][k - 1]["universe_id"] not in ra:
                    newc[bi][k] += w
    P = []
    for bi in range(nb):
        M = []
        for lvl in range(1, len(FEATURES) + 1):
            M.append({})
        gsum = [0.0] * 11
        gtot = 0.0
        for key, nk in totals[bi][1].items():
            gtot += nk
            c = counts[bi][1][key]
            for k in range(11):
                gsum[k] += c[k]
        gtop = sum(gsum[1:]) + 1e-9
        gbase = [(gsum[k] + 1.0) / (gtop + 10.0) for k in range(11)]
        gbase[0] = 0.0
        for lvl in range(1, len(FEATURES) + 1):
            keys = totals[bi][lvl].keys()
            for key in keys:
                nk = totals[bi][lvl][key]
                c = counts[bi][lvl][key]
                if lvl == 1:
                    prior = gbase
                    s = SHRINK[1]
                else:
                    prior = M[lvl - 2][key[:-1]]
                    s = SHRINK.get(lvl, 5.0)
                M[lvl - 1][key] = [(c[k] + s * prior[k]) / (nk + s) for k in range(11)]
        P.append(M)
    NEW = []
    for bi in range(nb):
        pooled = (sum(newc[bi][1:]) + 1e-9) / (sum(tot[bi][1:]) + 1e-9)
        NEW.append([0.0] + [(newc[bi][k] + SHRINK_NEW * pooled) / (tot[bi][k] + SHRINK_NEW) for k in range(1, 11)])
    return P, NEW


def forecast(reads, name_max_rank=999):
    reads = sorted(reads, key=lambda r: r["at"])
    P, NEW = transitions(reads)
    cm = load(reads)
    i = len(cm) - 1
    feats = features_at(cm, i)
    last = cm[i][2]
    wsum = sum(BUCKET_W)
    nlvl = len(FEATURES)
    newm = [sum(BUCKET_W[bi] * NEW[bi][k] for bi in range(len(BUCKETS))) / wsum for k in range(11)]
    probs = []
    for k in range(1, 11):
        rows = {}
        for g in last["up_and_coming"]:
            if g["rank"] > name_max_rank:
                continue
            f = feats[g["universe_id"]]
            v = 0.0
            for bi in range(len(BUCKETS)):
                key = _key(f, nlvl)
                cell = P[bi][nlvl - 1].get(key)
                if cell is None:
                    cell = P[bi][0].get(_key(f, 1))
                    if cell is None:
                        cell = [1.0 / 10.0] * 11
                        cell[0] = 0.0
                v += BUCKET_W[bi] * cell[k]
            rows[g["universe_id"]] = v / wsum
        if TEMP != 1.0:
            rows = {u: (v ** TEMP if v > 0 else 0.0) for u, v in rows.items()}
        s = sum(rows.values())
        target = min(0.999, max(0.0, (1.0 - newm[k]) * NAMED_SCALE))
        if s > 0:
            f2 = target / s
            for u in rows:
                rows[u] *= f2
        for u, v in rows.items():
            probs.append({"universe_id": u, "place": k, "p": round(v, 8)})
    return {"probabilities": probs, "top10": [g["universe_id"] for g in last["up_and_coming"][:10]]}


# ---- entry point -----------------------------------------------------------
import os, sys

def main():
    inp = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    reads = []
    with open(os.path.join(inp, "history.ndjson")) as f:
        for line in f:
            line = line.strip()
            if line:
                reads.append(json.loads(line))
    fc = forecast(reads)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "forecast.json"), "w") as f:
        json.dump({"probabilities": fc["probabilities"], "top10": fc["top10"]}, f)

if __name__ == "__main__":
    main()
