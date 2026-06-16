#!/usr/bin/env python3
"""
build_lib.py — shared analytics for the PSX dashboard.

compute_embed(det, fetch_charts=True) takes a long-format DataFrame with columns:
    Date (datetime.date), Symbol (str), Open, High, Low, Current (float),
    'Change %' (PERCENT, e.g. 8.41 not 0.0841), Volume (number)
and returns the dict the dashboard expects:
    snapshot, stats, positions, movers, history, charts
"""
import json, urllib.request, datetime as _dt
from collections import defaultdict
import pandas as pd, numpy as np

WINDOW = 10        # trading days to follow each new entry
CHART_DAYS = 180   # sessions of price history per symbol for the chart
UA = {"User-Agent": "Mozilla/5.0 (psx-dashboard)"}


def _movers(hist, window_dates, topk=15):
    wset = set(window_dates); rows = []
    for s, h in hist.items():
        pts = [(d, p) for (d, p) in h if d in wset and p and p > 0]
        if len(pts) < 2 or pts[0][0] == pts[-1][0]:
            continue
        (sd, sp), (ed, ep) = pts[0], pts[-1]
        rows.append(dict(symbol=s, ret=round((ep - sp) / sp * 100, 2),
            start_price=round(sp, 2), end_price=round(ep, 2),
            start=str(sd), end=str(ed), points=len(pts)))
    return dict(gainers=sorted(rows, key=lambda x: -x["ret"])[:topk],
                losers=sorted(rows, key=lambda x: x["ret"])[:topk],
                universe=len(rows),
                **{"from": str(window_dates[0]), "to": str(window_dates[-1])})


def _fetch_eod_charts(symbols):
    charts = {}
    for k, sym in enumerate(symbols):
        try:
            req = urllib.request.Request(
                f"https://dps.psx.com.pk/timeseries/eod/{sym}", headers=UA)
            rows = json.load(urllib.request.urlopen(req, timeout=25)).get("data", [])
        except Exception:
            rows = []
        if not rows:
            continue
        rows = sorted(rows, key=lambda r: r[0])[-CHART_DAYS:]
        charts[sym] = dict(
            d=[int(_dt.datetime.utcfromtimestamp(r[0]).strftime("%Y%m%d")) for r in rows],
            o=[round(r[3], 2) for r in rows],
            c=[round(r[1], 2) for r in rows],
            v=[int(r[2]) for r in rows])
        if k % 25 == 0:
            print(f"  charts {k}/{len(symbols)}…")
    return charts


def compute_embed(det, fetch_charts=True):
    det = (det.dropna(subset=["Date"])
              .sort_values("Volume", ascending=False)
              .drop_duplicates(["Date", "Symbol"])).copy()

    # Pull authoritative EOD history once (also used for the charts), and use it
    # to correct each day's close/volume — the source spreadsheet can carry stale
    # values forward, which skews the movers/history. EOD is the exchange's own data.
    charts = _fetch_eod_charts(sorted(det["Symbol"].unique())) if fetch_charts else {}
    if charts:
        eod = {}
        for sym, ch in charts.items():
            for i, dd in enumerate(ch["d"]):
                eod[(sym, dd)] = (ch["c"][i], ch["v"][i])
        cur, vol, corrected = [], [], 0
        for r in det.itertuples():
            key = (r.Symbol, int(r.Date.strftime("%Y%m%d")))
            if key in eod:
                c, v = eod[key]
                if abs(c - r.Current) > 1e-6:
                    corrected += 1
                cur.append(c); vol.append(v)
            else:
                cur.append(r.Current); vol.append(r.Volume)
        det["Current"] = cur
        det["Volume"] = vol
        print(f"Reconciled {corrected} day-prices against PSX EOD.")

    dates  = sorted(det["Date"].unique())
    latest = dates[-1]
    price   = {(r.Date, r.Symbol): r.Current for r in det.itertuples()}
    members = {d: set(det[det["Date"] == d]["Symbol"]) for d in dates}
    ranked  = {d: list(det[det["Date"] == d].sort_values("Volume", ascending=False)["Symbol"])
               for d in dates}

    # latest leaderboard
    sub = det[det["Date"] == latest].sort_values("Volume", ascending=False)
    chg = dict(zip(zip(det["Date"], det["Symbol"]), det["Change %"]))
    top_n = [dict(symbol=r.Symbol, sector="", close=round(r.Current, 2),
                  change_pct=round(chg.get((latest, r.Symbol), 0.0), 2),
                  volume=int(r.Volume)) for r in sub.itertuples()]
    new_latest  = sorted(members[latest] - members[dates[-2]]) if len(dates) > 1 else ranked[latest]
    drop_latest = sorted(members[dates[-2]] - members[latest]) if len(dates) > 1 else []

    # forward-return tracking of new entries
    events = []
    for i, d in enumerate(dates):
        if i == 0:
            continue
        for sym in (members[d] - members[dates[i-1]]):
            ep = price[(d, sym)]
            if not ep or ep <= 0:
                continue
            peak = final = 0.0; seen = dtp = 0
            for k in range(1, WINDOW + 1):
                if i + k >= len(dates):
                    break
                fd = dates[i + k]
                if sym in members[fd]:
                    ret = (price[(fd, sym)] - ep) / ep * 100
                    seen, final = k, ret
                    if ret > peak:
                        peak, dtp = ret, k
            cur = price[(dates[min(i + seen, len(dates) - 1)], sym)] if seen else ep
            is_open = (len(dates) - 1 - i < WINDOW) and (sym in members[latest])
            events.append(dict(symbol=sym, sector="", entry_date=str(d),
                entry_price=round(ep, 2), current_price=round(cur, 2),
                return_pct=round(final, 2), peak_return_pct=round(peak, 2),
                days_tracked=seen, still_in_topN=(sym in members[latest]),
                days_to_peak=dtp, status="open" if is_open else "closed",
                outcome=None if is_open else ("win" if final > 0 else "loss")))

    T = [e for e in events if e["days_tracked"] > 0]
    peaks  = [e["peak_return_pct"] for e in T]
    finals = [e["return_pct"] for e in T]
    stats = dict(
        total_tracked=len(T),
        closed=sum(1 for e in events if e["status"] == "closed"),
        win_rate_pct=round(100 * np.mean([p >= 5 for p in peaks]), 1) if peaks else None,
        avg_return_pct=round(float(np.mean(finals)), 2) if finals else None,
        avg_peak_pct=round(float(np.mean(peaks)), 2) if peaks else None,
        pct_up10=round(100 * np.mean([p >= 10 for p in peaks]), 1) if peaks else None,
        pct_neverup=round(100 * np.mean([p <= 0 for p in peaks]), 1) if peaks else None,
        last_date=str(latest))

    ev_sorted = sorted(events, key=lambda e: e["entry_date"], reverse=True)
    opens   = sorted([e for e in ev_sorted if e["status"] == "open"], key=lambda e: -e["return_pct"])
    closeds = sorted([e for e in ev_sorted if e["status"] == "closed"][:40], key=lambda e: -e["peak_return_pct"])

    # movers
    hist = defaultdict(list)
    for r in det.itertuples():
        hist[r.Symbol].append((r.Date, r.Current))
    for s in hist:
        hist[s].sort()
    movers = {"1W": _movers(hist, dates[-5:]), "2W": _movers(hist, dates[-10:]),
              "1M": _movers(hist, dates[-20:]), "ALL": _movers(hist, dates)}

    # per-day history (newest first)
    volmap = {(r.Date, r.Symbol): r.Volume for r in det.itertuples()}
    history = []
    for i, d in enumerate(dates):
        syms = ranked[d]
        prev = set(ranked[dates[i-1]]) if i > 0 else set()
        history.append(dict(d=str(d), syms=syms,
            vol=[round(volmap[(d, s)] / 1e6, 1) for s in syms],
            new=[s for s in syms if s not in prev],
            drop=[s for s in prev if s not in set(syms)] if i > 0 else []))
    history.reverse()

    out = dict(snapshot=dict(date=str(latest), top_n=top_n,
               new_entries=new_latest, dropped=drop_latest),
               stats=stats, positions=opens + closeds, movers=movers, history=history,
               charts=charts)
    return out


def render_html(template_path, embed, out_path):
    tpl = open(template_path, encoding="utf-8").read()
    html = tpl.replace("__DATA__", json.dumps(embed, separators=(",", ":")))
    open(out_path, "w", encoding="utf-8").write(html)
    return out_path
