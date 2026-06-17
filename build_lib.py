#!/usr/bin/env python3
"""
build_lib.py — shared analytics for the PSX dashboard.

compute_embed(det, fetch_charts=True) takes a long-format DataFrame with columns:
    Date (datetime.date), Symbol (str), Open, High, Low, Current (float),
    'Change %' (PERCENT, e.g. 8.41 not 0.0841), Volume (number)
and returns the dict the dashboard expects:
    snapshot, stats, positions, movers, history, charts
"""
import json, re, urllib.request, datetime as _dt
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import pandas as pd, numpy as np
from bs4 import BeautifulSoup

WINDOW = 10        # trading days to follow each new entry
CHART_DAYS = 180   # sessions of price history per symbol for the chart
UA = {"User-Agent": "Mozilla/5.0 (psx-dashboard)"}


def _cnum(s):
    if s is None:
        return None
    s = str(s).strip().replace(",", "").replace("%", "")
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    try:
        v = float(s)
        return -v if neg else v
    except ValueError:
        return None


def _fetch_company(sym):
    """Parse fundamentals, insider filings and catalysts from the PSX company page."""
    try:
        req = urllib.request.Request(f"https://dps.psx.com.pk/company/{sym}", headers=UA)
        html = urllib.request.urlopen(req, timeout=25).read().decode("utf-8", "ignore")
    except Exception:
        return None
    soup = BeautifulSoup(html, "lxml")
    T = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))

    def grab(label, pat=r"\s*([-\(\d,\.\)]+)"):
        m = re.search(label + pat, T)
        return _cnum(m.group(1)) if m else None

    title = soup.title.get_text() if soup.title else ""
    mname = re.search(r"Stock quote for (.+?) - Pakistan Stock Exchange", title)
    name = mname.group(1).strip() if mname else sym
    msec = re.search(re.escape(name) + r"\s+([A-Z][A-Z &/\-]{3,}?)\s+Rs\.", T)
    sector = msec.group(1).strip() if msec else None

    o = dict(name=name, sector=sector)
    m52 = re.search(r"52-WEEK RANGE[^\d]*([\d,\.]+)\s*[—\-–]\s*([\d,\.]+)", T)
    if m52:
        o["wk52_low"] = _cnum(m52.group(1))
        o["wk52_high"] = _cnum(m52.group(2))
    mc = grab(r"Market Cap \(000'?\s*s\s*\)")
    o["mcap_mn"] = round(mc / 1000, 1) if mc else None            # millions of PKR
    o["shares"] = grab(r"\bShares\b")
    ff = re.search(r"Free Float\s*([\d\.]+)%", T)
    o["free_float_pct"] = _cnum(ff.group(1)) if ff else None
    o["pe"] = grab(r"P/E Ratio \(TTM\)\s*\*\*")
    m52 = re.search(r"52-WEEK RANGE[^\d]*([\d,\.]+)\s*[\u2014\-\u2013]\s*([\d,\.]+)", T)
    if m52:
        o["low52"] = _cnum(m52.group(1))
        o["high52"] = _cnum(m52.group(2))

    def table_cols(tbl):
        rows = tbl.find_all("tr")
        head = [c.get_text(strip=True) for c in rows[0].find_all(["th", "td"])]
        body = {}
        for r in rows[1:]:
            cells = [c.get_text(strip=True) for c in r.find_all(["td", "th"])]
            if cells and cells[0]:
                body[cells[0]] = cells[1:]
        return head[1:], body

    annual = quarterly = ratios = None
    for tbl in soup.find_all("table"):
        rows = tbl.find_all("tr")
        if not rows:
            continue
        head = [c.get_text(strip=True) for c in rows[0].find_all(["th", "td"])]
        labels = [r.find(["td", "th"]).get_text(strip=True) for r in rows[1:] if r.find(["td", "th"])]
        if "EPS" in labels and len(head) > 1 and head[0] == "" and head[1][:1].isdigit():
            annual = table_cols(tbl)
        elif "EPS" in labels and len(head) > 1 and head[1].startswith("Q"):
            quarterly = table_cols(tbl)
        elif any("Margin" in l for l in labels):
            ratios = table_cols(tbl)

    def series(parsed, key):
        if not parsed or key not in parsed[1]:
            return None
        return [_cnum(x) for x in parsed[1][key]]

    if annual:
        o["years"] = annual[0]
        o["eps"] = series(annual, "EPS")
        o["sales"] = series(annual, "Sales")
        o["pat"] = series(annual, "Profit after Taxation")
    if quarterly:
        o["q_labels"] = quarterly[0]
        o["q_eps"] = series(quarterly, "EPS")
        o["q_sales"] = series(quarterly, "Sales")
        o["q_pat"] = series(quarterly, "Profit after Taxation")
        o["q_sales"] = series(quarterly, "Sales")
        o["q_pat"] = series(quarterly, "Profit after Taxation")
    if ratios:
        o["gross_margin"] = series(ratios, "Gross Profit Margin (%)")
        o["net_margin"] = series(ratios, "Net Profit Margin (%)")
        o["eps_growth"] = series(ratios, "EPS Growth (%)")
        o["peg"] = series(ratios, "PEG")

    insider, catalysts = [], []
    for tbl in soup.find_all("table"):
        for r in tbl.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in r.find_all("td")]
            if len(cells) >= 2 and re.match(r"[A-Z][a-z]{2} \d", cells[0]):
                t = cells[1]
                if re.search(r"Disclosure of Interest|Substantial Shareholder|acquisition of shares|disposal of shares", t, re.I):
                    insider.append([cells[0], t[:85]])
                elif re.search(r"dividend|bonus|book closure|sub.?division|split|right|board meeting|EOGM|AGM|merger|results", t, re.I):
                    catalysts.append([cells[0], t[:85]])
    o["insider"] = insider[:6]
    o["catalysts"] = catalysts[:6]
    return o


def _fetch_companies(symbols):
    out = {}
    print(f"Fetching company fundamentals for {len(symbols)} symbols…")
    with ThreadPoolExecutor(max_workers=12) as ex:
        for sym, c in ex.map(lambda s: (s, _fetch_company(s)), symbols):
            if c:
                out[sym] = c
    return out


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


def _eod_one(sym):
    try:
        req = urllib.request.Request(
            f"https://dps.psx.com.pk/timeseries/eod/{sym}", headers=UA)
        rows = json.load(urllib.request.urlopen(req, timeout=25)).get("data", [])
    except Exception:
        rows = []
    if not rows:
        return sym, None
    rows = sorted(rows, key=lambda r: r[0])[-CHART_DAYS:]
    return sym, dict(
        d=[int(_dt.datetime.utcfromtimestamp(r[0]).strftime("%Y%m%d")) for r in rows],
        o=[round(r[3], 2) for r in rows],
        c=[round(r[1], 2) for r in rows],
        v=[int(r[2]) for r in rows])


def _fetch_eod_charts(symbols):
    charts = {}
    with ThreadPoolExecutor(max_workers=12) as ex:
        for sym, ch in ex.map(_eod_one, symbols):
            if ch:
                charts[sym] = ch
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
    out["companies"] = _fetch_companies(sorted(det["Symbol"].unique())) if fetch_charts else {}
    return out


def render_html(template_path, embed, out_path):
    import math
    def _clean(o):
        if isinstance(o, float):
            return None if (math.isnan(o) or math.isinf(o)) else o
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_clean(v) for v in o]
        return o
    tpl = open(template_path, encoding="utf-8").read()
    html = tpl.replace("__DATA__", json.dumps(_clean(embed), separators=(",", ":")))
    open(out_path, "w", encoding="utf-8").write(html)
    return out_path
