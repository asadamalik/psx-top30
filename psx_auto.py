#!/usr/bin/env python3
"""
psx_auto.py — fully automated daily updater for the PSX Volume-Signal dashboard.

Each run it:
  1. fetches the live market-watch from https://dps.psx.com.pk (all symbols, real OHLCV),
  2. appends today's Top-30 to a running history file (psx_data/snapshots.json),
  3. recomputes new/dropped entries, the performance tracker, movers and history,
  4. pulls fresh price history for the charts,
  5. writes a ready-to-open psx_dashboard.html with everything baked in.

First run: if psx_data/snapshots.json doesn't exist and PSX_Top30_Daily.xlsx is
present in this folder, it seeds the history from your spreadsheet so the dashboard
starts fully populated. After that the spreadsheet is no longer needed.

Idempotent: running twice on the same day overwrites that day's snapshot.
Skips weekends. (Public holidays: it will re-store the last session under today's
date; harmless, but you can delete stray dates from psx_data/snapshots.json.)

Requires: pip install pandas openpyxl beautifulsoup4 lxml numpy
"""
import os, sys, json, datetime as dt, urllib.request
import pandas as pd
from bs4 import BeautifulSoup
import build_lib

HERE     = os.path.dirname(os.path.abspath(__file__))
STATE    = os.path.join(HERE, "psx_data")
SNAP     = os.path.join(STATE, "snapshots.json")
TEMPLATE = os.path.join(HERE, "dashboard_template.html")
OUT      = os.path.join(HERE, "psx_dashboard.html")
XLSX     = os.path.join(HERE, "PSX_Top30_Daily.xlsx")
TOP_N    = 30
UA       = {"User-Agent": "Mozilla/5.0 (psx-auto)"}


def fetch_marketwatch():
    """Return list of dicts for every symbol: symbol, sector, o,h,l,c, chg(%), vol."""
    req = urllib.request.Request("https://dps.psx.com.pk/market-watch", headers=UA)
    html = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "ignore")
    soup = BeautifulSoup(html, "lxml")

    def num(t):
        try:
            return float(t.replace(",", "").replace("%", ""))
        except ValueError:
            return None

    best = {}
    for tr in soup.select("tr"):
        tds = tr.find_all("td")
        if len(tds) < 11:
            continue
        sym = tds[0].get_text(strip=True)
        if not sym or sym == "ALLSHR":
            continue
        c = num(tds[7].get_text(strip=True))
        vtd = tds[10]
        vraw = vtd.get("data-order") or vtd.get_text(strip=True).replace(",", "")
        try:
            vol = int(float(vraw))
        except ValueError:
            continue
        if c is None:
            continue
        if sym not in best or vol > best[sym]["vol"]:
            best[sym] = dict(symbol=sym, sector=tds[1].get_text(strip=True),
                o=num(tds[4].get_text(strip=True)), h=num(tds[5].get_text(strip=True)),
                l=num(tds[6].get_text(strip=True)), c=c,
                chg=num(tds[9].get_text(strip=True)) or 0.0, vol=vol)
    return list(best.values())


def load_snaps():
    return json.load(open(SNAP)) if os.path.exists(SNAP) else []


def save_snaps(snaps):
    os.makedirs(STATE, exist_ok=True)
    json.dump(snaps, open(SNAP, "w"), separators=(",", ":"))


def seed_from_xlsx():
    if not os.path.exists(XLSX):
        return []
    print("Seeding history from PSX_Top30_Daily.xlsx …")
    det = pd.read_excel(XLSX, "Details")
    det["Date"] = pd.to_datetime(det["Date"], errors="coerce").dt.date
    det = det.dropna(subset=["Date", "Symbol", "Current", "Volume"])
    det = det.sort_values("Volume", ascending=False).drop_duplicates(["Date", "Symbol"])
    det = det.rename(columns={"Change %": "ChgPct"})
    for col in ("Open", "High", "Low", "ChgPct"):
        if col not in det.columns:
            det[col] = float("nan")
    snaps = []
    for d in sorted(det["Date"].unique()):
        sub = det[det["Date"] == d].sort_values("Volume", ascending=False).head(TOP_N)
        top = []
        for r in sub.itertuples():
            def f(x):
                try: return float(x)
                except Exception: return None
            top.append(dict(symbol=r.Symbol, sector="", o=f(r.Open), h=f(r.High),
                l=f(r.Low), c=float(r.Current),
                chg=round((f(r.ChgPct) or 0.0) * 100, 2), vol=int(r.Volume)))
        snaps.append(dict(date=str(d), top=top))
    return snaps


def snaps_to_det(snaps):
    rows = []
    for snap in snaps:
        d = dt.date.fromisoformat(snap["date"])
        for r in snap["top"]:
            rows.append(dict(Date=d, Symbol=r["symbol"], Open=r.get("o"),
                High=r.get("h"), Low=r.get("l"), Current=r["c"],
                **{"Change %": r.get("chg", 0.0)}, Volume=r["vol"]))
    return pd.DataFrame(rows)


def main():
    today = dt.date.today()
    if today.weekday() >= 5:
        print(f"{today} is a weekend — PSX is closed, nothing to do.")
        # still rebuild so the file exists / reflects latest stored data
    snaps = load_snaps()
    if not snaps:
        snaps = seed_from_xlsx()

    if today.weekday() < 5:
        try:
            rows = fetch_marketwatch()
        except Exception as e:
            print(f"Could not fetch PSX market-watch: {e}")
            rows = []
        if rows:
            rows.sort(key=lambda r: -r["vol"])
            top = rows[:TOP_N]
            rec = dict(date=str(today), top=top)
            snaps = [s for s in snaps if s["date"] != str(today)]  # overwrite today
            snaps.append(rec)
            snaps.sort(key=lambda s: s["date"])
            save_snaps(snaps)
            if len(snaps) < 2:
                new_syms = "(history too short)"
            else:
                diff = {r["symbol"] for r in top} - {r["symbol"] for r in snaps[-2]["top"]}
                new_syms = ", ".join(sorted(diff)) or "none"
            print(f"{today}: stored Top-{TOP_N}.  New vs previous session: {new_syms}")

    if not snaps:
        print("No data yet. Put PSX_Top30_Daily.xlsx here for an initial seed, "
              "or run again on a trading day.")
        return

    det = snaps_to_det(snaps)
    embed = build_lib.compute_embed(det, fetch_charts=True)
    build_lib.render_html(TEMPLATE, embed, OUT)
    print(f"Dashboard written -> {OUT}")
    print(f"Open it in a browser. History spans {embed['snapshot']['date']} "
          f"and {len(embed['history'])} trading days.")


if __name__ == "__main__":
    main()
