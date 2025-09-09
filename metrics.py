#!/usr/bin/env python3
import os
import math
import logging
import json
import time as time_module
from datetime import datetime, date, timedelta, time
from contextlib import contextmanager

import mysql.connector
import pandas as pd
import numpy as np

# -------- CONFIGURATION --------
#TODO Create CLI input for database credentials

# windows in number of bars (rows); 1 bar ≈ 1 minute
WINDOWS = [15, 30, 60, 360, 720, 1440, 10080, 43200]
SUFFIX = {
    15:     "15m",
    30:     "30m",
    60:     "1h",
    360:    "6h",
    720:    "12h",
    1440:   "24h",
    10080:  "7d",
    43200:  "30d"
}

TABLE_NAME = "ohlc_metrics_daily"
STATE_FILE = os.path.join(os.path.dirname(__file__), "state.json")

# -------- LOGGING SETUP --------
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ohlc_metrics")

def load_state():
    """
    Load state.json. If it doesn’t exist, return a fresh template.
    {
      "symbols": { SYMBOL: "YYYY-MM-DD", … },
      "last_full_backfill": null
    }
    """
    if os.path.isfile(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)

    # No file yet → initialize an empty state
    state = {
        "symbols": {},            # we'll populate per‐symbol start‐dates later
        "last_full_backfill": None
    }
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    return state

def save_state(state):
    """Overwrite state.json with the in‐memory state."""
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, sort_keys=True)

# -------- DATABASE WRAPPER --------
class Database:
    def __init__(self, cfg):
        self.cfg = cfg
        self.conn = None

    def connect(self):
        if self.conn and self.conn.is_connected():
            return
        self.conn = mysql.connector.connect(**self.cfg)
        # ensure UTC
        try:
            self.conn.cursor().execute("SET time_zone = '+00:00'")
        except:
            pass

    def close(self):
        if self.conn:
            try:
                self.conn.close()
            except:
                pass
        self.conn = None

    @contextmanager
    def cursor(self, dict_cursor=False):
        self.connect()
        cur = self.conn.cursor(dictionary=dict_cursor, buffered=True)
        try:
            yield cur
            self.conn.commit()
        except:
            self.conn.rollback()
            raise
        finally:
            cur.close()

    def get_symbols(self):
        "Pull distinct symbols from ohlcvt"
        sql = "SELECT DISTINCT symbol FROM ohlcvt"
        with self.cursor(dict_cursor=True) as cur:
            cur.execute(sql)
            rows = cur.fetchall()
        return [r["symbol"] for r in rows]

    def get_symbol_start_date(self, symbol) -> date:
        sql = "SELECT MIN(`timestamp`) AS first_ts FROM ohlcvt WHERE symbol=%s"
        with self.cursor(dict_cursor=True) as cur:
            cur.execute(sql, (symbol,))
            row = cur.fetchone()
        return row and row["first_ts"] and row["first_ts"].date()

    def fetch_ohlcvt(self, symbol, limit_rows):
        """
        Fetch the most recent `limit_rows` rows for `symbol`,
        return as a DataFrame sorted ascending by timestamp.
        """
        sql = """
        SELECT `timestamp`, `open`, `high`, `low`, `close`, `volume`, `trades`
          FROM ohlcvt
         WHERE symbol = %s
         ORDER BY `timestamp` DESC
         LIMIT %s
        """
        with self.cursor(dict_cursor=True) as cur:
            cur.execute(sql, (symbol, limit_rows))
            rows = cur.fetchall()
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        # parse as naïve datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        # set index and drop the column
        df = df.set_index("timestamp", drop=True)
        # strip any tzinfo
        df.index = df.index.tz_localize(None)
        for c in ("open","high","low","close","volume","trades"):
            df[c] = df[c].astype(float)
        # rows came back newest-first; reverse to oldest-first
        return df.iloc[::-1]

    def ensure_table(self):
        """
        Create the target table if it does not exist.
        One row per (date, symbol). Columns:
          date, symbol, last_updated_ts, then for each window:
          close_mean_{sfx}, ... , adx_{sfx}, average_trades_{sfx}.
        """
        cols = [
            "`metric_date` DATE NOT NULL",
            "`symbol` VARCHAR(50) NOT NULL",
            "`last_updated_ts` DATETIME NOT NULL"
        ]
        # base A-group metrics
        base_metrics = [
            "close_mean","close_stddev","close_min","close_max",
            "volume_sum","vwap","price_deviation_vwap",
            "roc","returns_log","sma",
            "ema_fast","ema_slow","macd","macd_signal","macd_histogram",
            "bollinger_upper","bollinger_lower","bollinger_width",
            "rsi","atr","stochastic_k","stochastic_d",
            "obv","cci","adx"
        ]
        for w in WINDOWS:
            s = SUFFIX[w]
            for m in base_metrics:
                cols.append(f"`{m}_{s}` DOUBLE NULL")
            # extra: average number of trades
            cols.append(f"`average_trades_{s}` DOUBLE NULL")

        ddl = f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
          {','.join(cols)},
          PRIMARY KEY(`metric_date`,`symbol`)
        ) ENGINE=InnoDB;
        """
        with self.cursor() as cur:
            cur.execute(ddl)

    def upsert_metrics(self, metric_date: date, symbol: str, metrics: dict, updated_ts: datetime):
        """
        Insert or update one row in ohlc_metrics_daily.
        metrics: dict of column_name→value
        """
        clean = {}
        for k, v in metrics.items():
            # sanitize NaN/Inf → NULL
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                clean[k] = None
            else:
                clean[k] = v

        cols = ["`metric_date`","`symbol`","`last_updated_ts`"] + [f"`{c}`" for c in clean]
        ph   = ",".join(["%s"]*len(cols))
        upd  = ",".join([f"`last_updated_ts`=VALUES(`last_updated_ts`)"]
                        + [f"`{c}`=VALUES(`{c}`)" for c in clean])
        sql = f"""
        INSERT INTO {TABLE_NAME} ({','.join(cols)})
        VALUES ({ph})
        ON DUPLICATE KEY UPDATE {upd}
        """
        vals = [metric_date, symbol, updated_ts] + [clean[c] for c in clean]
        with self.cursor() as cur:
            cur.execute(sql, vals)

    def fetch_ohlcvt_full(self, symbol: str) -> pd.DataFrame:
        """
        Fetch the entire history of OHLCVT bars for `symbol`,
        return as a DataFrame sorted ascending by timestamp.
        """
        sql = """
        SELECT `timestamp`, `open`, `high`, `low`, `close`, `volume`, `trades`
          FROM ohlcvt
         WHERE symbol = %s
         ORDER BY `timestamp` ASC
        """
        with self.cursor(dict_cursor=True) as cur:
            cur.execute(sql, (symbol,))
            rows = cur.fetchall()

        if not rows:
            # no data
            return pd.DataFrame()

        # build DataFrame
        df = pd.DataFrame(rows)
        # parse timestamps as naïve (no tz)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        # ensure index is the timestamp column, drop the column
        df = df.set_index("timestamp", drop=True)
        # strip any accidental tzinfo
        df.index = df.index.tz_localize(None)
        # cast numeric columns
        for col in ("open", "high", "low", "close", "volume", "trades"):
            df[col] = df[col].astype(float)
        return df
    
    def has_metrics_row(self, metric_date: date, symbol: str) -> bool:
        """
        Return True if we've already inserted a metrics row for (metric_date, symbol).
        """
        sql = f"SELECT 1 FROM {TABLE_NAME} WHERE metric_date=%s AND symbol=%s LIMIT 1"
        with self.cursor() as cur:
            cur.execute(sql, (metric_date, symbol))
            return cur.fetchone() is not None

    def get_latest_timestamp(self, symbol=None):
        """
        If symbol is None, returns the global max timestamp from ohlcvt;
        otherwise the max timestamp for that symbol.
        """
        if symbol:
            sql = "SELECT MAX(`timestamp`) AS ts FROM ohlcvt WHERE symbol=%s"
            params = (symbol,)
        else:
            sql = "SELECT MAX(`timestamp`) AS ts FROM ohlcvt"
            params = ()
        with self.cursor(dict_cursor=True) as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
        return row and row["ts"]

# -------- PADDER --------
def pad_to_window(df: pd.DataFrame, window: int, freq='1min') -> pd.DataFrame:
    """
    Ensure df has exactly `window` rows ending at df.index[-1],
    filling forward/backward as needed.
    """
    if df.empty:
        return df
    end = df.index[-1]
    idx = pd.date_range(end=end, periods=window, freq=freq)
    df2 = df.reindex(idx).ffill().bfill()
    return df2

# -------- Backfill --------
def backfill_symbol(db, calc, state, symbol, cutoff_date=None):
    """
    1) Reads state['symbols'][symbol] → first_date (ISO string)  
    2) Fetches *all* bars for that symbol, once.  
    3) Walks day by day from first_date up through cutoff_date (defaults to yesterday).  
    4) Slices the big DataFrame at each day, computes metrics, upserts, and advances state.
    """
    # a) figure out start & end
    start_iso = state["symbols"].get(symbol)
    if not start_iso:
        log.warning(f"{symbol} has no start date in state; skipping.")
        return

    start_date = date.fromisoformat(start_iso)
    last_ts = db.get_latest_timestamp(symbol)
    if last_ts is None:
        log.warning(f"{symbol}: no bars in ohlcvt, skipping backfill.")
        return
    last_date = last_ts.date()

    # by default backfill through yesterday _relative to the last bar_
    end_date = cutoff_date or (last_date - timedelta(days=1))
    if start_date > end_date:
        log.info(f"{symbol} already backfilled through {start_iso}.")
        return

    # b) pull the full history once (ascending by timestamp)
    df_all = db.fetch_ohlcvt_full(symbol)  
    if df_all.empty:
        log.warning(f"{symbol}: no history found.")
        return

    # c) walk each day
    cursor = start_date
    now     = datetime.utcnow()
    while cursor <= end_date:
        # slice up through end‐of‐day (exclusive of midnight next day)
        day_end = datetime.combine(cursor + timedelta(days=1), time.min)
        df_day  = df_all.loc[:day_end].copy()
        if df_day.empty:
            log.warning(f"{symbol} ‑ no bars up to {cursor}; skipping.")
        else:
            # compute all windows at that snapshot
            metrics = calc.compute_for_symbol(df_day)
            db.upsert_metrics(cursor, symbol, metrics, now)
            log.info(f"{symbol} @ {cursor} backfilled.")

            # update state immediately
            state["symbols"][symbol] = cursor.isoformat()
            save_state(state)

        cursor += timedelta(days=1)

# -------- Live Loop --------
def get_due_windows(now: datetime) -> list[int]:
    due = []
    # every 30′ → update 15m & 30m
    if now.minute % 30 == 0:
        due += [15, 30]
    # on the hour → 1h, and if aligned → 6h, 12h
    if now.minute == 0:
        due.append(60)
        if now.hour % 6  == 0:  due.append(360)
        if now.hour % 12 == 0:  due.append(720)
    return due

def is_end_of_day(now: datetime) -> bool:
    # fires exactly at UTC midnight
    return now.hour == 0 and now.minute == 0

def live_intraday_update(db, calc, symbols, now=None):
    if now is None:
        latest = db.get_latest_timestamp()
        if latest is None:
            return
        now = latest
    today = now.date()
    due = get_due_windows(now)
    if not due:
        return

    # map windows → suffixes
    due_sfx = { SUFFIX[w] for w in due }

    for sym in symbols:
        df = db.fetch_ohlcvt(sym, max(WINDOWS))
        if df.empty:
            continue

        # compute every metric once
        all_metrics = calc.compute_for_symbol(df)

        # decide full vs partial upsert
        if not db.has_metrics_row(today, sym):
            # first tick of the day → write all windows
            to_upsert = all_metrics
        else:
            # overwrite only the due windows
            to_upsert = {
                k: v for k, v in all_metrics.items()
                if any(k.endswith(f"_{sfx}") for sfx in due_sfx)
            }

        db.upsert_metrics(today, sym, to_upsert, now)
        log.info(f"[Intraday] {sym} ({today}): updated windows {due}")

def live_end_of_day_closeout(db, calc, symbols, state, now=None):
    """
    At “midnight” of the data, recompute *yesterday* full row
    (all windows) and advance JSON state.
    """
    # 1) Derive now from DB if not explicitly passed
    if now is None:
        latest = db.get_latest_timestamp()   # returns a naïve datetime
        if latest is None:
            log.warning("[EOD] No data in ohlcvt; skipping.")
            return
        now = latest

    # 2) Yesterday’s date
    yday = now.date() - timedelta(days=1)

    # 3) Compute cutoff = 00:00:00 of “today” (exclusive)
    today_mid = datetime.combine(yday + timedelta(days=1), time.min)
    
    for sym in symbols:
        # fetch full history
        df_all = db.fetch_ohlcvt_full(sym)
        if df_all.empty:
            log.warning(f"[EOD] {sym}: no history; skipping.")
            continue

        # slice through end-of-yesterday
        df_slice = df_all.loc[:today_mid]
        if df_slice.empty:
            log.warning(f"[EOD] {sym}: no bars to close out for {yday}")
            continue

        # compute and upsert all windows
        metrics = calc.compute_for_symbol(df_slice)
        db.upsert_metrics(yday, sym, metrics, now)
        log.info(f"[EOD] {sym} @ {yday}: full metrics written")

        # advance JSON cursor & persist
        state["symbols"][sym] = yday.isoformat()
        save_state(state)

# -------- METRICS CALCULATOR --------
class MetricsCalculator:
    def __init__(self):
        self.windows = WINDOWS
        self.sfx     = SUFFIX

    def compute_for_symbol(self, df: pd.DataFrame) -> dict:
        """
        Given a DataFrame of up to max(WINDOWS) rows,
        compute each window's A-group metrics plus average_trades.
        Returns flat dict { "<metric>_<sfx>": value, ... }.
        """
        out = {}
        # pre-cache numpy/log
        logfn = math.log

        for w in self.windows:
            s = self.sfx[w]
            sub = df.copy().iloc[-w:]
            # pad to exactly w rows if needed
            if len(sub) < w:
                sub = pad_to_window(sub, w)

            c = sub["close"]
            h = sub["high"]
            l = sub["low"]
            v = sub["volume"]
            t = sub["trades"]

            # basic statistics
            mean_c = c.mean()
            std_c  = c.std(ddof=0)
            mn     = c.min()
            mx     = c.max()
            vsum   = v.sum()
            vwap   = (c*v).sum()/vsum if vsum>0 else 0.0
            last   = c.iloc[-1]
            devvw  = ((last-vwap)/vwap) if (vwap and vwap!=0) else 0.0
            roc    = ((last-c.iloc[0])/c.iloc[0]*100) if c.iloc[0]!=0 else 0.0
            try:
                rlog = logfn(last/c.iloc[0]) if c.iloc[0]>0 else 0.0
            except:
                rlog = 0.0
            sma_v = mean_c

            # EMAs & MACD
            span_fast   = max(2, int(w*0.2))
            span_slow   = max(2, int(w*0.4))
            ema_fast    = c.ewm(span=span_fast, adjust=False).mean().iloc[-1]
            ema_slow    = c.ewm(span=span_slow, adjust=False).mean().iloc[-1]
            macd_v      = ema_fast - ema_slow
            span_signal = max(2, int(w*0.15))
            macd_ser    = c.ewm(span=span_fast, adjust=False).mean() \
                         - c.ewm(span=span_slow, adjust=False).mean()
            macd_sig    = macd_ser.ewm(span=span_signal, adjust=False).mean().iloc[-1]
            macd_hist   = macd_v - macd_sig

            # Bollinger (on typical price)
            tp          = (h + l + c)/3
            tp_mean     = tp.mean()
            tp_std      = tp.std(ddof=0)
            boll_up     = tp_mean + 2*tp_std
            boll_lo     = tp_mean - 2*tp_std
            boll_w      = boll_up - boll_lo

            # RSI
            diff        = c.diff().dropna()
            gains       = diff.clip(lower=0)
            losses      = -diff.clip(upper=0)
            avg_g       = gains.mean() if not gains.empty else None
            avg_l       = losses.mean() if not losses.empty else None
            if avg_l in (None,0):
                rsi_v = 100 if (avg_g and avg_g>0) else 50
            else:
                rs    = avg_g/avg_l
                rsi_v = 100 - (100/(1+rs))

            # ATR
            tr1 = h - l
            tr2 = (h - c.shift(1)).abs()
            tr3 = (l - c.shift(1)).abs()
            tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).dropna()
            atr = tr.mean() if not tr.empty else 0.0

            # Stochastic
            min_c = c.min()
            max_c = c.max()
            st_k  = 100*(last - min_c)/(max_c - min_c) if (max_c-min_c)!=0 else 0.0
            st_d  = st_k

            # OBV
            obv = (np.sign(c.diff().fillna(0))*v).cumsum().iloc[-1]

            # CCI
            typical = tp
            sma_tp  = typical.mean()
            mad     = np.mean(np.abs(typical - sma_tp))
            cci_v   = ((typical.iloc[-1]-sma_tp)/(0.015*mad)) if mad!=0 else 0.0

            # ADX
            if len(c)>=2 and atr>0:
                dm_p = h.diff().clip(lower=0).dropna()
                dm_m = (-l.diff()).clip(lower=0).dropna()
                adx_v = None
                if not dm_p.empty and not dm_m.empty:
                    ap = dm_p.mean(); am = dm_m.mean()
                    if (ap+am)!=0:
                        pdi = 100*ap/atr
                        mdi = 100*am/atr
                        adx_v = 100*abs(pdi-mdi)/(pdi+mdi)
            else:
                adx_v = 0.0

            # Average number of trades in window
            avg_trades = t.mean() if not t.empty else 0.0

            # pack into out
            out.update({
                f"close_mean_{s}": mean_c,
                f"close_stddev_{s}": std_c,
                f"close_min_{s}": mn,
                f"close_max_{s}": mx,
                f"volume_sum_{s}": vsum,
                f"vwap_{s}": vwap,
                f"price_deviation_vwap_{s}": devvw,
                f"roc_{s}": roc,
                f"returns_log_{s}": rlog,
                f"sma_{s}": sma_v,
                f"ema_fast_{s}": ema_fast,
                f"ema_slow_{s}": ema_slow,
                f"macd_{s}": macd_v,
                f"macd_signal_{s}": macd_sig,
                f"macd_histogram_{s}": macd_hist,
                f"bollinger_upper_{s}": boll_up,
                f"bollinger_lower_{s}": boll_lo,
                f"bollinger_width_{s}": boll_w,
                f"rsi_{s}": rsi_v,
                f"atr_{s}": atr,
                f"stochastic_k_{s}": st_k,
                f"stochastic_d_{s}": st_d,
                f"obv_{s}": obv,
                f"cci_{s}": cci_v,
                f"adx_{s}": adx_v,
                f"average_trades_{s}": avg_trades
            })

        return out


# -------- MAIN ORCHESTRATION --------
def main():
    log.info("Starting OHLC metrics computation.")
    db    = Database(DB_CONFIG)
    calc  = MetricsCalculator()
    state = load_state()

    try:
        # 1) ensure target table exists
        db.ensure_table()

        # 2) load all symbols
        symbols = db.get_symbols()
        log.info(f"Found {len(symbols)} symbols to process.")

        # 3) initialize any brand‐new symbols in state.json
        for sym in symbols:
            if sym not in state["symbols"]:
                first_dt = db.get_symbol_start_date(sym)
                if first_dt:
                    state["symbols"][sym] = first_dt.isoformat()
                    save_state(state)

        # 4) backfill up through *yesterday* for each symbol
        latest = db.get_latest_timestamp()
        if latest is None:
            log.error("No bars in ohlcvt; nothing to do.")
            return

        cutoff = latest.date() - timedelta(days=1)
        for sym in symbols:
            backfill_symbol(db, calc, state, sym, cutoff_date=cutoff)

        log.info("Backfill complete — entering live loop.")

        # 5) live loop
        while True:
            now = datetime.utcnow()

            # 5a) intraday update (15m, 30m, 1h, 6h, 12h as due)
            live_intraday_update(db, calc, symbols, now)

            # 5b) end-of-day closeout at UTC midnight
            if is_end_of_day(now):
                live_end_of_day_closeout(db, calc, symbols, state, now)

            # 5c) sleep until the top of the next minute
            secs = 60 - now.second
            time_module.sleep(secs)

    except KeyboardInterrupt:
        log.info("Received interrupt – shutting down cleanly.")
    finally:
        db.close()
        log.info("All done.")


if __name__ == "__main__":
    main()