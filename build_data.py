#!/usr/bin/env python3
"""
build_data.py — モメンタム銘柄相関係数 本番バッチ
18銘柄の日次リターン相関を 10/30/50 日の3期間で計算し、data.json を出力する。
第3号 kioxia-sandisk/build_data.py の作法を踏襲。
"""

import json
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

# =========================================================================
# 定数
# =========================================================================
JST            = timezone(timedelta(hours=9))
FETCH_START    = "2026-04-01"
MAX_RETRIES    = 5
RETRY_INTERVAL = 10
PERIODS        = [10, 30, 50]
STALE_DAYS     = 5  # 最新データが何日以上古い場合に警告

STOCKS = [
    {"code": "285A", "ticker": "285A.T",  "name": "キオクシアHLDG"},
    {"code": "6976", "ticker": "6976.T",  "name": "太陽誘電"},
    {"code": "9984", "ticker": "9984.T",  "name": "ソフトバンクグループ"},
    {"code": "6981", "ticker": "6981.T",  "name": "村田製作所"},
    {"code": "8035", "ticker": "8035.T",  "name": "東京エレクトロン"},
    {"code": "6920", "ticker": "6920.T",  "name": "レーザーテック"},
    {"code": "4062", "ticker": "4062.T",  "name": "イビデン"},
    {"code": "5801", "ticker": "5801.T",  "name": "古河電気工業"},
    {"code": "5706", "ticker": "5706.T",  "name": "三井金属"},
    {"code": "6525", "ticker": "6525.T",  "name": "KOKUSAI ELECTRIC"},
    {"code": "6723", "ticker": "6723.T",  "name": "ルネサスエレクトロニクス"},
    {"code": "4004", "ticker": "4004.T",  "name": "レゾナック・HLDG"},
    {"code": "6762", "ticker": "6762.T",  "name": "TDK"},
    {"code": "6752", "ticker": "6752.T",  "name": "パナソニックHLDG"},
    {"code": "3436", "ticker": "3436.T",  "name": "SUMCO"},
    {"code": "7735", "ticker": "7735.T",  "name": "SCREEN HLDG"},
    {"code": "6098", "ticker": "6098.T",  "name": "リクルートHLDG"},
    {"code": "5803", "ticker": "5803.T",  "name": "フジクラ"},
]

SCRIPT_DIR   = Path(__file__).parent
DATA_PATH    = SCRIPT_DIR / "data.json"
CODE_TO_NAME = {s["code"]: s["name"] for s in STOCKS}


# =========================================================================
# ユーティリティ
# =========================================================================
def now_jst() -> datetime:
    return datetime.now(JST)


def to_iso_jst(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(JST).strftime("%Y-%m-%dT%H:%M:%S+09:00")


# =========================================================================
# 株価取得（リトライ付き）
# =========================================================================
def fetch_close_series(stock: dict) -> pd.Series | None:
    ticker_str = stock["ticker"]
    for attempt in range(MAX_RETRIES):
        try:
            hist = yf.Ticker(ticker_str).history(start=FETCH_START, auto_adjust=True)
            if hist.empty:
                raise ValueError("empty history")
            closes = hist["Close"].dropna()
            if closes.empty:
                raise ValueError("all NaN closes")
            closes.index = pd.DatetimeIndex(closes.index.date)
            closes.name = stock["code"]
            return closes
        except Exception as e:
            print(f"  [{ticker_str}] 取得失敗({attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_INTERVAL)
    return None


# =========================================================================
# クラスタ並び替え（貪欲最近傍法・scipy不要）
# =========================================================================
def greedy_cluster_order(corr: pd.DataFrame) -> list[str]:
    """
    50日相関を基準に貪欲最近傍法で銘柄順を決める。
    起点: 対角を除いた平均相関が最大の銘柄（群れの中心）。
    直前に並べた銘柄と最も相関の高い未配置銘柄を順に連結する。
    """
    codes = corr.columns.tolist()
    n = len(codes)
    mask = ~np.eye(n, dtype=bool)
    avg = (corr.values * mask).sum(axis=1) / (n - 1)
    start = codes[int(np.argmax(avg))]

    ordered = [start]
    remaining = set(codes) - {start}
    while remaining:
        last = ordered[-1]
        best = max(remaining, key=lambda c: float(corr.loc[last, c]))
        ordered.append(best)
        remaining.remove(best)
    return ordered


# =========================================================================
# ペアランキング
# =========================================================================
def build_pair_ranking(corr: pd.DataFrame, top_n: int = 10) -> tuple[list, list]:
    codes = corr.columns.tolist()
    pairs = []
    for i in range(len(codes)):
        for j in range(i + 1, len(codes)):
            pairs.append({
                "a":   codes[i],
                "b":   codes[j],
                "rho": round(float(corr.iloc[i, j]), 4),
            })
    pairs.sort(key=lambda x: x["rho"], reverse=True)
    return pairs[:top_n], pairs[-top_n:][::-1]


# =========================================================================
# メイン
# =========================================================================
def build_data() -> None:
    generated_at = now_jst()
    today_str    = generated_at.date().isoformat()
    alerts: list[str] = []

    print("=== build_data.py: モメンタム銘柄相関係数 ===")
    print(f"生成日時(JST): {to_iso_jst(generated_at)}\n")

    # --- 株価取得 ---
    print("▼ 終値取得中...")
    close_dict: dict[str, pd.Series] = {}
    failed_codes: list[str] = []

    for stock in STOCKS:
        series = fetch_close_series(stock)
        if series is not None:
            close_dict[stock["code"]] = series
            n     = len(series)
            first = series.index[0]
            last  = series.index[-1]
            print(f"  {stock['ticker']:10s}  {stock['name']:24s}  {n:3d}本  {first}〜{last}")
        else:
            failed_codes.append(stock["code"])
            alerts.append(f"[警告] {stock['ticker']}({stock['name']}): 取得失敗。計算から除外。")
            print(f"  {stock['ticker']:10s}  {stock['name']:24s}  [取得失敗]")

    if not close_dict:
        print("[ERROR] 全銘柄取得失敗。data.json を更新しません。")
        sys.exit(1)

    # --- 共通営業日・リターン計算 ---
    close_df   = pd.concat(close_dict.values(), axis=1, join="inner")
    returns_df = close_df.pct_change().dropna()
    common_days    = len(close_df)
    common_returns = len(returns_df)

    # --- 日付逆行 / Staleチェック ---
    latest_date = close_df.index[-1].date() if hasattr(close_df.index[-1], 'date') else close_df.index[-1]
    today_date  = generated_at.date()
    days_diff   = (today_date - latest_date).days
    if days_diff > STALE_DAYS:
        msg = (f"[警告] 最新データ日付({latest_date})が取得日({today_date})より{days_diff}日古い。"
               f"yfinance反映遅延の可能性。")
        alerts.append(msg)
        print(msg)

    print(f"\n▼ データ健全性")
    print(f"  共通終値本数: {common_days}  /  50日充足(51本): {'OK' if common_days >= 51 else 'NG'}")
    print(f"  共通リターン本数: {common_returns}")
    print(f"  最新日付: {latest_date}  /  取得日: {today_date}  /  差分: {days_diff}日")
    if failed_codes:
        print(f"  除外銘柄: {', '.join(failed_codes)}")

    # --- 期間別相関行列 ---
    print("\n▼ 相関行列算出...")
    periods_out: dict[str, dict] = {}
    corr_50     = None
    overall_status = "complete"

    for n in PERIODS:
        if common_returns < n:
            msg = f"[警告] {n}日相関: リターン本数({common_returns})が不足。スキップ。"
            alerts.append(msg)
            print(f"  {msg}")
            overall_status = "partial"
            continue

        recent = returns_df.iloc[-n:]
        corr   = recent.corr()

        # 分布統計（対角除く）
        mask = ~np.eye(len(corr), dtype=bool)
        vals = corr.values[mask].astype(float)
        q25, q50, q75 = np.percentile(vals, [25, 50, 75])

        pair_high, pair_low = build_pair_ranking(corr, top_n=10)

        periods_out[str(n)] = {
            "corr_matrix": {
                row: {col: round(float(corr.loc[row, col]), 4) for col in corr.columns}
                for row in corr.index
            },
            "stats": {
                "mean_corr": round(float(np.mean(vals)), 4),
                "min":       round(float(np.min(vals)), 4),
                "q25":       round(float(q25), 4),
                "median":    round(float(q50), 4),
                "q75":       round(float(q75), 4),
                "max":       round(float(np.max(vals)), 4),
                "neg_count": int((vals < 0).sum()),
            },
            "pair_ranking_high": pair_high,
            "pair_ranking_low":  pair_low,
        }

        if n == 50:
            corr_50 = corr

        print(f"  {n}日: 平均相関={np.mean(vals):.4f}  min={np.min(vals):.4f}  max={np.max(vals):.4f}")

    # --- クラスタ並び替え（50日基準・貪欲法） ---
    if corr_50 is not None:
        cluster_order = greedy_cluster_order(corr_50)
    else:
        cluster_order = [s["code"] for s in STOCKS if s["code"] in close_dict]

    print(f"\n▼ クラスタ順（50日基準）: {' > '.join(cluster_order)}")

    if failed_codes:
        overall_status = "partial"

    # --- data.json 構築 ---
    output = {
        "_meta": {
            "schema_version": "1.0",
            "generated_at":   to_iso_jst(generated_at),
            "overall_status": overall_status,
            "_status_vocab": "complete=全銘柄・全期間正常 / partial=一部除外または期間不足 / failed=計算不能",
        },
        "summary": {
            "fetch_date":  today_str,
            "common_days": common_days,
            "stocks": [
                {
                    "code":         s["code"],
                    "name":         s["name"],
                    "return_count": int(returns_df[s["code"]].notna().sum())
                                    if s["code"] in returns_df.columns else None,
                    "failed":       s["code"] in failed_codes,
                }
                for s in STOCKS
            ],
        },
        "cluster_order": cluster_order,
        "periods":       periods_out,
        "alerts":        {"messages": alerts},
    }

    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    label = {"complete": "OK", "partial": "WARN"}.get(overall_status, overall_status)
    print(f"\n[{label}] data.json 書き出し完了  overall_status={overall_status}  date={today_str}")
    if alerts:
        print("--- alerts ---")
        for a in alerts:
            print(f"  {a}")


if __name__ == "__main__":
    build_data()
