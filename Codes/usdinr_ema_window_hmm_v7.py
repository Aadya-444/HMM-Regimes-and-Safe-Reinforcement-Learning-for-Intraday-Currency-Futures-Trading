#!/usr/bin/env python3
"""USD/INR 30-minute regime forecasting with a windowed EMA hidden Markov model (final model, V7).

Self-contained: data loading, session grid and activity audit, features, the volatility-band
label rule, the 3-state Gaussian HMM with causal forward filtering, evaluation (day-block
bootstrap, permutation test, baselines) and the per-bar prediction exports all live here.

Usage
    python3 usdinr_ema_window_hmm_v7.py                 # holdout evaluation -> holdout_metrics.csv, two_forecasts_every_30min.csv
    python3 usdinr_ema_window_hmm_v7.py spans           # one HMM per EMA span (10, 21) -> two_ema_spans_every_30min.csv, comparison_metrics.csv
    python3 usdinr_ema_window_hmm_v7.py span10          # every bar of the 54 active days -> span10_predictions_every_30min.csv
    python3 usdinr_ema_window_hmm_v7.py all6months      # every bar of all 82 dates -> span10_predictions_every_30min_ALL_6_months.csv (RL input)
    python3 usdinr_ema_window_hmm_v7.py all             # holdout + all exports
    python3 usdinr_ema_window_hmm_v7.py diagnostics     # training curves, model selection, calibration, docs -> outputs/diagnostics, outputs/docs

Pipeline
    Kite 30-minute candles (stitched most-active contract) -> 16-slot session grid with carry-forward flags
    -> activity filter (>= 12 of 16 bars present, >= 8 traded) -> causal features -> label: next move vs
    band = max(2 ticks, rolling one-third quantile of |recent moves|) -> EMA(10 bars) of the close ->
    3-state Gaussian HMM on the EMA change, fitted on the first 80% of dates, forward-filtered causally
    -> next-bar regime probabilities -> prior-adjusted argmax -> scored on unseen dates.
"""

from __future__ import annotations

import logging
import os
import sys
import warnings
from pathlib import Path
from typing import List, Sequence, Tuple

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
warnings.filterwarnings("ignore")
logging.getLogger("hmmlearn.base").setLevel(logging.ERROR)

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from sklearn.metrics import balanced_accuracy_score, log_loss

# ----------------------------------------------------------------------------- constants
TIME_ZONE = "Asia/Kolkata"
TICK_SIZE = 0.0025
BAR_MINUTES = 30
BARS_PER_SESSION = 16
SESSION_START = "09:00"
SESSION_END = "17:00"
REGIMES = ("bearish", "sideways", "bullish")
REGIME_TO_ID = {name: index for index, name in enumerate(REGIMES)}
INTRADAY_HORIZONS = {"next_30_minutes": 1, "next_1_hour": 2, "next_2_hours": 4}
SIDEWAYS_TICKS = 2.0
SIDEWAYS_QUANTILE = 1.0 / 3.0
VOL_LOOKBACK_BARS = 80
HMM_FEATURE_COLUMNS = ["bar_log_return", "log_range", "realized_vol_8"]   # kept for the activity/feature audit

SOURCE = Path("kite_usdinr_30minute_6_months/USDINR_30minute_continuous.csv")
OUT = Path("ema_window_hmm_v7_outputs")
EMA_SPAN = 10
WINDOWS = (10, 21)
RET, BAND = "next_30_minutes_future_log_return", "next_30_minutes_band"


def _parse_timestamp(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    return parsed.dt.tz_convert(TIME_ZONE)


def _regularize_one_day(day: pd.DataFrame) -> pd.DataFrame:
    """Place one trading date on the 16-slot 30-minute session grid."""
    day = day.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    trading_date = pd.Timestamp(day["timestamp"].iloc[0]).normalize()
    grid = pd.date_range(
        trading_date + pd.Timedelta(SESSION_START + ":00"),
        trading_date + pd.Timedelta(SESSION_END + ":00") - pd.Timedelta(minutes=BAR_MINUTES),
        freq=f"{BAR_MINUTES}min",
    )
    indexed = day.set_index("timestamp").reindex(grid)
    indexed.index.name = "timestamp"
    indexed["is_observed"] = indexed["close"].notna().astype(int)
    indexed["is_traded"] = (indexed["volume"].fillna(0) > 0).astype(int)
    indexed["close"] = indexed["close"].ffill().bfill()
    for column in ("open", "high", "low"):
        indexed[column] = indexed[column].where(indexed["is_observed"].eq(1), indexed["close"])
    indexed["volume"] = indexed["volume"].fillna(0.0)
    indexed["oi"] = indexed["oi"].ffill().bfill().fillna(0.0)
    for column in ("tradingsymbol", "contract_expiry"):
        if column in indexed:
            indexed[column] = indexed[column].ffill().bfill()
    indexed["trading_date"] = str(day["trading_date"].iloc[0])
    return indexed.reset_index()


def load_30min_bars(
    source_csv: Path,
    min_observed_fraction: float = 0.75,
    min_traded_bars: int = 8,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load the stitched 30-minute series and keep only active trading dates.

    Returns (active frame, per-day audit table, profile dictionary).
    """
    raw = pd.read_csv(source_csv)
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Source is missing required columns: {sorted(missing)}")
    if "oi" not in raw:
        raw["oi"] = 0.0
    raw["timestamp"] = _parse_timestamp(raw["date"])
    for column in ("open", "high", "low", "close", "volume", "oi"):
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    raw = raw.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"])
    session = raw["timestamp"].dt.time.ge(pd.Timestamp(SESSION_START).time()) & raw["timestamp"].dt.time.lt(
        pd.Timestamp(SESSION_END).time()
    )
    raw = raw.loc[session].copy()
    raw["trading_date"] = raw["timestamp"].dt.date.astype(str)

    grid = pd.concat(
        [_regularize_one_day(day) for _, day in raw.groupby("trading_date", sort=True)],
        ignore_index=True,
    )
    audit = (
        grid.groupby("trading_date")
        .agg(
            bars_observed=("is_observed", "sum"),
            bars_traded=("is_traded", "sum"),
            volume=("volume", "sum"),
            tradingsymbol=("tradingsymbol", "first"),
        )
        .reset_index()
    )
    audit["observed_fraction"] = audit["bars_observed"] / BARS_PER_SESSION
    audit["traded_fraction"] = audit["bars_traded"] / BARS_PER_SESSION
    audit["is_active"] = (audit["observed_fraction"] >= min_observed_fraction) & (
        audit["bars_traded"] >= min_traded_bars
    )
    active_dates = set(audit.loc[audit["is_active"], "trading_date"])
    frame = grid.loc[grid["trading_date"].isin(active_dates)].copy().reset_index(drop=True)
    frame = frame.merge(audit[["trading_date", "traded_fraction"]], on="trading_date", how="left")
    frame["day_weight"] = frame["traded_fraction"].clip(lower=0.25)

    profile = {
        "source_csv": str(Path(source_csv).resolve()),
        "input_rows": int(len(raw)),
        "all_trading_dates": int(audit.shape[0]),
        "active_trading_dates": int(len(active_dates)),
        "dropped_trading_dates": int(audit.shape[0] - len(active_dates)),
        "active_bars": int(len(frame)),
        "active_bars_observed_fraction": float(frame["is_observed"].mean()) if len(frame) else float("nan"),
        "first_active_date": min(active_dates) if active_dates else "",
        "last_active_date": max(active_dates) if active_dates else "",
        "min_observed_fraction": min_observed_fraction,
        "min_traded_bars": min_traded_bars,
        "contracts": sorted(frame["tradingsymbol"].dropna().unique().tolist()) if "tradingsymbol" in frame else [],
    }
    return frame, audit, profile


def add_bar_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Causal bar features.  Returns are within-day so no overnight gap leaks in."""
    frame = frame.sort_values("timestamp").reset_index(drop=True).copy()
    by_day = frame.groupby("trading_date", sort=False)
    previous_close = by_day["close"].shift(1)
    frame["bar_log_return"] = np.log(frame["close"] / previous_close.fillna(frame["open"]))
    frame["log_range"] = np.log(frame["high"] / frame["low"]).clip(lower=0.0)
    frame["log_volume"] = np.log1p(frame["volume"].clip(lower=0))
    frame["oi_change"] = (
        by_day["oi"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )
    # Continuous across days: regimes persist overnight, so these use the full series.
    returns = frame["bar_log_return"]
    for window in (8, 16):
        frame[f"realized_vol_{window}"] = returns.rolling(window, min_periods=4).std(ddof=0)
    for window in (1, 2, 4, 8, 16):
        frame[f"momentum_{window}"] = returns.rolling(window, min_periods=1).sum()
    frame["price_vs_mean_16"] = np.log(frame["close"] / frame["close"].rolling(16, min_periods=1).mean())
    volume_mean = frame["log_volume"].rolling(16, min_periods=4).mean()
    volume_std = frame["log_volume"].rolling(16, min_periods=4).std(ddof=0)
    frame["volume_zscore_16"] = ((frame["log_volume"] - volume_mean) / volume_std.replace(0, np.nan)).fillna(0.0)
    frame["traded_fraction_8"] = frame["is_traded"].rolling(8, min_periods=1).mean()
    slot = (frame["timestamp"].dt.hour * 60 + frame["timestamp"].dt.minute - 9 * 60) / (BARS_PER_SESSION * BAR_MINUTES)
    frame["session_sin"] = np.sin(2 * np.pi * slot)
    frame["session_cos"] = np.cos(2 * np.pi * slot)
    frame["bar_index"] = by_day.cumcount()
    return frame


def add_intraday_targets(frame: pd.DataFrame) -> pd.DataFrame:
    """Future returns within the same session, plus the causal sideways band."""
    frame = frame.copy()
    by_day = frame.groupby("trading_date", sort=False)
    tick_return = SIDEWAYS_TICKS * TICK_SIZE / frame["close"]
    for name, bars in INTRADAY_HORIZONS.items():
        future_close = by_day["close"].shift(-bars)
        frame[f"{name}_future_close"] = future_close
        frame[f"{name}_future_timestamp"] = by_day["timestamp"].shift(-bars)
        frame[f"{name}_future_log_return"] = np.log(future_close / frame["close"])
        frame[f"{name}_future_traded"] = by_day["is_traded"].shift(-bars)
        # Causal scale of horizon-length moves: the rolling one-third quantile of past
        # absolute horizon returns, so by construction about a third of moves are "sideways".
        past_horizon_return = np.log(frame["close"] / frame["close"].shift(bars)).abs()
        scale = past_horizon_return.rolling(VOL_LOOKBACK_BARS, min_periods=16).quantile(SIDEWAYS_QUANTILE)
        frame[f"{name}_band"] = np.maximum(scale, tick_return)
    return frame


def classify(returns: np.ndarray, band: np.ndarray) -> np.ndarray:
    return np.where(returns > band, REGIME_TO_ID["bullish"], np.where(returns < -band, REGIME_TO_ID["bearish"], REGIME_TO_ID["sideways"])).astype(int)


def block_bootstrap_ci(
    dates: np.ndarray,
    actual: np.ndarray,
    predicted: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
) -> Tuple[float, float]:
    """Resample whole trading dates to respect within-day dependence."""
    rng = np.random.default_rng(seed)
    unique = np.unique(dates)
    groups = {d: np.flatnonzero(dates == d) for d in unique}
    scores = []
    for _ in range(n_boot):
        pick = rng.choice(unique, size=len(unique), replace=True)
        index = np.concatenate([groups[d] for d in pick])
        if np.unique(actual[index]).size < 2:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores.append(balanced_accuracy_score(actual[index], predicted[index]))
    if not scores:
        return float("nan"), float("nan")
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def permutation_p_value(actual: np.ndarray, predicted: np.ndarray, n_perm: int = 2000, seed: int = 0) -> float:
    """Probability of the observed balanced accuracy if predictions were unrelated to labels."""
    rng = np.random.default_rng(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        observed = balanced_accuracy_score(actual, predicted)
        null = np.array([balanced_accuracy_score(actual, rng.permutation(predicted)) for _ in range(n_perm)])
    return float((np.sum(null >= observed) + 1) / (n_perm + 1))


def split(frame: pd.DataFrame, train_fraction: float = 0.8):
    dates = sorted(frame["trading_date"].unique())
    cut = int(round(len(dates) * train_fraction))
    return (
        frame.loc[frame["trading_date"].isin(dates[:cut])].reset_index(drop=True),
        frame.loc[frame["trading_date"].isin(dates[cut:])].reset_index(drop=True),
    )


def score(name, actual, predicted, dates, probabilities=None, n_boot=1000):
    low, high = block_bootstrap_ci(dates, actual, predicted, n_boot=n_boot)
    row = {"model": name, "windows": int(len(actual)), "dates": int(np.unique(dates).size),
           "balanced_accuracy": balanced_accuracy_score(actual, predicted), "ci_low": low, "ci_high": high,
           "permutation_p": permutation_p_value(actual, predicted), "accuracy": float((actual == predicted).mean()),
           "predicted_classes": int(np.unique(predicted).size)}
    for k, r in enumerate(REGIMES):
        row[f"recall_{r}"] = float((predicted[actual == k] == k).mean()) if (actual == k).any() else np.nan
    row["log_loss"] = log_loss(actual, np.clip(probabilities, 1e-6, 1), labels=[0, 1, 2]) if probabilities is not None else np.log(3)
    return row


def error_columns(frame: pd.DataFrame, pred_col: str, actual_col: str) -> pd.DataFrame:
    pred, act = frame[pred_col], frame[actual_col]
    d = {"bullish", "bearish"}
    frame["correct"] = (pred == act).astype(int)
    frame["opposite_direction"] = (((pred == "bullish") & (act == "bearish")) | ((pred == "bearish") & (act == "bullish"))).astype(int)
    frame["predicted_direction_actual_sideways"] = (pred.isin(d) & (act == "sideways")).astype(int)
    frame["predicted_sideways_actual_direction"] = ((pred == "sideways") & act.isin(d)).astype(int)
    frame["error_type"] = "correct"
    for c in ("opposite_direction", "predicted_direction_actual_sideways", "predicted_sideways_actual_direction"):
        frame.loc[frame[c].eq(1), "error_type"] = c
    return frame


def add_ema_stream(bars: pd.DataFrame, span: int = EMA_SPAN) -> pd.DataFrame:
    f = bars.sort_values("timestamp").reset_index(drop=True).copy()
    f["ema"] = f["close"].ewm(span=span, adjust=False).mean()
    f["ema_change_bps"] = np.log(f["ema"] / f["ema"].shift(1)).fillna(0) * 1e4
    f["time"] = f["timestamp"].dt.strftime("%H:%M")
    return f


class WindowHMM:
    def __init__(self, n_states=3, seeds=(7, 21, 42, 99)):
        self.n_states, self.seeds = n_states, seeds

    def fit(self, train: pd.DataFrame) -> "WindowHMM":
        x = train[["ema_change_bps"]].to_numpy()
        self.scale_ = x.std() + 1e-9
        best, best_score = None, -np.inf
        for seed in self.seeds:
            m = GaussianHMM(n_components=self.n_states, covariance_type="diag", n_iter=500, tol=1e-4, min_covar=1e-3, random_state=seed)
            try:
                m.fit(x / self.scale_)
            except Exception:
                continue
            s = m.score(x / self.scale_)
            if s > best_score:
                best, best_score = m, s
        self.model_ = best
        self.order_ = np.argsort(best.means_[:, 0])
        self.means_bps_ = best.means_[self.order_, 0] * self.scale_
        self.transmat_ = best.transmat_[np.ix_(self.order_, self.order_)]
        # state -> next-bar regime, learned on training bars using the full causal filter
        post = self.filter_window(train, window=None)
        ok = (train[RET].notna() & train[BAND].notna()).to_numpy()
        labels = classify(train.loc[ok, RET].to_numpy(), train.loc[ok, BAND].to_numpy())
        soft = post[ok]
        table = np.ones((self.n_states, 3))
        for s in range(self.n_states):
            counts = np.array([soft[labels == k, s].sum() for k in range(3)])
            table[s] = (counts + 1) / (counts.sum() + 3)
        self.state_to_next_ = table
        self.prior_ = (np.bincount(labels, minlength=3) + 1) / (len(labels) + 3)
        return self

    def filter_window(self, frame: pd.DataFrame, window: int | None) -> np.ndarray:
        """P(state_t | last `window` EMA observations).  window=None -> full history."""
        em = self.model_._compute_log_likelihood(frame[["ema_change_bps"]].to_numpy() / self.scale_)[:, self.order_]
        log_start = np.log(np.clip(self.model_.startprob_[self.order_], 1e-300, None))
        log_trans = np.log(np.clip(self.transmat_, 1e-300, None))
        n = len(frame)
        out = np.empty((n, self.n_states))
        if window is None:
            alpha = log_start + em[0]; alpha -= logsumexp(alpha); out[0] = np.exp(alpha)
            for t in range(1, n):
                alpha = em[t] + logsumexp(alpha[:, None] + log_trans, axis=0); alpha -= logsumexp(alpha); out[t] = np.exp(alpha)
            return out
        for t in range(n):
            start = max(0, t - window + 1)              # exactly `window` observations: start..t
            alpha = log_start + em[start]; alpha -= logsumexp(alpha)
            for u in range(start + 1, t + 1):
                alpha = em[u] + logsumexp(alpha[:, None] + log_trans, axis=0); alpha -= logsumexp(alpha)
            out[t] = np.exp(alpha)
        return out

    def predict_next(self, frame: pd.DataFrame, window: int | None):
        filtered = self.filter_window(frame, window)
        probs = (filtered @ self.transmat_) @ self.state_to_next_
        return probs, (probs / self.prior_).argmax(1), filtered   # prior-adjusted argmax so all classes can appear


def _active_frame(span: int):
    bars, _, _ = load_30min_bars(SOURCE, 0.75, 8)
    return bars, add_ema_stream(add_intraday_targets(add_bar_features(bars)), span=span)


def _test_mask(test):
    return (test[RET].notna() & test[BAND].notna() & test["next_30_minutes_future_traded"].fillna(0).gt(0)).to_numpy()


def export_spans(n_boot: int = 1000) -> None:
    """One HMM per EMA span (10 and 21 bars); compare and dump per-bar forecasts."""
    rows, out = [], None
    for span in (10, 21):
        _, f = _active_frame(span)
        train, test = split(f)
        hmm = WindowHMM().fit(train)
        ok = _test_mask(test)
        act = classify(test.loc[ok, RET].to_numpy(), test.loc[ok, BAND].to_numpy())
        dates = test.loc[ok, "trading_date"].to_numpy()
        p, pred, _ = hmm.predict_next(test, None)
        rows.append(score(f"EMA span {span} bars, own HMM", act, pred[ok], dates, p[ok], n_boot))
        if out is None:
            out = test.loc[ok, ["timestamp", "trading_date", "time", "close"]].copy()
            out["actual_next_30min"] = [REGIMES[i] for i in act]
        out[f"ema_span{span}"] = test.loc[ok, "ema"].round(4).to_numpy()
        out[f"pred_ema{span}"] = [REGIMES[i] for i in pred[ok]]
        out[f"correct_ema{span}"] = (pred[ok] == act).astype(int)
    prev = pd.read_csv(OUT / "holdout_metrics.csv")
    prev = prev[prev["model"].isin(["ema_hmm_last_10_obs", "ema_hmm_last_21_obs", "baseline_last_bar_continues", "baseline_always_bullish"])].copy()
    prev["model"] = prev["model"].replace({"ema_hmm_last_10_obs": "EMA span 10, filter window 10 (previous)", "ema_hmm_last_21_obs": "EMA span 10, filter window 21 (previous)"})
    out["agree"] = (out["pred_ema10"] == out["pred_ema21"]).astype(int)
    out.to_csv(OUT / "two_ema_spans_every_30min.csv", index=False)
    pd.concat([pd.DataFrame(rows), prev], ignore_index=True).to_csv(OUT / "comparison_metrics.csv", index=False)
    print("wrote two_ema_spans_every_30min.csv, comparison_metrics.csv")


def _prediction_table(f: pd.DataFrame, hmm: WindowHMM, first_test, train_dates: set, active_dates: set | None) -> pd.DataFrame:
    probs, pred, _ = hmm.predict_next(f, 10)   # each forecast sees exactly the last 10 EMA observations
    out = pd.DataFrame({
        "timestamp": f["timestamp"], "trading_date": f["trading_date"], "time": f["time"], "close": f["close"],
        "ema_span10": f["ema"].round(4), "ema_observations_seen": np.minimum(np.arange(1, len(f) + 1), 10),
    })
    if active_dates is not None:
        out.insert(3, "contract", f["tradingsymbol"])
        out["bar_traded"] = f["is_traded"]
        out["day_active_for_training"] = f["trading_date"].isin(active_dates).astype(int)
        out["split"] = np.where(f["timestamp"] >= first_test, "test (unseen)",
                                np.where(f["trading_date"].isin(train_dates), "train (in-sample)", "inactive (not used for training)"))
    else:
        out["split"] = np.where(f["timestamp"] >= first_test, "test (unseen)", "train (in-sample)")
    labelled = f[RET].notna() & f[BAND].notna()
    real = np.where(labelled, classify(f[RET].fillna(0).to_numpy(), f[BAND].fillna(1).to_numpy()), -1)
    out["real_situation_next_30min"] = [REGIMES[i] if i >= 0 else "" for i in real]
    out["real_next_move_bps"] = (f[RET] * 1e4).round(2)
    if active_dates is not None:
        out["next_bar_traded"] = f["next_30_minutes_future_traded"].fillna(0).astype(int)
    out["predicted_situation_next_30min"] = [REGIMES[i] for i in pred]
    for k, r in enumerate(REGIMES):
        out[f"p_{r}"] = probs[:, k].round(3)
    out.loc[:9, ["predicted_situation_next_30min", "p_bearish", "p_sideways", "p_bullish"]] = ""   # first 10 bars: nothing to look back on
    has = (out["predicted_situation_next_30min"] != "") & (out["real_situation_next_30min"] != "")
    out["correct"] = ""
    out.loc[has, "correct"] = (out.loc[has, "predicted_situation_next_30min"] == out.loc[has, "real_situation_next_30min"]).astype(int)
    return out


def export_span10() -> None:
    """Span-10 model, every 30-minute bar of the 54 active days."""
    _, f = _active_frame(10)
    train, test = split(f)
    hmm = WindowHMM().fit(train)
    out = _prediction_table(f, hmm, test["timestamp"].min(), set(train["trading_date"]), None)
    out.to_csv(OUT / "span10_predictions_every_30min.csv", index=False)
    print(f"wrote span10_predictions_every_30min.csv ({len(out)} rows)")


def export_all_6_months() -> None:
    """Same trained span-10 model applied to every date Kite returned (no activity filter)."""
    act_bars, fa = _active_frame(10)
    train, test = split(fa)
    hmm = WindowHMM().fit(train)
    all_bars, _, _ = load_30min_bars(SOURCE, 0.0, 0)
    f = add_ema_stream(add_intraday_targets(add_bar_features(all_bars)), span=10)
    out = _prediction_table(f, hmm, test["timestamp"].min(), set(train["trading_date"]), set(act_bars["trading_date"]))
    out.to_csv(OUT / "span10_predictions_every_30min_ALL_6_months.csv", index=False)
    print(f"wrote span10_predictions_every_30min_ALL_6_months.csv ({len(out)} rows, {out['trading_date'].nunique()} dates)")


def main(n_boot: int = 1000):
    OUT.mkdir(exist_ok=True)
    bars, _, profile = load_30min_bars(SOURCE, 0.75, 8)
    frame = add_ema_stream(add_intraday_targets(add_bar_features(bars)))
    train, test = split(frame)
    hmm = WindowHMM().fit(train)
    print(f"EMA span {EMA_SPAN} bars; train {train['trading_date'].nunique()} dates, test {test['trading_date'].nunique()} dates")
    print("hidden states, mean EMA change bps:", np.round(hmm.means_bps_, 3), " self-transition:", np.round(np.diag(hmm.transmat_), 3))
    print("state -> next regime (bear/side/bull):\n", np.round(hmm.state_to_next_, 3))

    ok = (test[RET].notna() & test[BAND].notna() & test["next_30_minutes_future_traded"].fillna(0).gt(0)).to_numpy()
    actual = classify(test.loc[ok, RET].to_numpy(), test.loc[ok, BAND].to_numpy())
    dates = test.loc[ok, "trading_date"].to_numpy()
    b = test.loc[ok, BAND].to_numpy()
    out = test.loc[ok, ["timestamp", "trading_date", "time", "close", "ema", "ema_change_bps"]].copy()
    out["actual_next_30min"] = [REGIMES[i] for i in actual]
    out["actual_next_move_bps"] = (test.loc[ok, RET] * 1e4).round(2).to_numpy()
    out["band_bps"] = (b * 1e4).round(2)
    rows = []
    for w in WINDOWS:
        probs, pred, _ = hmm.predict_next(test, w)
        out[f"pred_last{w}"] = [REGIMES[i] for i in pred[ok]]
        for k, r in enumerate(REGIMES):
            out[f"p_{r}_last{w}"] = probs[ok][:, k].round(3)
        out[f"correct_last{w}"] = (pred[ok] == actual).astype(int)
        out[f"opposite_last{w}"] = (((pred[ok] == 0) & (actual == 2)) | ((pred[ok] == 2) & (actual == 0))).astype(int)
        rows.append(score(f"ema_hmm_last_{w}_obs", actual, pred[ok], dates, probs[ok], n_boot))
    probs_full, pred_full, _ = hmm.predict_next(test, None)
    rows.append(score("ema_hmm_full_history", actual, pred_full[ok], dates, probs_full[ok], n_boot))
    last_bar = classify(test.loc[ok, "bar_log_return"].to_numpy(), b)
    rows.append(score("baseline_last_bar_continues", actual, last_bar, dates, None, n_boot))
    rows.append(score("baseline_always_bullish", actual, np.full(len(actual), 2), dates, None, n_boot))
    out["agree"] = (out["pred_last10"] == out["pred_last21"]).astype(int)
    metrics = pd.DataFrame(rows)
    out.to_csv(OUT / "two_forecasts_every_30min.csv", index=False)
    metrics.to_csv(OUT / "holdout_metrics.csv", index=False)
    pd.set_option("display.width", 250)
    print(metrics[["model", "windows", "balanced_accuracy", "ci_low", "ci_high", "permutation_p", "accuracy", "predicted_classes", "recall_bearish", "recall_sideways", "recall_bullish"]].round(3).to_string(index=False))
    print(f"\nthe two forecasts agree on {out['agree'].mean():.0%} of bars; when they agree accuracy is {out.loc[out.agree.eq(1),'correct_last10'].mean():.1%} (n={out.agree.sum()})")
    print(out.head(16)[["trading_date", "time", "ema", "ema_change_bps", "pred_last10", "pred_last21", "actual_next_30min", "correct_last10", "correct_last21"]].to_string(index=False))


# =============================================================================
# Training diagnostics, model-selection studies and explanatory documents
# =============================================================================
DIAG = OUT / "diagnostics"
DOCS = OUT / "docs"
N_STATES_DEFAULT = 3
COLORS = {"bearish": "#D64B40", "sideways": "#7A8491", "bullish": "#16865C"}
STATE_COLORS = ["#D64B40", "#7A8491", "#16865C", "#5278A8", "#C99A2E", "#8E5EA2"]


def _prep(span: int = EMA_SPAN):
    bars, _, profile = load_30min_bars(SOURCE, 0.75, 8)
    frame = add_ema_stream(add_intraday_targets(add_bar_features(bars)), span=span)
    train, test = split(frame)
    return frame, train, test, profile


def _mask_and_labels(d: pd.DataFrame):
    ok = (d[RET].notna() & d[BAND].notna() & d["next_30_minutes_future_traded"].fillna(0).gt(0)).to_numpy()
    labels = classify(d.loc[ok, RET].to_numpy(), d.loc[ok, BAND].to_numpy())
    return ok, labels


def _wrap(model: GaussianHMM, scale: float, train: pd.DataFrame) -> "WindowHMM":
    """Turn a fitted hmmlearn model into a WindowHMM (same post-processing as WindowHMM.fit)."""
    w = WindowHMM(n_states=model.n_components)
    w.scale_, w.model_ = scale, model
    w.order_ = np.argsort(model.means_[:, 0])
    w.means_bps_ = model.means_[w.order_, 0] * scale
    w.transmat_ = model.transmat_[np.ix_(w.order_, w.order_)]
    post = w.filter_window(train, window=None)
    ok = (train[RET].notna() & train[BAND].notna()).to_numpy()
    labels = classify(train.loc[ok, RET].to_numpy(), train.loc[ok, BAND].to_numpy())
    soft = post[ok]
    table = np.ones((w.n_states, 3))
    for st in range(w.n_states):
        counts = np.array([soft[labels == k, st].sum() for k in range(3)])
        table[st] = (counts + 1) / (counts.sum() + 3)
    w.state_to_next_ = table
    w.prior_ = (np.bincount(labels, minlength=3) + 1) / (len(labels) + 3)
    return w


def _bal_acc(w: "WindowHMM", d: pd.DataFrame) -> float:
    ok, labels = _mask_and_labels(d)
    _, pred, _ = w.predict_next(d, None)
    return float(balanced_accuracy_score(labels, pred[ok]))


def _n_params(k: int) -> int:
    # transition rows (k*(k-1)) + start (k-1) + means (k) + diagonal variances (k), 1-dimensional observation
    return k * (k - 1) + (k - 1) + 2 * k


def em_trace(train, test, seed: int, n_states: int = N_STATES_DEFAULT, n_iter: int = 60) -> pd.DataFrame:
    """Run Baum-Welch one iteration at a time and record what changed after each iteration."""
    x = train[["ema_change_bps"]].to_numpy()
    scale = x.std() + 1e-9
    xs, xt = x / scale, test[["ema_change_bps"]].to_numpy() / scale
    m = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=1, tol=-1.0, min_covar=1e-3, random_state=seed)
    rows = []
    prev = None
    for it in range(1, n_iter + 1):
        if it > 1:
            m.init_params = ""
        m.fit(xs)
        ll_tr, ll_te = m.score(xs) / len(xs), m.score(xt) / len(xt)
        w = _wrap(m, scale, train)
        rows.append({
            "iteration": it, "seed": seed,
            "train_loglik_per_bar": ll_tr, "test_loglik_per_bar": ll_te,
            "improvement": (ll_tr - prev) if prev is not None else np.nan,
            "train_balanced_accuracy": _bal_acc(w, train), "test_balanced_accuracy": _bal_acc(w, test),
            **{f"state{k}_mean_bps": w.means_bps_[k] for k in range(n_states)},
            **{f"state{k}_stay_prob": w.transmat_[k, k] for k in range(n_states)},
        })
        prev = ll_tr
    return pd.DataFrame(rows)


def _md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        cells = []
        for v in r:
            if isinstance(v, (bool, np.bool_)):
                cells.append(str(v))
            elif isinstance(v, (int, np.integer)):
                cells.append(str(int(v)))
            elif isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else (str(int(v)) if float(v).is_integer() and abs(v) >= 1 else floatfmt.format(v)))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def diagnostics(n_boot: int = 1000) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.2, "font.size": 10})
    DIAG.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    frame, train, test, profile = _prep()
    ok_tr, y_tr = _mask_and_labels(train)
    ok_te, y_te = _mask_and_labels(test)

    # ---------------------------------------------------------------- 1. data & labels
    share = pd.DataFrame({
        "split": ["train", "test"],
        "dates": [train["trading_date"].nunique(), test["trading_date"].nunique()],
        "scored_bars": [len(y_tr), len(y_te)],
        "bearish": [np.mean(y_tr == 0), np.mean(y_te == 0)],
        "sideways": [np.mean(y_tr == 1), np.mean(y_te == 1)],
        "bullish": [np.mean(y_tr == 2), np.mean(y_te == 2)],
        "median_band_bps": [train.loc[ok_tr, BAND].median() * 1e4, test.loc[ok_te, BAND].median() * 1e4],
        "median_abs_move_bps": [train.loc[ok_tr, RET].abs().median() * 1e4, test.loc[ok_te, RET].abs().median() * 1e4],
    })
    share.to_csv(DIAG / "label_shares.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    xpos = np.arange(3)
    ax.bar(xpos - 0.18, share.loc[0, ["bearish", "sideways", "bullish"]].astype(float), 0.36, label="train", color="#5278A8")
    ax.bar(xpos + 0.18, share.loc[1, ["bearish", "sideways", "bullish"]].astype(float), 0.36, label="test", color="#C99A2E")
    ax.axhline(1 / 3, ls=":", color="#4B4B4B"); ax.set_xticks(xpos, ["bearish", "sideways", "bullish"]); ax.set_ylabel("share of bars"); ax.legend()
    ax.set_title("Class shares of the next-30-minute label, train vs test")
    plt.tight_layout(); plt.savefig(DIAG / "01_label_shares.png", dpi=160); plt.close()

    # ---------------------------------------------------------------- 2. EM training curve (all seeds)
    traces = pd.concat([em_trace(train, test, seed) for seed in (7, 21, 42, 99)], ignore_index=True)
    traces.to_csv(DIAG / "training_curve_per_iteration.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    for seed, g in traces.groupby("seed"):
        axes[0].plot(g["iteration"], g["train_loglik_per_bar"], label=f"seed {seed} train")
        axes[0].plot(g["iteration"], g["test_loglik_per_bar"], ls="--", alpha=0.7, label=f"seed {seed} test")
        axes[1].plot(g["iteration"], g["improvement"].abs().clip(lower=1e-9), label=f"seed {seed}")
        axes[2].plot(g["iteration"], g["test_balanced_accuracy"], label=f"seed {seed} test")
        axes[2].plot(g["iteration"], g["train_balanced_accuracy"], ls="--", alpha=0.6, label=f"seed {seed} train")
    axes[0].set_title("Log-likelihood per bar (higher = better fit)"); axes[0].set_xlabel("EM iteration"); axes[0].legend(fontsize=7, ncol=2)
    axes[1].set_yscale("log"); axes[1].set_title("Improvement per iteration (train log-likelihood)"); axes[1].set_xlabel("EM iteration"); axes[1].legend(fontsize=7)
    axes[2].axhline(1 / 3, ls=":", color="#4B4B4B"); axes[2].set_ylim(0.2, 0.6); axes[2].set_title("Balanced accuracy of the next-bar forecast"); axes[2].set_xlabel("EM iteration"); axes[2].legend(fontsize=7, ncol=2)
    plt.tight_layout(); plt.savefig(DIAG / "02_training_curve.png", dpi=160); plt.close()
    best_seed_trace = traces[traces["seed"] == 7]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for k in range(N_STATES_DEFAULT):
        ax.plot(best_seed_trace["iteration"], best_seed_trace[f"state{k}_mean_bps"], label=f"state {k} mean EMA change (bps)", color=STATE_COLORS[k])
    ax.set_xlabel("EM iteration"); ax.set_title("How the three hidden states separate during training (seed 7)"); ax.legend()
    plt.tight_layout(); plt.savefig(DIAG / "03_state_means_per_iteration.png", dpi=160); plt.close()

    # ---------------------------------------------------------------- 3. random restarts (final models)
    seed_rows = []
    for seed in (7, 21, 42, 99):
        g = traces[traces["seed"] == seed].iloc[-1]
        seed_rows.append({"seed": seed, "final_train_loglik_per_bar": g["train_loglik_per_bar"], "final_test_loglik_per_bar": g["test_loglik_per_bar"],
                          "train_balanced_accuracy": g["train_balanced_accuracy"], "test_balanced_accuracy": g["test_balanced_accuracy"],
                          "iterations_to_converge": int(((traces[traces["seed"] == seed]["train_loglik_per_bar"] >= g["train_loglik_per_bar"] - 1e-3).to_numpy()).argmax() + 1)})
    seeds = pd.DataFrame(seed_rows)
    seeds.to_csv(DIAG / "random_restarts.csv", index=False)

    # ---------------------------------------------------------------- 4. number of states (AIC / BIC / held-out)
    sel_rows = []
    for k in (2, 3, 4, 5, 6):
        w = WindowHMM(n_states=k).fit(train)
        xs = train[["ema_change_bps"]].to_numpy() / w.scale_
        xt = test[["ema_change_bps"]].to_numpy() / w.scale_
        ll = w.model_.score(xs); n = len(xs); pcount = _n_params(k)
        sel_rows.append({"states": k, "parameters": pcount, "train_loglik_per_bar": ll / n, "test_loglik_per_bar": w.model_.score(xt) / len(xt),
                         "AIC": 2 * pcount - 2 * ll, "BIC": pcount * np.log(n) - 2 * ll,
                         "train_balanced_accuracy": _bal_acc(w, train), "test_balanced_accuracy": _bal_acc(w, test)})
    sel = pd.DataFrame(sel_rows); sel.to_csv(DIAG / "state_count_selection.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].plot(sel["states"], sel["AIC"], marker="o", label="AIC"); axes[0].plot(sel["states"], sel["BIC"], marker="s", label="BIC")
    axes[0].set_xlabel("number of hidden states"); axes[0].set_title("Information criteria (lower = better)"); axes[0].legend()
    axes[1].plot(sel["states"], sel["train_balanced_accuracy"], marker="o", label="train"); axes[1].plot(sel["states"], sel["test_balanced_accuracy"], marker="s", label="test")
    axes[1].axhline(1 / 3, ls=":", color="#4B4B4B"); axes[1].set_ylim(0.25, 0.6); axes[1].set_xlabel("number of hidden states"); axes[1].set_title("Balanced accuracy of the forecast"); axes[1].legend()
    plt.tight_layout(); plt.savefig(DIAG / "04_state_count_selection.png", dpi=160); plt.close()

    # ---------------------------------------------------------------- 5. EMA span sensitivity
    span_rows = []
    for span in (5, 10, 15, 21, 30):
        _, tr_s, te_s, _ = _prep(span)
        w = WindowHMM().fit(tr_s)
        ok_s, y_s = _mask_and_labels(te_s); _, pred_s, _ = w.predict_next(te_s, None)
        span_rows.append({"ema_span_bars": span, "train_balanced_accuracy": _bal_acc(w, tr_s), "test_balanced_accuracy": _bal_acc(w, te_s),
                          "permutation_p": permutation_p_value(y_s, pred_s[ok_s]), "memory_half_life_bars": float(np.log(0.5) / np.log(np.sort(np.abs(np.linalg.eigvals(w.transmat_)))[-2]))})
    spans = pd.DataFrame(span_rows); spans.to_csv(DIAG / "ema_span_sensitivity.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(spans["ema_span_bars"], spans["train_balanced_accuracy"], marker="o", label="train"); ax.plot(spans["ema_span_bars"], spans["test_balanced_accuracy"], marker="s", label="test")
    ax.axhline(1 / 3, ls=":", color="#4B4B4B"); ax.set_ylim(0.25, 0.6); ax.set_xlabel("EMA span (bars)"); ax.set_title("Effect of the EMA smoothing length"); ax.legend()
    plt.tight_layout(); plt.savefig(DIAG / "05_ema_span_sensitivity.png", dpi=160); plt.close()

    # ---------------------------------------------------------------- 6. final model: train vs test metrics, confusion, calibration, time of day
    hmm = WindowHMM().fit(train)
    p_tr, pred_tr, filt_tr = hmm.predict_next(train, None)
    p_te, pred_te, filt_te = hmm.predict_next(test, None)
    dates_tr, dates_te = train.loc[ok_tr, "trading_date"].to_numpy(), test.loc[ok_te, "trading_date"].to_numpy()
    m_tr = score("train (in-sample)", y_tr, pred_tr[ok_tr], dates_tr, p_tr[ok_tr], n_boot)
    m_te = score("test (unseen dates)", y_te, pred_te[ok_te], dates_te, p_te[ok_te], n_boot)
    metrics = pd.DataFrame([m_tr, m_te]); metrics.to_csv(DIAG / "train_vs_test_metrics.csv", index=False)

    from sklearn.metrics import confusion_matrix
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (name, y, pr) in zip(axes, [("train", y_tr, pred_tr[ok_tr]), ("test", y_te, pred_te[ok_te])]):
        cm = confusion_matrix(y, pr, labels=[0, 1, 2]); cmn = cm / cm.sum(1, keepdims=True)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cm[i, j]}\n({cmn[i, j]:.0%})", ha="center", va="center", color="white" if cmn[i, j] > 0.5 else "black", fontsize=9)
        ax.set_xticks(range(3), REGIMES); ax.set_yticks(range(3), REGIMES); ax.set_xlabel("predicted"); ax.set_ylabel("actual"); ax.set_title(f"Confusion matrix, {name}"); ax.grid(False)
    plt.tight_layout(); plt.savefig(DIAG / "06_confusion_matrices.png", dpi=160); plt.close()
    pd.DataFrame(confusion_matrix(y_te, pred_te[ok_te], labels=[0, 1, 2]), index=[f"actual_{r}" for r in REGIMES], columns=[f"pred_{r}" for r in REGIMES]).to_csv(DIAG / "confusion_matrix_test.csv")

    conf = p_te[ok_te].max(1); correct = (pred_te[ok_te] == y_te)
    bins = [0.33, 0.4, 0.45, 0.5, 0.6, 1.0]
    cal_rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel_b = (conf >= lo) & (conf < hi)
        if sel_b.sum():
            cal_rows.append({"confidence_bin": f"{lo:.2f}-{hi:.2f}", "bars": int(sel_b.sum()), "mean_confidence": float(conf[sel_b].mean()), "accuracy": float(correct[sel_b].mean())})
    cal = pd.DataFrame(cal_rows); cal.to_csv(DIAG / "calibration.csv", index=False)
    cov_rows = []
    for cutoff in (0.0, 0.4, 0.45, 0.5, 0.55, 0.6):
        keep = conf >= cutoff
        if keep.sum() >= 5:
            cov_rows.append({"min_confidence": cutoff, "coverage": float(keep.mean()), "bars": int(keep.sum()), "accuracy": float(correct[keep].mean()),
                             "balanced_accuracy": float(balanced_accuracy_score(y_te[keep], pred_te[ok_te][keep])) if np.unique(y_te[keep]).size > 1 else np.nan})
    cov = pd.DataFrame(cov_rows); cov.to_csv(DIAG / "coverage_accuracy.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].plot([0.3, 0.8], [0.3, 0.8], ls=":", color="#4B4B4B", label="perfect calibration"); axes[0].plot(cal["mean_confidence"], cal["accuracy"], marker="o", color="#16865C", label="model")
    for _, r in cal.iterrows(): axes[0].annotate(f"n={r['bars']}", (r["mean_confidence"], r["accuracy"]), textcoords="offset points", xytext=(4, 4), fontsize=8)
    axes[0].set_xlabel("model confidence (highest class probability)"); axes[0].set_ylabel("actual accuracy"); axes[0].set_title("Reliability diagram, test"); axes[0].legend()
    axes[1].plot(cov["coverage"], cov["accuracy"], marker="o", color="#5278A8"); axes[1].set_xlabel("share of bars where the model acts"); axes[1].set_ylabel("accuracy on those bars"); axes[1].set_title("Accuracy when only confident forecasts are used")
    plt.tight_layout(); plt.savefig(DIAG / "07_calibration_and_coverage.png", dpi=160); plt.close()

    tod = test.loc[ok_te, ["time"]].copy(); tod["correct"] = correct.astype(int); tod["pred"] = pred_te[ok_te]; tod["actual"] = y_te
    tod_tbl = tod.groupby("time").agg(bars=("correct", "size"), accuracy=("correct", "mean")).reset_index()
    tod_tbl.to_csv(DIAG / "accuracy_by_time_of_day.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.bar(tod_tbl["time"], tod_tbl["accuracy"], color="#5278A8"); ax.axhline(1 / 3, ls=":", color="#4B4B4B"); ax.set_ylim(0, 1); ax.tick_params(axis="x", rotation=45); ax.set_title("Test accuracy by time of day (11 bars per slot)")
    plt.tight_layout(); plt.savefig(DIAG / "08_accuracy_by_time_of_day.png", dpi=160); plt.close()

    # ---------------------------------------------------------------- 7. hidden states: transition matrix, timeline
    state_tbl = pd.DataFrame({"state": range(N_STATES_DEFAULT), "mean_ema_change_bps": hmm.means_bps_, "stay_probability": np.diag(hmm.transmat_),
                              "expected_run_bars": 1 / np.clip(1 - np.diag(hmm.transmat_), 1e-9, None),
                              "p_next_bearish": hmm.state_to_next_[:, 0], "p_next_sideways": hmm.state_to_next_[:, 1], "p_next_bullish": hmm.state_to_next_[:, 2],
                              "share_of_test_bars": [np.mean(filt_te.argmax(1) == k) for k in range(N_STATES_DEFAULT)]})
    state_tbl.to_csv(DIAG / "hidden_states.csv", index=False)
    pd.DataFrame(hmm.transmat_).to_csv(DIAG / "transition_matrix.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1, 1.4]})
    axes[0].imshow(hmm.transmat_, cmap="Blues", vmin=0, vmax=1); axes[0].grid(False)
    for i in range(3):
        for j in range(3): axes[0].text(j, i, f"{hmm.transmat_[i, j]:.0%}", ha="center", va="center", color="white" if hmm.transmat_[i, j] > 0.5 else "black")
    axes[0].set_xticks(range(3), [f"to {k}" for k in range(3)]); axes[0].set_yticks(range(3), [f"from {k}" for k in range(3)]); axes[0].set_title("Transition matrix (state 0 = falling EMA, 2 = rising EMA)")
    axes[1].imshow(hmm.state_to_next_, cmap="Greens", vmin=0.2, vmax=0.5); axes[1].grid(False)
    for i in range(3):
        for j in range(3): axes[1].text(j, i, f"{hmm.state_to_next_[i, j]:.1%}", ha="center", va="center")
    axes[1].set_xticks(range(3), [f"next {r}" for r in REGIMES]); axes[1].set_yticks(range(3), [f"state {k}" for k in range(3)]); axes[1].set_title("What each state says about the next bar")
    plt.tight_layout(); plt.savefig(DIAG / "09_transition_and_forecast_link.png", dpi=160); plt.close()

    fig, axes = plt.subplots(2, 1, figsize=(15, 7), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    hard = filt_te.argmax(1); t = test["timestamp"]
    axes[0].plot(t, test["close"], color="#26364A", lw=0.8, label="close"); axes[0].plot(t, test["ema"], color="#C99A2E", lw=1.4, label="EMA(10 bars)")
    for k in range(3): axes[0].scatter(t[hard == k], test["close"][hard == k], s=10, color=STATE_COLORS[k], zorder=3, label=f"state {k}")
    axes[0].set_title("Test period: close, EMA and the filtered hidden state of every bar"); axes[0].legend(ncol=5, fontsize=8)
    axes[1].stackplot(t, filt_te[:, 0], filt_te[:, 1], filt_te[:, 2], colors=STATE_COLORS[:3], alpha=0.85); axes[1].set_ylabel("state probability"); axes[1].set_ylim(0, 1)
    plt.tight_layout(); plt.savefig(DIAG / "10_state_timeline_test.png", dpi=160); plt.close()

    # ---------------------------------------------------------------- 8. uncertainty: bootstrap distribution
    rng = np.random.default_rng(0); uniq = np.unique(dates_te); groups = {d: np.flatnonzero(dates_te == d) for d in uniq}; boots = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[d] for d in rng.choice(uniq, size=len(uniq), replace=True)])
        if np.unique(y_te[idx]).size > 1: boots.append(balanced_accuracy_score(y_te[idx], pred_te[ok_te][idx]))
    boots = np.array(boots); pd.Series(boots, name="balanced_accuracy").to_csv(DIAG / "bootstrap_balanced_accuracy.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.hist(boots, bins=30, color="#5278A8", alpha=0.85); ax.axvline(1 / 3, ls=":", color="#4B4B4B", label="chance 33.3%"); ax.axvline(m_te["balanced_accuracy"], color="#16865C", lw=2, label=f"observed {m_te['balanced_accuracy']:.1%}")
    ax.axvline(np.percentile(boots, 2.5), color="#D64B40", ls="--", label="95% interval"); ax.axvline(np.percentile(boots, 97.5), color="#D64B40", ls="--")
    ax.set_xlabel("balanced accuracy"); ax.set_title("Day-block bootstrap of the test balanced accuracy"); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(DIAG / "11_bootstrap_distribution.png", dpi=160); plt.close()

    cmp = compare_ema_spans(n_boot)
    _write_docs(profile, share, traces, seeds, sel, spans, metrics, cal, cov, tod_tbl, state_tbl, boots, hmm, cmp)
    print(f"diagnostics written to {DIAG} and documents to {DOCS}")


def compare_ema_spans(n_boot: int = 1000) -> dict:
    """Head-to-head of the 10-bar and 21-bar EMA models: separate HMMs, same protocol."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    res = {}
    for span in (10, 21):
        frame, train, test, _ = _prep(span)
        hmm = WindowHMM().fit(train)
        trace = em_trace(train, test, seed=7)   # same seed for both spans so the curves are comparable
        ok_tr, y_tr = _mask_and_labels(train); ok_te, y_te = _mask_and_labels(test)
        p_tr, pred_tr, filt_tr = hmm.predict_next(train, None); p_te, pred_te, filt_te = hmm.predict_next(test, None)
        m_tr = score(f"EMA{span} train", y_tr, pred_tr[ok_tr], train.loc[ok_tr, "trading_date"].to_numpy(), p_tr[ok_tr], n_boot)
        m_te = score(f"EMA{span} test", y_te, pred_te[ok_te], test.loc[ok_te, "trading_date"].to_numpy(), p_te[ok_te], n_boot)
        xs = train[["ema_change_bps"]].to_numpy() / hmm.scale_; xt = test[["ema_change_bps"]].to_numpy() / hmm.scale_
        res[span] = dict(hmm=hmm, trace=trace, test=test, ok_te=ok_te, y_te=y_te, pred_te=pred_te[ok_te], p_te=p_te[ok_te], filt_te=filt_te,
                         metrics=[m_tr, m_te], ll_train=hmm.model_.score(xs) / len(xs), ll_test=hmm.model_.score(xt) / len(xt),
                         cm=confusion_matrix(y_te, pred_te[ok_te], labels=[0, 1, 2]),
                         states=pd.DataFrame({"model": f"EMA{span}", "state": range(3), "mean_ema_change_bps": hmm.means_bps_, "stay_probability": np.diag(hmm.transmat_),
                                              "expected_run_bars": 1 / np.clip(1 - np.diag(hmm.transmat_), 1e-9, None),
                                              "p_next_bearish": hmm.state_to_next_[:, 0], "p_next_sideways": hmm.state_to_next_[:, 1], "p_next_bullish": hmm.state_to_next_[:, 2]}),
                         half_life=float(np.log(0.5) / np.log(np.sort(np.abs(np.linalg.eigvals(hmm.transmat_)))[-2])),
                         ema_std_bps=float(train["ema_change_bps"].std()))
    # tables
    metrics = pd.DataFrame(res[10]["metrics"] + res[21]["metrics"])
    metrics.insert(1, "ema_span_bars", [10, 10, 21, 21])
    metrics.to_csv(DIAG / "ema10_vs_ema21_metrics.csv", index=False)
    summary = pd.DataFrame([{
        "ema_span_bars": sp, "ema_change_std_bps": r["ema_std_bps"], "train_loglik_per_bar": r["ll_train"], "test_loglik_per_bar": r["ll_test"],
        "memory_half_life_bars": r["half_life"], "iterations_to_converge": int(((r["trace"]["train_loglik_per_bar"] >= r["trace"]["train_loglik_per_bar"].iloc[-1] - 1e-3).to_numpy()).argmax() + 1),
        "train_balanced_accuracy": r["metrics"][0]["balanced_accuracy"], "test_balanced_accuracy": r["metrics"][1]["balanced_accuracy"],
        "test_ci_low": r["metrics"][1]["ci_low"], "test_ci_high": r["metrics"][1]["ci_high"], "test_permutation_p": r["metrics"][1]["permutation_p"],
        "test_log_loss": r["metrics"][1]["log_loss"], "classes_predicted": r["metrics"][1]["predicted_classes"],
        "test_recall_bearish": r["metrics"][1]["recall_bearish"], "test_recall_sideways": r["metrics"][1]["recall_sideways"], "test_recall_bullish": r["metrics"][1]["recall_bullish"],
    } for sp, r in res.items()])
    summary.to_csv(DIAG / "ema10_vs_ema21_summary.csv", index=False)
    pd.concat([res[10]["states"], res[21]["states"]]).to_csv(DIAG / "ema10_vs_ema21_hidden_states.csv", index=False)
    agree = (res[10]["pred_te"] == res[21]["pred_te"])
    y = res[10]["y_te"]
    agreement = pd.DataFrame([
        {"subset": "all test bars", "bars": int(len(y)), "EMA10_accuracy": float((res[10]["pred_te"] == y).mean()), "EMA21_accuracy": float((res[21]["pred_te"] == y).mean())},
        {"subset": "bars where both agree", "bars": int(agree.sum()), "EMA10_accuracy": float((res[10]["pred_te"][agree] == y[agree]).mean()), "EMA21_accuracy": float((res[21]["pred_te"][agree] == y[agree]).mean())},
        {"subset": "bars where they disagree", "bars": int((~agree).sum()), "EMA10_accuracy": float((res[10]["pred_te"][~agree] == y[~agree]).mean()), "EMA21_accuracy": float((res[21]["pred_te"][~agree] == y[~agree]).mean())},
    ])
    agreement.to_csv(DIAG / "ema10_vs_ema21_agreement.csv", index=False)
    pred_counts = pd.DataFrame({"regime": REGIMES, "EMA10_predicted": [int((res[10]["pred_te"] == k).sum()) for k in range(3)], "EMA21_predicted": [int((res[21]["pred_te"] == k).sum()) for k in range(3)], "actual": [int((y == k).sum()) for k in range(3)]})
    pred_counts.to_csv(DIAG / "ema10_vs_ema21_predicted_counts.csv", index=False)
    # figures
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.2, "font.size": 10})
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
    for sp, c in ((10, "#16865C"), (21, "#5278A8")):
        t = res[sp]["trace"]
        axes[0].plot(t["iteration"], t["train_loglik_per_bar"], color=c, label=f"EMA{sp} train"); axes[0].plot(t["iteration"], t["test_loglik_per_bar"], color=c, ls="--", label=f"EMA{sp} test")
        axes[1].plot(t["iteration"], t["test_balanced_accuracy"], color=c, label=f"EMA{sp} test"); axes[1].plot(t["iteration"], t["train_balanced_accuracy"], color=c, ls="--", alpha=0.6, label=f"EMA{sp} train")
    axes[0].set_title("Log-likelihood per bar during training"); axes[0].set_xlabel("EM iteration"); axes[0].legend(fontsize=8)
    axes[1].axhline(1 / 3, ls=":", color="#4B4B4B"); axes[1].set_ylim(0.2, 0.6); axes[1].set_title("Balanced accuracy during training"); axes[1].set_xlabel("EM iteration"); axes[1].legend(fontsize=8)
    x = np.arange(3); w = 0.35
    for i, (sp, c) in enumerate(((10, "#16865C"), (21, "#5278A8"))):
        m = res[sp]["metrics"][1]
        axes[2].bar(x + (i - 0.5) * w, [m["recall_bearish"], m["recall_sideways"], m["recall_bullish"]], w, color=c, label=f"EMA{sp}")
    axes[2].set_xticks(x, REGIMES); axes[2].set_ylim(0, 1); axes[2].set_title("Per-class recall on test"); axes[2].legend()
    plt.tight_layout(); plt.savefig(DIAG / "12_ema10_vs_ema21_training_and_recall.png", dpi=160); plt.close()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, sp in zip(axes, (10, 21)):
        cm = res[sp]["cm"]; cmn = cm / cm.sum(1, keepdims=True)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1); ax.grid(False)
        for i in range(3):
            for j in range(3): ax.text(j, i, f"{cm[i, j]}\n({cmn[i, j]:.0%})", ha="center", va="center", color="white" if cmn[i, j] > 0.5 else "black", fontsize=9)
        ax.set_xticks(range(3), REGIMES); ax.set_yticks(range(3), REGIMES); ax.set_xlabel("predicted"); ax.set_ylabel("actual"); ax.set_title(f"EMA{sp}: confusion matrix, test")
    plt.tight_layout(); plt.savefig(DIAG / "13_ema10_vs_ema21_confusion.png", dpi=160); plt.close()
    fig, axes = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
    for ax, sp in zip(axes, (10, 21)):
        test = res[sp]["test"]; hard = res[sp]["filt_te"].argmax(1); t = test["timestamp"]
        ax.plot(t, test["close"], color="#26364A", lw=0.8); ax.plot(t, test["ema"], color="#C99A2E", lw=1.4, label=f"EMA({sp} bars)")
        for k in range(3): ax.scatter(t[hard == k], test["close"][hard == k], s=10, color=STATE_COLORS[k], zorder=3, label=f"state {k}")
        ax.set_title(f"EMA{sp}: filtered hidden state of every test bar"); ax.legend(ncol=4, fontsize=8)
    plt.tight_layout(); plt.savefig(DIAG / "14_ema10_vs_ema21_state_timeline.png", dpi=160); plt.close()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, sp in zip(axes, (10, 21)):
        T = res[sp]["hmm"].transmat_; ax.imshow(T, cmap="Blues", vmin=0, vmax=1); ax.grid(False)
        for i in range(3):
            for j in range(3): ax.text(j, i, f"{T[i, j]:.0%}", ha="center", va="center", color="white" if T[i, j] > 0.5 else "black")
        ax.set_xticks(range(3), [f"to {k}" for k in range(3)]); ax.set_yticks(range(3), [f"from {k}" for k in range(3)]); ax.set_title(f"EMA{sp}: transition matrix (half-life {res[sp]['half_life']:.1f} bars)")
    plt.tight_layout(); plt.savefig(DIAG / "15_ema10_vs_ema21_transitions.png", dpi=160); plt.close()
    # filter-window comparison on the EMA10 model (same model, different memory)
    hmm10, test10 = res[10]["hmm"], res[10]["test"]; ok10, y10 = res[10]["ok_te"], res[10]["y_te"]
    _, pf, _ = hmm10.predict_next(test10, None)
    win_rows = []
    for wlen in (1, 2, 3, 5, 8, 10, 21):
        pw, predw, _ = hmm10.predict_next(test10, wlen)
        win_rows.append({"filter_window_bars": wlen, "balanced_accuracy": balanced_accuracy_score(y10, predw[ok10]), "agrees_with_full_history": float((predw[ok10] == pf[ok10]).mean()), "max_abs_prob_diff": float(np.abs(pw[ok10] - res[10]["p_te"]).max())})
    windows = pd.DataFrame(win_rows); windows.to_csv(DIAG / "ema10_filter_window_comparison.csv", index=False)
    return {"summary": summary, "metrics": metrics, "agreement": agreement, "pred_counts": pred_counts, "states": pd.concat([res[10]["states"], res[21]["states"]]), "windows": windows}


def _write_docs(profile, share, traces, seeds, sel, spans, metrics, cal, cov, tod_tbl, state_tbl, boots, hmm, cmp):
    fig = "../diagnostics/"
    tr7 = traces[traces["seed"] == 7]
    every5 = tr7[tr7["iteration"].isin([1, 2, 3, 5, 10, 15, 20, 30, 40, 50, 60])][["iteration", "train_loglik_per_bar", "test_loglik_per_bar", "improvement", "train_balanced_accuracy", "test_balanced_accuracy"]]
    mt = metrics.set_index("model")
    te, trn = mt.loc["test (unseen dates)"], mt.loc["train (in-sample)"]
    docs = {}

    docs["01_overview.md"] = f"""# 1. What this model is and how to read these documents

**Model:** a 3-state Gaussian hidden Markov model (HMM) on the bar-to-bar change of a 10-bar exponential moving average (EMA) of the USD/INR futures 30-minute close. It is trained once on the first 80% of active trading dates and never sees the last 20%.

**What it predicts:** for every 30-minute bar, the probability that the *next* bar is bearish, sideways or bullish, using only bars up to now.

**Headline numbers**

| | train (in-sample) | test (unseen dates) |
|---|---|---|
| dates | {int(share.loc[0,'dates'])} | {int(share.loc[1,'dates'])} |
| scored bars | {int(share.loc[0,'scored_bars'])} | {int(share.loc[1,'scored_bars'])} |
| balanced accuracy (chance = 33.3%) | {trn['balanced_accuracy']:.1%} | {te['balanced_accuracy']:.1%} |
| 95% interval | {trn['ci_low']:.1%} to {trn['ci_high']:.1%} | {te['ci_low']:.1%} to {te['ci_high']:.1%} |
| permutation p-value | {trn['permutation_p']:.3f} | {te['permutation_p']:.3f} |

**How an HMM is trained, in one paragraph.** An HMM has no epochs in the neural-network sense. It is fitted by the Baum-Welch algorithm, a form of expectation-maximisation (EM). Each iteration does two steps: guess which hidden state each bar was in (E-step), then re-estimate the state means, variances and transition probabilities from those guesses (M-step). The quantity being maximised is the **log-likelihood** of the training data, the HMM equivalent of a loss curve, except that it goes *up*. Baum-Welch is guaranteed never to decrease it, so the curve is monotone and flattens when a local optimum is reached. Because it can only find a local optimum, the model is fitted from several random starting points and the best is kept.

**Documents in this folder**

| file | question it answers |
|---|---|
| 02_data_and_labels.md | What does the model learn from, and what is it asked to predict? |
| 03_training_curve.md | How did the fit improve iteration by iteration, on train and on test? |
| 04_random_restarts.md | Do different starting points reach the same model? |
| 05_model_selection.md | Why 3 states and a 10-bar EMA? |
| 06_train_vs_test.md | How does performance on seen data compare with unseen data? |
| 07_probabilities_and_calibration.md | Are the probabilities honest, and does confidence help? |
| 08_hidden_states.md | What do the three states mean and how do they move? |
| 09_uncertainty.md | How sure can we be about the test number? |
| 10_summary.md | What it all means, in plain words. |
| 11_ema10_vs_ema21.md | Full head-to-head of the 10-bar and 21-bar EMA models, and of the 10- and 21-bar filter windows. |

All tables are generated from CSV files in `diagnostics/` by `python3 usdinr_ema_window_hmm_v7.py diagnostics`.
"""

    docs["02_data_and_labels.md"] = f"""# 2. Data and labels

**Source.** Kite Connect 30-minute candles of USD/INR futures, stitched across live contracts, {profile['all_trading_dates']} dates from {profile['first_active_date'][:4]}. A date is kept only if at least 12 of its 16 session bars exist and at least 8 of them traded. That leaves **{profile['active_trading_dates']} active dates** ({profile['first_active_date']} to {profile['last_active_date']}) and {profile['active_bars']} bars.

**Input to the HMM.** One number per bar: the change in the 10-bar EMA of the close, in basis points. Nothing else. Volume, open interest, time of day and raw prices are deliberately excluded so that the states describe price behaviour only.

**Label (what is being predicted).** The next bar's close-to-close move is compared with a band:

```
band = max( 2 ticks ,  rolling one-third quantile of |past moves| over 80 bars )
bullish  if next move >  band
bearish  if next move < -band
sideways otherwise
```

The band adapts to recent volatility and can never shrink below two ticks (0.005 rupees), so "sideways" always means a move too small to trade.

**Class shares and band size, train vs test**

{_md_table(share, "{:.3f}")}

![label shares]({fig}01_label_shares.png)

*Reading the figure:* the three classes are of similar size in both halves and none is dominant, so a model that always predicts one class cannot score above about 40% plain accuracy. The test period has slightly more bullish bars and a slightly narrower band than training, which is a mild distribution shift, not a large one.
"""

    docs["03_training_curve.md"] = f"""# 3. Training curve: what changed at each iteration

The HMM was refitted one Baum-Welch iteration at a time so that every iteration could be measured. Four things are recorded after each iteration:

- **train log-likelihood per bar**: how well the model explains the training bars (the "loss", but higher is better);
- **test log-likelihood per bar**: the same score on the unseen bars, which shows whether the fit generalises;
- **improvement**: how much the train log-likelihood rose in that iteration;
- **balanced accuracy** of the next-bar forecast on train and test, using the model as it stood after that iteration.

![training curve]({fig}02_training_curve.png)

**Seed 7 (the seed the final model uses), selected iterations**

{_md_table(every5, "{:.4f}")}

**How to read this**

1. *Left panel.* Log-likelihood rises quickly in the first few iterations and then flattens. Most of the fit is found within about 10 iterations; after 20 the changes are below 0.001 per bar. This is normal for EM: big gains first, then a long tail.
2. *Middle panel.* The per-iteration improvement falls roughly geometrically (a straight line on the log scale). The final model stops when this falls below 0.0001, the `tol` setting.
3. *Right panel.* Balanced accuracy on **test** moves very little across iterations. This is the most important observation in this document: the HMM keeps improving its description of the training data, but the forecast quality it delivers on unseen data barely changes after the first few iterations. Fitting the past better does not translate into predicting the future better, because the states carry little information about direction.
4. *Test versus train log-likelihood.* The test curve peaks at iteration {int(tr7.loc[tr7['test_loglik_per_bar'].idxmax(),'iteration'])} ({tr7['test_loglik_per_bar'].max():.4f}) and then drifts down to {tr7['test_loglik_per_bar'].iloc[-1]:.4f} while the train curve keeps rising. That gap of about {tr7['test_loglik_per_bar'].max()-tr7['test_loglik_per_bar'].iloc[-1]:.3f} per bar is a small, genuine sign that the later iterations fit details of the training period that do not recur. It is small because the model has only {_n_params(3)} parameters for {int(share.loc[0,'scored_bars'])} bars, and it does not change the forecast accuracy, which is what matters downstream.

![state means]({fig}03_state_means_per_iteration.png)

*Reading the figure:* the three state means start close together at the random initialisation and separate within the first ten iterations into a falling-EMA state, a flat state and a rising-EMA state. After that they are stable. This is the model "discovering" the three regimes.
"""

    docs["04_random_restarts.md"] = f"""# 4. Random restarts: do different starting points agree?

Baum-Welch only finds a local optimum, so the model is fitted from four different random starting points (seeds 7, 21, 42, 99) and the one with the highest training log-likelihood is kept.

{_md_table(seeds, "{:.4f}")}

**How to read this**

- {int((np.abs(seeds['final_train_loglik_per_bar'] - seeds['final_train_loglik_per_bar'].max()) < 1e-3).sum())} of the 4 seeds reach the same best optimum ({seeds['final_train_loglik_per_bar'].max():.4f} per bar) and produce the same three regimes and the same test accuracy. Seed {int(seeds.loc[seeds['final_train_loglik_per_bar'].idxmin(),'seed'])} gets stuck in a clearly worse local optimum ({seeds['final_train_loglik_per_bar'].min():.4f}) whose forecast is *below* chance on test. This is exactly why Baum-Welch is run from several starting points and only the best kept.
- The training-curve figure in document 3 shows a second failure mode: one seed sits on a flat plateau for about 25 iterations before escaping to the best optimum. A stopping rule based only on "small improvement" would have stopped it early, which is why the final model uses a strict tolerance and up to 500 iterations.
- `iterations_to_converge` is the first iteration at which the train log-likelihood came within 0.001 per bar of its final value.
"""

    docs["05_model_selection.md"] = f"""# 5. Model selection: why three states and a 10-bar EMA

## Number of hidden states

A model with more states can always fit the training data at least as well, so the raw log-likelihood cannot choose the number of states. Two standard corrections penalise the number of parameters: **AIC** (Akaike) and **BIC** (Bayesian). Lower is better. The held-out log-likelihood and the test balanced accuracy are shown alongside.

{_md_table(sel, "{:.3f}")}

![state count]({fig}04_state_count_selection.png)

**How to read this.** BIC penalises complexity more heavily than AIC. On likelihood grounds the criteria favour more states: BIC is lowest at {int(sel.loc[sel['BIC'].idxmin(),'states'])} states and AIC at {int(sel.loc[sel['AIC'].idxmin(),'states'])}, and the held-out log-likelihood also keeps improving as states are added. But the quantity the project cares about, the forecast on unseen dates, is flat: test balanced accuracy stays between {sel['test_balanced_accuracy'].min():.1%} and {sel['test_balanced_accuracy'].max():.1%} from 2 to 6 states while train accuracy creeps up. Extra states describe the EMA's behaviour in finer detail without saying more about the next bar. Three states is therefore chosen as the smallest model that separates falling, flat and rising behaviour, matches the three classes being predicted, and loses nothing on the forecast. A 5-state model is the natural candidate if the target is changed to volatility rather than direction.

## EMA smoothing length

The same model was trained on EMAs of different lengths. A longer EMA is smoother and lags the market more.

{_md_table(spans, "{:.3f}")}

![span sensitivity]({fig}05_ema_span_sensitivity.png)

**How to read this.** `memory_half_life_bars` is how many bars it takes for the influence of an observation on the filtered state to halve; it comes from the second eigenvalue of the transition matrix. Short EMAs give the model a memory of a few bars and the best test score; long EMAs make the states last longer but the forecast lags and test accuracy falls. The permutation p-value is the chance of seeing that balanced accuracy if forecasts and outcomes were unrelated.
"""

    docs["06_train_vs_test.md"] = f"""# 6. Performance on training data versus unseen data

The same fitted model is scored on the bars it was trained on (in-sample) and on the last 20% of dates it never saw. All numbers are on non-overlapping 30-minute windows; intervals are from a bootstrap that resamples whole trading days.

{_md_table(metrics[["model","windows","dates","balanced_accuracy","ci_low","ci_high","permutation_p","accuracy","log_loss","recall_bearish","recall_sideways","recall_bullish","predicted_classes"]], "{:.3f}")}

**How to read this**

- Train and test balanced accuracy are close ({trn['balanced_accuracy']:.1%} vs {te['balanced_accuracy']:.1%}). A large gap would mean memorisation; there is none, because the model has only {_n_params(3)} parameters.
- Log loss measures the quality of the probabilities; 1.099 is the value of a uniform guess over three classes. Train is {trn['log_loss']:.3f} and test {te['log_loss']:.3f}, so the probabilities are essentially uninformative: about as good as saying one third each. The small accuracy edge comes from the argmax rule, not from sharp probabilities.
- Recall per class shows *where* the model is right: it recognises sideways bars well and bullish bars moderately, but bearish bars poorly. The confusion matrices below show the same thing in counts.

![confusion]({fig}06_confusion_matrices.png)

*Reading the figure:* rows are what actually happened, columns what the model said. Percentages are per row. The bearish row is mostly predicted as sideways: when the market is about to fall, the model usually says "flat". That is the single biggest weakness of the forecast and it is present in training as well as test, so it is a property of the information available, not of overfitting.
"""

    docs["07_probabilities_and_calibration.md"] = f"""# 7. Are the probabilities honest, and does confidence help?

The model outputs three probabilities per bar. A well-calibrated model is right about 50% of the time when it says 50%. The reliability diagram groups test bars by the model's highest probability and compares it with the actual accuracy in that group.

{_md_table(cal, "{:.3f}")}

![calibration]({fig}07_calibration_and_coverage.png)

**Reading the left panel.** Points near the dotted line are well calibrated. Points below it mean the model is over-confident. With {int(share.loc[1,'scored_bars'])} test bars the bins are small (n shown next to each point), so the picture is indicative rather than precise.

**Selective prediction (right panel).** If the model only acts when its confidence is above a cutoff, coverage falls and accuracy on the remaining bars may rise:

{_md_table(cov, "{:.3f}")}

If accuracy rose as coverage fell, the probabilities would contain ranking information. Here it does not: on the {int(cov.iloc[-1]['bars'])} bars where the model is most confident, accuracy is {cov.iloc[-1]['accuracy']:.1%}, lower than the {cov.iloc[0]['accuracy']:.1%} over all bars. The model's highest probability never exceeds {cal['mean_confidence'].max()+0.03:.2f}, so "confidence" is a difference between 0.39 and 0.43, which is noise. **Conclusion: the model's own confidence should not be used as a trading gate.** This is why the RL layer gates on persistence of the predicted direction across consecutive bars and on time-of-day statistics instead.
"""

    docs["08_hidden_states.md"] = f"""# 8. The hidden states: what they mean and how they move

Each state is a Gaussian distribution over the EMA change. After fitting, the states are sorted by their mean so that state 0 is the most negative (falling EMA) and state 2 the most positive (rising EMA).

{_md_table(state_tbl, "{:.3f}")}

- `stay_probability` is the chance the market stays in the same state next bar; `expected_run_bars` is the average length of a run in that state.
- `p_next_*` is the forecasting link: on training bars, how often the *next* bar was bearish, sideways or bullish given the current state. Rows that are nearly flat are the reason accuracy is modest: knowing the state tells you little about the next move.

![transition]({fig}09_transition_and_forecast_link.png)

*Reading the figure:* the left matrix is the chance of moving from one state to another in one bar; high diagonal values mean regimes persist for several bars. The right matrix is the forecasting link. Compare the rows: they differ by only a few points.

![timeline]({fig}10_state_timeline_test.png)

*Reading the figure:* the top panel shows the test-period close (dark line), the 10-bar EMA (gold) and the filtered state of every bar (coloured dots). The bottom panel shows the state probabilities as a stacked area, so you can see when the model is certain (one colour fills the band) and when it hesitates (colours share the band). The states track the EMA slope cleanly, which is what they were fitted to do; the question is only whether the current slope predicts the next move.
"""

    lo, hi = np.percentile(boots, [2.5, 97.5])
    docs["09_uncertainty.md"] = f"""# 9. How sure can we be about the test number?

The test balanced accuracy is {te['balanced_accuracy']:.1%} on {int(share.loc[1,'scored_bars'])} bars from {int(share.loc[1,'dates'])} days. Two checks say how much to trust it.

**Day-block bootstrap.** Whole trading days are resampled with replacement {len(boots)} times and the balanced accuracy recomputed each time. Resampling days rather than bars respects the fact that bars within a day are related.

- 95% interval: **{lo:.1%} to {hi:.1%}**
- share of bootstrap samples at or below chance (33.3%): **{np.mean(boots <= 1/3):.1%}**

![bootstrap]({fig}11_bootstrap_distribution.png)

**Permutation test.** The forecasts are shuffled against the outcomes {2000} times. The p-value, {te['permutation_p']:.3f}, is the chance of a balanced accuracy at least this high if forecasts and outcomes were unrelated.

**Accuracy by time of day.** With only {int(share.loc[1,'dates'])} test days, each half-hour slot has {int(share.loc[1,'dates'])} bars, so the per-slot values below swing widely and should not be over-read.

{_md_table(tod_tbl, "{:.3f}")}

![time of day]({fig}08_accuracy_by_time_of_day.png)

**What this means.** The interval touches the chance level. The honest statement is that the model is probably a little better than chance at this task, that the size of the edge cannot be pinned down with {int(share.loc[1,'dates'])} test days, and that only more unseen trading days can narrow the interval.
"""

    docs["10_summary.md"] = f"""# 10. Summary in plain words

1. **The model trains cleanly, but restarts matter.** The log-likelihood rises and flattens within about 20 iterations and the three states separate into falling, flat and rising EMA regimes. Three of four random starts reach the same optimum; one gets stuck in a worse one and another needs 25 extra iterations to escape a plateau, which is why several seeds and a strict stopping rule are used (documents 3, 4, 8).
2. **It barely overfits.** Test log-likelihood peaks early and then slips by about {tr7['test_loglik_per_bar'].max()-tr7['test_loglik_per_bar'].iloc[-1]:.03f} per bar while training keeps improving, a small effect for a {_n_params(3)}-parameter model on {int(share.loc[0,'scored_bars'])} bars. Train and test balanced accuracy are {trn['balanced_accuracy']:.1%} and {te['balanced_accuracy']:.1%} (documents 3, 6).
3. **Fitting better does not mean forecasting better.** Across training iterations the description of the past keeps improving while the forecast on unseen data barely moves (document 3). The reason is visible in the forecasting link: each state gives almost the same probabilities for the next bar (document 8).
4. **Three states is a judgement, not a likelihood optimum.** BIC prefers {int(sel.loc[sel['BIC'].idxmin(),'states'])} states and held-out likelihood keeps rising with more states, but test accuracy is flat from 2 to 6 states, so the smallest interpretable model is kept. A short EMA (5 to 10 bars) is clearly better than a long one (document 5).
5. **The forecast is modestly above chance and the uncertainty is wide.** {te['balanced_accuracy']:.1%} balanced accuracy on {int(share.loc[1,'scored_bars'])} unseen bars, interval {lo:.1%} to {hi:.1%}, permutation p = {te['permutation_p']:.3f}. It recognises sideways bars well and bearish bars poorly (documents 6, 9).
6. **The probabilities are flat and confidence does not help.** The highest class probability never exceeds about 0.45, log loss is at the uniform-guess level, and the model's more confident bars are not more accurate (document 7). The forecast is useful as a categorical signal, not as a probability, which is how the RL layer uses it.

**References used for the method**

- Rabiner, L. R. (1989). A tutorial on hidden Markov models. *Proceedings of the IEEE*: Baum-Welch, forward filtering.
- hmmlearn documentation, `GaussianHMM` and `ConvergenceMonitor`: iteration-by-iteration log-likelihood and the `tol` stopping rule (https://hmmlearn.readthedocs.io/en/stable/api.html).
- Akaike (1974) and Schwarz (1978): AIC and BIC for choosing the number of states.
- Niculescu-Mizil and Caruana (2005): reliability diagrams for probability calibration.
- Efron and Tibshirani (1993): the bootstrap; block resampling by day follows the dependence structure of intraday data.
"""
    sm = cmp["summary"].set_index("ema_span_bars")
    ag = cmp["agreement"]
    docs["11_ema10_vs_ema21.md"] = f"""# 11. Head-to-head: 10-bar EMA versus 21-bar EMA

Two questions were asked during the project and they are different:

1. **Which EMA length should the model see?** A 10-bar EMA (5 hours) or a 21-bar EMA (10.5 hours). Each needs its own trained HMM, because the input series is different.
2. **How much history should the filter use?** Given one trained model, does it matter whether the forward filter is restarted with the last 10 or the last 21 observations?

## Part A: two separately trained models

Everything below is the same protocol as the rest of these documents: 43 training dates, 11 unseen dates, non-overlapping bars, day-block bootstrap, permutation test.

{_md_table(cmp["summary"][["ema_span_bars","ema_change_std_bps","train_loglik_per_bar","test_loglik_per_bar","memory_half_life_bars","iterations_to_converge"]], "{:.3f}")}

{_md_table(cmp["summary"][["ema_span_bars","train_balanced_accuracy","test_balanced_accuracy","test_ci_low","test_ci_high","test_permutation_p","test_log_loss","classes_predicted"]], "{:.3f}")}

{_md_table(cmp["summary"][["ema_span_bars","test_recall_bearish","test_recall_sideways","test_recall_bullish"]], "{:.3f}")}

![training and recall]({fig}12_ema10_vs_ema21_training_and_recall.png)

**How to read this**

- *Input smoothness.* The 21-bar EMA changes about {sm.loc[10,'ema_change_std_bps']/sm.loc[21,'ema_change_std_bps']:.1f} times less per bar than the 10-bar EMA (standard deviation {sm.loc[21,'ema_change_std_bps']:.2f} vs {sm.loc[10,'ema_change_std_bps']:.2f} bps). A smoother input is easier to describe, which is why the 21-bar model reaches a higher log-likelihood per bar, but it also lags the market more.
- *Memory.* The half-life of the filtered state is {sm.loc[10,'memory_half_life_bars']:.1f} bars for EMA10 and {sm.loc[21,'memory_half_life_bars']:.1f} bars for EMA21. The 21-bar model holds a state for longer and reacts later.
- *Forecast.* Test balanced accuracy is {sm.loc[10,'test_balanced_accuracy']:.1%} (EMA10) against {sm.loc[21,'test_balanced_accuracy']:.1%} (EMA21). The EMA21 permutation p-value is {sm.loc[21,'test_permutation_p']:.2f}, so it is not distinguishable from chance, and it predicts only {int(sm.loc[21,'classes_predicted'])} of the 3 classes. The training-curve panel shows this is not a fitting problem: the 21-bar model converges normally but its forecast never rises above chance at any iteration.

**Predicted class counts on the 165 test bars**

{_md_table(cmp["pred_counts"], "{:.0f}")}

![confusion]({fig}13_ema10_vs_ema21_confusion.png)

**Where the two models differ**

{_md_table(ag, "{:.3f}")}

The two forecasts agree on {ag.loc[1,'bars']} of {ag.loc[0,'bars']} bars. On the {ag.loc[2,'bars']} bars where they disagree, EMA10 is right {ag.loc[2,'EMA10_accuracy']:.0%} of the time and EMA21 {ag.loc[2,'EMA21_accuracy']:.0%}. Most disagreements sit around turns in the price, where the slower EMA is still pointing the old way.

**Hidden states of the two models**

{_md_table(cmp["states"], "{:.3f}")}

![transitions]({fig}15_ema10_vs_ema21_transitions.png)

![timelines]({fig}14_ema10_vs_ema21_state_timeline.png)

*Reading the figures:* both models find a falling, a flat and a rising state, but the 21-bar states are stickier (higher diagonal) and switch later. In the timeline the EMA21 states change colour several bars after the EMA10 states at each turn in the test period. That lag is the whole story: on a series that mean-reverts within a few bars, a slower trend estimate arrives after the move it describes.

## Part B: same model, different filter window

The EMA10 model was also run with the forward filter restarted so that only the last N observations are used at each bar.

{_md_table(cmp["windows"], "{:.3f}")}

**How to read this.** With 8 or more observations the forecasts are identical to using the full history, and with 10 or 21 the maximum difference in any probability is below 0.001. The filter forgets geometrically at the rate of the transition matrix, and after two half-lives almost nothing of the earlier bars remains. So "last 10" and "last 21" observations are the same forecast, which is what the earlier two-forecast experiment found. Only windows shorter than about 5 bars change anything, and those differences are within noise: a single bar of history drops the score to chance, while 2 to 5 bars land within a point or two of the full-history result.

## Conclusion

The 10-bar EMA is the better input: it reacts faster, predicts all three classes, and is the only one of the two whose test result passes the permutation test. The 21-bar EMA describes the training data more smoothly but lags the market, and its forecast is at chance. Window length in the filter is irrelevant beyond 8 bars, so the choice that matters is the EMA span, not the memory of the filter.
"""
    for name, text in docs.items():
        (DOCS / name).write_text(text)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "run"
    OUT.mkdir(exist_ok=True)
    if what in ("run", "all"):
        main()
    if what in ("spans", "all"):
        export_spans()
    if what in ("span10", "all"):
        export_span10()
    if what in ("all6months", "all"):
        export_all_6_months()
    if what == "diagnostics":
        diagnostics()
