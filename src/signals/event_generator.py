import numpy as np
import pandas as pd


EVENT_FEATURE_COLUMNS = [
    "event_breakout_up_20",
    "event_breakout_down_20",
    "event_volatility_expansion_10",
    "event_momentum_shift_up",
    "event_momentum_shift_down",
    "event_trend_alignment_up",
    "event_trend_alignment_down",
    "event_range_impulse",
]


def _as_float_series(df, column, default=0.0):
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=np.float32)
    return pd.to_numeric(df[column], errors="coerce").astype(np.float32)


def add_event_features(df):
    if df is None or df.empty:
        return df

    enriched = df.copy()
    close = _as_float_series(enriched, "close")
    high = _as_float_series(enriched, "high")
    low = _as_float_series(enriched, "low")
    open_price = _as_float_series(enriched, "open")
    ret_3 = _as_float_series(enriched, "ret_3", default=(close - close.shift(3)))
    ret_10 = _as_float_series(enriched, "ret_10", default=(close - close.shift(10)))
    ma_5 = _as_float_series(enriched, "ma_5", default=close.rolling(5).mean())
    ma_20 = _as_float_series(enriched, "ma_20", default=close.rolling(20).mean())
    range_1 = _as_float_series(enriched, "range_1", default=(high - low))

    breakout_up = close > high.rolling(20).max().shift(1)
    breakout_down = close < low.rolling(20).min().shift(1)
    avg_range_10 = range_1.rolling(10).mean()
    volatility_expansion = range_1 > (avg_range_10 * 1.25)
    momentum_shift_up = (ret_3 > 0) & (ret_10 <= 0)
    momentum_shift_down = (ret_3 < 0) & (ret_10 >= 0)
    trend_alignment_up = (close > ma_5) & (ma_5 > ma_20)
    trend_alignment_down = (close < ma_5) & (ma_5 < ma_20)
    range_impulse = range_1 > range_1.rolling(20).quantile(0.75)

    events = {
        "event_breakout_up_20": breakout_up,
        "event_breakout_down_20": breakout_down,
        "event_volatility_expansion_10": volatility_expansion,
        "event_momentum_shift_up": momentum_shift_up,
        "event_momentum_shift_down": momentum_shift_down,
        "event_trend_alignment_up": trend_alignment_up,
        "event_trend_alignment_down": trend_alignment_down,
        "event_range_impulse": range_impulse,
    }

    for column, values in events.items():
        enriched[column] = values.fillna(False).astype(np.float32)

    return enriched


def has_entry_event(row, algorithm):
    direction = str(algorithm).upper()
    if isinstance(row, pd.Series):
        get_value = row.get
    else:
        get_value = lambda key, default=0.0: getattr(row, key, default)

    volatility_expansion = float(get_value("event_volatility_expansion_10", 0.0)) >= 0.5
    range_impulse = float(get_value("event_range_impulse", 0.0)) >= 0.5

    if direction == "UP":
        directional_signal = (
            float(get_value("event_breakout_up_20", 0.0)) >= 0.5 or
            float(get_value("event_momentum_shift_up", 0.0)) >= 0.5 or
            float(get_value("event_trend_alignment_up", 0.0)) >= 0.5
        )
    else:
        directional_signal = (
            float(get_value("event_breakout_down_20", 0.0)) >= 0.5 or
            float(get_value("event_momentum_shift_down", 0.0)) >= 0.5 or
            float(get_value("event_trend_alignment_down", 0.0)) >= 0.5
        )

    return bool(directional_signal and (volatility_expansion or range_impulse))
