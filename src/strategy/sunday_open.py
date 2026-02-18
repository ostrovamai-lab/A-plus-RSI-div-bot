"""Sunday Open — weekly bias filter for trend-following entries.

Long: price > Sunday Open = bullish week bias.
Short: price < Sunday Open = bearish week bias.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_sunday_open(df_8m: pd.DataFrame) -> np.ndarray:
    """Return per-bar array of the most recent Sunday open price.

    NaN until the first Sunday bar is encountered.

    Args:
        df_8m: DataFrame with a ``timestamp`` column (epoch milliseconds, UTC).

    Returns:
        1-D float array, same length as *df_8m*.
    """
    ts = pd.to_datetime(df_8m["timestamp"].values, unit="ms", utc=True)
    opens = df_8m["open"].values.astype(float)
    n = len(df_8m)
    result = np.full(n, np.nan)

    current_so = np.nan
    for i in range(n):
        dow = ts[i].weekday()  # Monday=0 … Sunday=6
        if dow == 6:
            current_so = opens[i]
        result[i] = current_so

    return result


def sunday_open_confirms(
    direction_is_long: bool,
    price: float,
    so_price: float,
) -> bool:
    """Check whether *price* is on the right side of the Sunday Open.

    Long  → price > SO (bullish week).
    Short → price < SO (bearish week).

    Returns ``True`` (pass) when *so_price* is NaN (no data yet).
    """
    if np.isnan(so_price):
        return True
    if direction_is_long:
        return price > so_price
    return price < so_price
