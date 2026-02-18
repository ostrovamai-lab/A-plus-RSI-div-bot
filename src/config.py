"""Configuration loader — merges config.yaml with .env environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default)


# ── Exchange credentials ──────────────────────────────────

BYBIT_API_KEY: str = _env("BYBIT_API_KEY")
BYBIT_API_SECRET: str = _env("BYBIT_API_SECRET")
BYBIT_DEMO: bool = _env("BYBIT_DEMO", "true").lower() == "true"

# ── Telegram ──────────────────────────────────────────────

TELEGRAM_BOT_TOKEN: str = _env("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID: str = _env("TELEGRAM_CHAT_ID")

# ── Paths ─────────────────────────────────────────────────

DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)


@dataclass
class APlusParams:
    pivot_lookback: int = 14
    fractal_len: int = 3
    step_window: int = 50
    gate_max_age: int = 20
    ema_fast: int = 3
    ema_slow: int = 21
    break_mode: str = "Wick+Buffer"
    buffer_ticks: int = 1
    need_retest: bool = True
    retest_lookback: int = 20
    allow_pre_cross: bool = True


@dataclass
class TPRsiParams:
    rsi_period: int = 50
    bb_period: int = 50
    bb_mult: float = 1.0
    bb_sigma: float = 0.1
    ribbon_ma_type: str = "EMA"
    ribbon_pairs: list[list[int]] = field(default_factory=lambda: [
        [21, 48], [51, 98], [103, 149], [155, 199], [206, 499],
    ])
    divergence_lookback: int = 5
    divergence_max_range: int = 60


@dataclass
class KAMAParams:
    er_length: int = 10
    fast_length: int = 2
    slow_length: int = 30
    slope_bars: int = 3


@dataclass
class FilterParams:
    """Hard rejection filters — applied before scoring."""
    # ADX
    adx_period: int = 14
    adx_hard_reject: float = 30.0    # reject counter-trend when ADX >= this
    adx_trend_threshold: float = 20.0  # trend is present
    adx_ranging_threshold: float = 15.0  # no trend (ranging)

    # ATR volatility regime
    atr_ema_period: int = 100
    min_atr_ratio: float = 0.5       # reject if ATR/ATR_EMA < this
    min_sl_range_pct: float = 0.003  # reject if |entry-fractal|/price < this

    # EMA200 trend
    ema200_period: int = 200

    # ATR-based SL
    sl_atr_multiplier: float = 0.0   # 0=disabled; set >0 for ATR-based SL floor


@dataclass
class ScoringWeights:
    aplus_signal: float = 25.0
    rsi_divergence: float = 20.0
    ema_ribbon: float = 15.0
    adx_trend: float = 0.0
    kama_trend: float = 10.0
    ema200_position: float = 0.0
    rsi_position: float = 10.0
    htf_alignment: float = 10.0
    bb_position: float = 5.0
    volume: float = 5.0


@dataclass
class EntryParams:
    fib_levels: list[float] = field(default_factory=lambda: [0.236, 0.382, 0.618])
    fib_volumes: list[float] = field(default_factory=lambda: [1.0, 1.272, 1.618])
    min_range_atr_mult: float = 0.5
    sl_buffer_pct: float = 0.002


@dataclass
class ExitParams:
    time_stop_bars: int = 200
    maker_fee: float = 0.0002
    taker_fee: float = 0.00055
    be_atr_mult: float = 1.0           # activate breakeven after 1×ATR favorable move (0=off)
    be_buffer_pct: float = 0.001       # 0.1% buffer above avg entry (covers fees)
    trail_activation_atr: float = 2.0  # start trailing after 2×ATR (0=off)
    trail_distance_atr: float = 1.5    # trail 1.5×ATR behind peak
    time_stop_trail_bars: int = 500    # extended hold when trailing is active


@dataclass
class PyramidParams:
    max_pyramids: int = 3
    size_scaling: list[float] = field(default_factory=lambda: [1.0, 0.75, 0.5])
    move_sl_to_be: bool = True


@dataclass
class ReinvestmentParams:
    enabled: bool = True
    min_scale: float = 0.5
    max_scale: float = 2.0


@dataclass
class ScannerParams:
    poll_interval_sec: int = 480
    batch_size: int = 4
    batch_delay_sec: float = 0.5
    history_bars_1m: int = 600


@dataclass
class V2Config:
    """V2 trend-following engine configuration."""

    # SMC regime
    smc_1h_swing_length: int = 10
    smc_8m_swing_length: int = 20
    smc_8m_internal_length: int = 5

    # Entry filters
    volume_mult: float = 1.0
    require_premium_discount: bool = True
    require_sunday_open: bool = False

    # Session / Pyramid
    max_sessions: int = 2
    max_pyramid_levels: int = 4
    pyramid_profit_fraction: float = 0.5
    pyramid_max_base_mult: float = 5.0

    # SL
    sl_atr_buffer_mult: float = 0.5
    sl_min_distance_pct: float = 0.002
    sl_max_distance_pct: float = 0.05
    trail_lookback: int = 10

    # Exit
    choch_partial_pct: float = 0.50

    # Fees
    maker_fee: float = 0.0002
    taker_fee: float = 0.00055


@dataclass
class BotConfig:
    """Full bot configuration."""

    symbols: list[str] = field(default_factory=list)
    capital_usd: float = 1000.0
    leverage: int = 10
    max_positions: int = 5
    max_positions_per_coin: int = 3
    max_drawdown_pct: float = 10.0

    min_score: float = 65.0
    pyramid_min_score: float = 70.0

    aplus: APlusParams = field(default_factory=APlusParams)
    tp_rsi: TPRsiParams = field(default_factory=TPRsiParams)
    kama: KAMAParams = field(default_factory=KAMAParams)
    filters: FilterParams = field(default_factory=FilterParams)
    scoring: ScoringWeights = field(default_factory=ScoringWeights)
    entry: EntryParams = field(default_factory=EntryParams)
    exit: ExitParams = field(default_factory=ExitParams)
    pyramid: PyramidParams = field(default_factory=PyramidParams)
    reinvestment: ReinvestmentParams = field(default_factory=ReinvestmentParams)
    scanner: ScannerParams = field(default_factory=ScannerParams)
    htf_intervals: list[str] = field(default_factory=lambda: ["60", "240"])
    v2: V2Config = field(default_factory=V2Config)


def _apply_dict(obj: Any, data: dict) -> None:
    """Recursively apply dict values to a dataclass instance."""
    for key, val in data.items():
        if hasattr(obj, key):
            current = getattr(obj, key)
            if isinstance(current, (APlusParams, TPRsiParams, KAMAParams,
                                    FilterParams, ScoringWeights, EntryParams,
                                    ExitParams, PyramidParams, ReinvestmentParams,
                                    ScannerParams, V2Config)):
                if isinstance(val, dict):
                    _apply_dict(current, val)
            else:
                setattr(obj, key, val)


def load_config(path: Path | str | None = None) -> BotConfig:
    """Load configuration from YAML file, falling back to defaults."""
    if path is None:
        path = PROJECT_ROOT / "config.yaml"
    path = Path(path)

    cfg = BotConfig()

    if path.exists():
        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        # Top-level scalars
        for key in ("symbols", "capital_usd", "leverage", "max_positions",
                     "max_positions_per_coin", "max_drawdown_pct"):
            if key in raw:
                setattr(cfg, key, raw[key])

        # Scoring section
        scoring_raw = raw.get("scoring", {})
        if "min_score" in scoring_raw:
            cfg.min_score = scoring_raw["min_score"]
        if "pyramid_min_score" in scoring_raw:
            cfg.pyramid_min_score = scoring_raw["pyramid_min_score"]
        if "weights" in scoring_raw:
            _apply_dict(cfg.scoring, scoring_raw["weights"])

        # Sub-sections
        for section, attr in [
            ("aplus", "aplus"), ("tp_rsi", "tp_rsi"), ("kama", "kama"),
            ("filters", "filters"), ("entry", "entry"), ("exit", "exit"),
            ("pyramid", "pyramid"), ("reinvestment", "reinvestment"),
            ("scanner", "scanner"), ("v2", "v2"),
        ]:
            if section in raw and isinstance(raw[section], dict):
                _apply_dict(getattr(cfg, attr), raw[section])

        # HTF
        if "htf" in raw and "intervals" in raw["htf"]:
            cfg.htf_intervals = raw["htf"]["intervals"]

    return cfg
