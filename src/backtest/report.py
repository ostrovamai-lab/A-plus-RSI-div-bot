"""Backtest report — print stats and plot equity curve."""

from __future__ import annotations

from collections import defaultdict

from models import BacktestResult


def print_report(result: BacktestResult, symbol: str = "") -> None:
    """Print formatted backtest summary to stdout."""
    print(f"\n{'=' * 60}")
    print(f"  A+ RSI DIVERGENCE BACKTEST — {symbol}")
    print(f"{'=' * 60}")
    print(f"  Total Trades:    {result.total_trades}")
    print(f"  Winning:         {result.winning_trades} ({result.win_rate:.1f}%)")
    print(f"  Losing:          {result.losing_trades}")
    print(f"  Avg Win:         ${result.avg_win:.2f}")
    print(f"  Avg Loss:        ${result.avg_loss:.2f}")
    print(f"{'─' * 60}")
    print(f"  Gross PnL:       ${result.total_pnl:.2f}")
    print(f"  Total Fees:      ${result.total_fees:.2f}")
    print(f"  Net PnL:         ${result.net_pnl:.2f}")
    print(f"  Profit Factor:   {result.profit_factor:.2f}")
    print(f"  Max Drawdown:    {result.max_drawdown_pct:.2f}%")
    print(f"  Sharpe Ratio:    {result.sharpe_ratio:.2f}")
    print(f"{'─' * 60}")

    # Breakdown by close reason
    reason_stats: dict[str, dict] = defaultdict(lambda: {"count": 0, "pnl": 0.0})
    for t in result.trades:
        r = t.reason or "unknown"
        reason_stats[r]["count"] += 1
        reason_stats[r]["pnl"] += t.pnl - t.fee
    print("  Breakdown by close reason:")
    for reason, stats in sorted(reason_stats.items(), key=lambda x: x[1]["pnl"]):
        print(f"    {reason:20s}  n={stats['count']:5d}  pnl=${stats['pnl']:+.2f}")

    # Score distribution
    if result.trades:
        scores = [t.score for t in result.trades if t.score > 0]
        if scores:
            avg_score = sum(scores) / len(scores)
            print(f"  Avg entry score:   {avg_score:.1f}")

    # Grid fill stats
    fills = [t.grid_fills for t in result.trades]
    if fills:
        avg_fills = sum(fills) / len(fills)
        print(f"  Avg grid fills:    {avg_fills:.1f}")

    print(f"{'=' * 60}\n")


def plot_equity_curve(result: BacktestResult, symbol: str = "", save_path: str | None = None) -> None:
    """Plot equity curve using matplotlib."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from datetime import datetime
    except ImportError:
        print("matplotlib not available — skipping plot")
        return

    if not result.equity_curve:
        print("No equity data to plot")
        return

    dates = [datetime.utcfromtimestamp(ts / 1000) for ts in result.timestamps]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(dates, result.equity_curve, linewidth=1, color="#2196F3")
    ax.fill_between(dates, result.equity_curve, alpha=0.1, color="#2196F3")
    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")

    ax.set_title(f"A+ RSI Divergence Bot — {symbol} Equity Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity ($)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()

    # Annotate key stats
    stats_text = (
        f"Net PnL: ${result.net_pnl:.2f}\n"
        f"Win Rate: {result.win_rate:.1f}%\n"
        f"Trades: {result.total_trades}\n"
        f"Max DD: {result.max_drawdown_pct:.1f}%\n"
        f"Sharpe: {result.sharpe_ratio:.2f}"
    )
    ax.text(
        0.02, 0.98, stats_text, transform=ax.transAxes,
        fontsize=9, verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Chart saved to {save_path}")
    else:
        plt.show()
