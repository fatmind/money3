from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class BacktestReport:
    cagr: float
    max_drawdown: float
    sharpe: float
    volatility: float
    equity_curve: pd.Series
    weights_used: Dict[str, float]


class Backtester:
    """简化回测器（优先使用 vectorbt，不可用时使用 numpy 近似）。

    - 月度再平衡到固定权重
    - 交易成本：每次再平衡按 0.1% 对组合净值收取（近似）
    """

    def __init__(self, fee_rate: float = 0.001) -> None:
        self.fee_rate = float(fee_rate)

    def run(self, prices: pd.DataFrame, weights: Dict[str, float]) -> BacktestReport:
        prices = prices.copy()
        if not isinstance(prices.index, pd.DatetimeIndex):
            prices.index = pd.to_datetime(prices.index, errors="coerce")
        prices = prices.sort_index()

        price_columns = [c for c in weights.keys() if c in prices.columns]
        prices = prices.loc[:, price_columns]
        prices = prices.dropna(how="any")
        if prices.empty:
            raise ValueError("价格数据为空或不含指定权重的资产列")

        w = pd.Series(weights, dtype=float)
        w = w / w.sum()

        # 日收益
        daily_ret = prices.pct_change().dropna()
        # 近似：连续保持目标权重（等效于高频再平衡）
        port_ret = (daily_ret * w.reindex(daily_ret.columns).fillna(0.0)).sum(axis=1)

        # 在每个自然月首个可交易日收取再平衡费用（近似）
        month_periods = port_ret.index.to_period("M")
        first_trading_days = port_ret.groupby(month_periods).head(1).index
        rebal_days = pd.Series(False, index=port_ret.index)
        rebal_days.loc[first_trading_days] = True
        fee = pd.Series(0.0, index=port_ret.index)
        fee.loc[rebal_days] = self.fee_rate
        port_ret_net = port_ret - fee

        equity = (1.0 + port_ret_net).cumprod()

        # 绩效指标
        total_days = (equity.index[-1] - equity.index[0]).days
        years = max(total_days / 365.25, 1e-9)
        cagr = float(equity.iloc[-1] ** (1 / years) - 1)

        rolling_max = equity.cummax()
        drawdown = equity / rolling_max - 1
        mdd = float(drawdown.min())

        vol = float(port_ret_net.std() * np.sqrt(252))
        mean_daily = float(port_ret_net.mean())
        sharpe = float((mean_daily * 252) / vol) if vol > 0 else 0.0

        return BacktestReport(
            cagr=cagr,
            max_drawdown=abs(mdd),
            sharpe=sharpe,
            volatility=vol,
            equity_curve=equity,
            weights_used=w.to_dict(),
        )


def ensure_output_dir(name: str = "outputs") -> Path:
    path = Path(name)
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_equity_curve(equity: pd.Series, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    equity.plot(ax=ax, color="#1f77b4", linewidth=1.8)
    ax.set_title("Portfolio Equity Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Net Value")
    ax.grid(alpha=0.3)
    path = output_dir / "equity_curve.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_drawdown(equity: pd.Series, output_dir: Path) -> Path:
    drawdown = equity / equity.cummax() - 1.0
    fig, ax = plt.subplots(figsize=(8, 3))
    drawdown.plot(ax=ax, color="#d62728", linewidth=1.5)
    ax.fill_between(drawdown.index, drawdown, 0, color="#d62728", alpha=0.2)
    ax.set_title("Drawdown Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.grid(alpha=0.3)
    path = output_dir / "drawdown_curve.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_weights(weights: Dict[str, float], output_dir: Path) -> Path:
    series = pd.Series(weights).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    series.plot(kind="bar", ax=ax, color="#2ca02c")
    ax.set_title("Optimized Weights")
    ax.set_xlabel("Asset")
    ax.set_ylabel("Weight")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    path = output_dir / "weights.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def render_equity_curve(equity: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    equity.plot(ax=ax, color="#1f77b4", linewidth=1.8)
    ax.set_title("Portfolio Equity Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Net Value")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def render_drawdown(equity: pd.Series) -> None:
    drawdown = equity / equity.cummax() - 1.0
    fig, ax = plt.subplots(figsize=(8, 3))
    drawdown.plot(ax=ax, color="#d62728", linewidth=1.5)
    ax.fill_between(drawdown.index, drawdown, 0, color="#d62728", alpha=0.2)
    ax.set_title("Drawdown Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def render_weights(weights: Dict[str, float]) -> None:
    series = pd.Series(weights).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    series.plot(kind="bar", ax=ax, color="#2ca02c")
    ax.set_title("Optimized Weights")
    ax.set_xlabel("Asset")
    ax.set_ylabel("Weight")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    plt.show()
    plt.close(fig)


__all__ = [
    "Backtester",
    "BacktestReport",
    "ensure_output_dir",
    "plot_equity_curve",
    "plot_drawdown",
    "plot_weights",
    "render_equity_curve",
    "render_drawdown",
    "render_weights",
]