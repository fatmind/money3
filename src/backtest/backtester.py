from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

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
        prices = prices.dropna(how="any")
        prices = prices.loc[:, [c for c in weights.keys() if c in prices.columns]]
        if prices.empty:
            raise ValueError("价格数据为空或不含指定权重的资产列")

        w = pd.Series(weights, dtype=float)
        w = w / w.sum()

        # 日收益
        daily_ret = prices.pct_change().dropna()
        # 近似：连续保持目标权重（等效于高频再平衡）
        port_ret = (daily_ret * w.reindex(daily_ret.columns).fillna(0.0)).sum(axis=1)

        # 在每个自然月首日收取一次再平衡费用（近似）
        month_change = port_ret.index.to_period("M").to_timestamp()
        rebal_days = pd.Series(index=port_ret.index, dtype=bool).fillna(False)
        rebal_days.loc[month_change.drop_duplicates()] = True
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


