from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from pypfopt import black_litterman, risk_models, EfficientFrontier
from money3.llm import LLMClient
from money3.data.TingoData import TingoData


class Optimizer:
    """Black-Litterman 优化器（基于 PyPortfolioOpt）。"""

    def __init__(self, llm_client: LLMClient, data_client: TingoData) -> None:
        self.llm_client = llm_client
        self.data_client = data_client

    def optimize(self, views_obj: Dict[str, Any]) -> Dict[str, Any]:
        # 1) 获取最近 1 年价格，估计协方差与先验
        market_data = self.data_client.fetch_market_data()
        prices_recent = market_data.prices
        
        returns = prices_recent.pct_change().dropna()
        cov = risk_models.CovarianceShrinkage(returns).ledoit_wolf()

        market_caps = self._equilibrium_market_caps(prices_recent.columns)
        delta = black_litterman.market_implied_risk_aversion(returns)
        prior = black_litterman.market_implied_prior_returns(market_caps, delta, cov)

        # 2) 将 LLM views 映射为 P、Q、Omega
        P, Q, omega = self._map_views(views_obj, prices_recent.columns, tau=0.05)

        # 3) 后验期望收益
        bl_returns = black_litterman.black_litterman_return(
            cov_matrix=cov,
            market_prior=prior,
            P=P,
            Q=Q,
            omega=omega,
            tau=0.05,
        )

        # 4) 约束优化
        ef = EfficientFrontier(bl_returns, cov)
        ef.add_constraint(lambda w: w >= 0)
        ef.add_constraint(lambda w: np.sum(w) == 1)
        
        ticker_list = list(prices_recent.columns)
        if "SPY" in ticker_list:
            spy_idx = ticker_list.index("SPY")
            ef.add_constraint(lambda w, i=spy_idx: w[i] <= 0.7)
        
        # 债券+黄金 不低于 0.3
        for lower, name in [(0.0, "GLD"), (0.0, "TLT")]:
            if name in ticker_list:
                idx = ticker_list.index(name)
                ef.add_constraint(lambda w, i=idx, l=lower: w[i] >= l)

        raw_weights = ef.max_sharpe()
        cleaned = ef.clean_weights()

        return {
            "weights": cleaned,
            "views_used": views_obj,
            "status": "ok",
        }

    # ---- 工具方法 ----
    def _equilibrium_market_caps(self, tickers: List[str]) -> Dict[str, float]:
        # 简化：用基准 60/30/10 作为等效市值权重
        weights = {"SPY": 0.6, "TLT": 0.3, "GLD": 0.1}
        caps = {t: float(weights.get(t, 1.0 / len(tickers))) for t in tickers}
        return caps

    def _map_views(self, views_obj: Dict[str, Any], columns: pd.Index, tau: float) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        tickers = list(columns)
        n = len(tickers)
        P_rows: List[List[float]] = []
        Q_vals: List[float] = []
        confs: List[float] = []

        for v in views_obj.get("views", []):
            vtype = v.get("type")
            conf = float(v.get("confidence", 0.5))
            if vtype == "absolute":
                asset = v.get("asset")
                exp_ret = float(v.get("expected_return", 0.0))
                row = [0.0] * n
                if asset in tickers:
                    row[tickers.index(asset)] = 1.0
                P_rows.append(row)
                Q_vals.append(exp_ret)
                confs.append(conf)
            elif vtype == "relative":
                assets = v.get("assets", [])
                outperf = float(v.get("outperformance", 0.0))
                row = [0.0] * n
                if len(assets) == 2 and all(a in tickers for a in assets):
                    i, j = tickers.index(assets[0]), tickers.index(assets[1])
                    row[i] = 1.0
                    row[j] = -1.0
                P_rows.append(row)
                Q_vals.append(outperf)
                confs.append(conf)

        if not P_rows:
            P = pd.DataFrame(np.zeros((1, n)), columns=tickers)
            Q = pd.Series([0.0])
            omega = pd.DataFrame(np.eye(1) * 1e-6)
            return P, Q, omega

        P = pd.DataFrame(P_rows, columns=tickers)
        Q = pd.Series(Q_vals)

        confs_arr = np.array(confs, dtype=float)
        eps = 1e-6
        var_scale = (1.0 - confs_arr + eps)
        omega = np.diag(var_scale * Q.var() if len(Q) > 1 else var_scale * 1e-4)
        omega = pd.DataFrame(omega)
        return P, Q, omega
