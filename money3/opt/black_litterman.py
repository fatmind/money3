from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import cvxpy as cp
import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, black_litterman, risk_models


@dataclass(frozen=True)
class BLResult:
    weights: Dict[str, float]
    posterior_returns: pd.Series
    covariance: pd.DataFrame
    prior_returns: pd.Series
    delta: float
    P: pd.DataFrame
    Q: pd.Series
    omega: pd.DataFrame


def optimize_with_black_litterman(
    prices: pd.DataFrame,
    views_obj: Dict[str, object],
    *,
    baseline_weights: Optional[Dict[str, float]] = None,
    tau: float = 0.05,
    risk_free_rate: float = 0.02,
) -> BLResult:
    """执行 Black-Litterman 组合优化。"""
    _validate_prices(prices)

    tickers = list(prices.columns)
    baseline_weights = _normalize_baseline_weights(baseline_weights, tickers)

    returns = prices.pct_change().dropna()
    
    # 确保收益率数据干净：移除任何包含 NaN/inf 的行
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    
    if returns.empty or len(returns) < 10:
        raise ValueError(
            f"Insufficient return data: {len(returns)} rows. "
            "Need at least 10 trading days for stable covariance estimation."
        )
    
    # 如果数据量太少（< 60 个交易日），使用简单协方差矩阵；否则使用 Ledoit-Wolf 收缩
    if len(returns) < 60:
        cov = returns.cov()
        # 数据量少时，简单协方差矩阵可能不是正定的，需要正则化
        cov = _make_positive_definite(cov)
    else:
        cov = risk_models.CovarianceShrinkage(returns).ledoit_wolf()
    
    # 确保协方差矩阵没有 NaN/inf
    if cov.isna().any().any() or np.isinf(cov.values).any():
        raise ValueError("Covariance matrix contains NaN or inf values. Check input data quality.")
    
    # 最终确保协方差矩阵是正定的
    cov = _make_positive_definite(cov)
    
    delta = black_litterman.market_implied_risk_aversion(returns, risk_free_rate=risk_free_rate)
    
    # 确保 delta 是标量值
    if isinstance(delta, pd.Series):
        delta = float(delta.iloc[0] if len(delta) > 0 else 2.0)
    else:
        delta = float(delta)
    
    # 验证 delta 是有效的
    if np.isnan(delta) or np.isinf(delta) or delta <= 0:
        # 如果 delta 无效，使用默认值
        delta = 2.0
        print(f"Warning: Invalid delta value. Using default delta={delta}.")
    
    market_caps = _weights_to_market_caps(baseline_weights)
    prior = black_litterman.market_implied_prior_returns(market_caps, delta, cov)
    
    # 验证先验收益
    if prior.isna().any() or np.isinf(prior.values).any():
        raise ValueError("Prior returns contain NaN or inf values. Check market_caps and delta.")

    P, Q, omega = _map_views_to_matrices(views_obj, tickers, cov, tau)
    
    # 验证 P, Q, omega 矩阵
    if not P.empty:
        if P.isna().any().any() or np.isinf(P.values).any():
            raise ValueError("P matrix contains NaN or inf values.")
    if not Q.empty:
        if Q.isna().any() or np.isinf(Q.values).any():
            raise ValueError("Q vector contains NaN or inf values.")
    if not omega.empty:
        if omega.isna().any().any() or np.isinf(omega.values).any():
            raise ValueError("Omega matrix contains NaN or inf values.")
        # 确保 omega 是正定的
        omega = _make_positive_definite(omega)

    bl_model = black_litterman.BlackLittermanModel(
        cov_matrix=cov,
        P=P if not P.empty else None,
        Q=Q if not Q.empty else None,
        pi=prior,
        tau=tau,
        omega=omega if not omega.empty else None,
        absolute_returns=False,
    )
    bl_returns = bl_model.bl_returns()
    
    # 验证后验收益：确保没有 NaN/inf
    if bl_returns.isna().any() or np.isinf(bl_returns.values).any():
        # 如果后验收益有问题，尝试使用先验收益作为后备
        print("Warning: Posterior returns contain NaN/inf. Using prior returns as fallback.")
        bl_returns = prior
    
    # 再次确保协方差矩阵是正定的（BL 模型可能修改了协方差矩阵）
    cov = _make_positive_definite(cov)

    # Use weight_bounds to enforce non-negativity; EF already enforces sum(weights) == 1
    ef = EfficientFrontier(bl_returns, cov, weight_bounds=(0.0, 1.0))

    _apply_position_constraints(ef, tickers)

    ef.max_sharpe(risk_free_rate=risk_free_rate)
    cleaned_weights = ef.clean_weights()

    return BLResult(
        weights=cleaned_weights,
        posterior_returns=bl_returns,
        covariance=cov,
        prior_returns=prior,
        delta=delta,
        P=P,
        Q=Q,
        omega=omega,
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _make_positive_definite(cov: pd.DataFrame, reg: float = 1e-6) -> pd.DataFrame:
    """确保协方差矩阵是正定的（positive definite）。
    
    如果矩阵不是正定的，添加小的正则化项（对角线元素增加 reg）。
    使用更稳健的方法：先尝试 Cholesky 分解，如果失败则添加正则化项。
    """
    cov_np = cov.values.copy()
    
    # 尝试 Cholesky 分解，如果成功则矩阵是正定的
    try:
        np.linalg.cholesky(cov_np)
        return cov
    except np.linalg.LinAlgError:
        # Cholesky 分解失败，矩阵不是正定的，需要正则化
        pass
    
    # 计算特征值
    eigenvalues = np.linalg.eigvals(cov_np)
    min_eigenvalue = np.min(eigenvalues.real)  # 只取实部
    
    if min_eigenvalue <= 0:
        # 添加正则化项使矩阵正定
        n = cov_np.shape[0]
        # 使用更小的正则化项，避免过度修改矩阵
        regularization = max(abs(min_eigenvalue) + reg, reg)
        cov_np = cov_np + np.eye(n) * regularization
        
        # 再次验证
        try:
            np.linalg.cholesky(cov_np)
        except np.linalg.LinAlgError:
            # 如果还是失败，使用更大的正则化项
            cov_np = cov_np + np.eye(n) * reg
        
        return pd.DataFrame(cov_np, index=cov.index, columns=cov.columns)
    
    return cov


def _validate_prices(prices: pd.DataFrame) -> None:
    if prices is None or prices.empty:
        raise ValueError("Price data is empty; cannot run Black-Litterman.")
    if prices.shape[1] < 2:
        raise ValueError("Black-Litterman needs at least two assets.")


def _normalize_baseline_weights(
    baseline_weights: Optional[Dict[str, float]],
    tickers: Iterable[str],
) -> Dict[str, float]:
    default_weights = {"SPY": 0.6, "TLT": 0.3, "GLD": 0.1}
    weights = {t: float(default_weights.get(t, 0.0)) for t in tickers}
    if baseline_weights:
        for ticker, value in baseline_weights.items():
            if ticker in weights:
                weights[ticker] = float(value)

    total = sum(weights.values())
    if total <= 0:
        raise ValueError("Baseline weights must sum to a positive value.")
    return {ticker: w / total for ticker, w in weights.items()}


def _weights_to_market_caps(weights: Dict[str, float]) -> Dict[str, float]:
    total_cap = 1_000_000_000.0
    return {ticker: w * total_cap for ticker, w in weights.items()}


def _map_views_to_matrices(
    views_obj: Dict[str, object],
    tickers: List[str],
    cov: pd.DataFrame,
    tau: float,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    rows: List[List[float]] = []
    targets: List[float] = []
    confidences: List[float] = []

    for raw_view in views_obj.get("views", []):
        view_type = raw_view.get("type")
        confidence = float(raw_view.get("confidence", 0.5))
        confidence = max(min(confidence, 0.999), 0.001)

        row = [0.0] * len(tickers)

        if view_type == "absolute":
            asset = raw_view.get("asset")
            if asset in tickers:
                row[tickers.index(asset)] = 1.0
                target = float(raw_view.get("expected_return", 0.0))
            else:
                continue
        elif view_type == "relative":
            assets = raw_view.get("assets", [])
            if len(assets) != 2:
                continue
            a, b = assets
            if a in tickers and b in tickers:
                row[tickers.index(a)] = 1.0
                row[tickers.index(b)] = -1.0
                target = float(raw_view.get("outperformance", 0.0))
            else:
                continue
        else:
            continue

        rows.append(row)
        targets.append(target)
        confidences.append(confidence)

    if not rows:
        return pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame()

    P = pd.DataFrame(rows, columns=tickers)
    Q = pd.Series(targets, name="Q")

    base_omega = P.to_numpy() @ cov.to_numpy() @ P.to_numpy().T
    # np.diag can return a read-only view in some contexts; create a writable copy
    diag = np.diag(base_omega).astype(float).copy()
    diag = np.maximum(diag, 1e-6)
    
    # 确保 base_omega 本身是正定的（用于数值稳定性）
    if np.any(np.isnan(base_omega)) or np.any(np.isinf(base_omega)):
        # 如果 base_omega 有问题，使用对角矩阵作为后备
        base_omega = np.eye(len(diag)) * diag.mean()

    confidence_scale = (1.0 - np.array(confidences)) + 1e-3
    omega_diag = tau * diag * confidence_scale
    omega_diag = np.maximum(omega_diag, 1e-6)
    
    # 确保 omega_diag 没有异常值
    omega_diag = np.clip(omega_diag, 1e-6, 1e6)

    omega = pd.DataFrame(np.diag(omega_diag), columns=P.index, index=P.index)
    return P, Q, omega


def _apply_position_constraints(ef: EfficientFrontier, tickers: List[str]) -> None:
    """应用仓位约束：
    - SPY ≤ 70%
    - TLT + GLD ≥ 30%
    - 每个资产 ≤ 80% (避免过度集中)
    """
    # SPY 上限约束
    if "SPY" in tickers:
        spy_idx = tickers.index("SPY")
        ef.add_constraint(lambda w, idx=spy_idx: w[idx] <= 0.7)

    # TLT + GLD 下限约束
    bond_gold = [t for t in ["TLT", "GLD"] if t in tickers]
    if len(bond_gold) == 2:
        gold_idx, tlt_idx = tickers.index("GLD"), tickers.index("TLT")

        ef.add_constraint(
            lambda w, i=gold_idx, j=tlt_idx: cp.sum(cp.hstack([w[i], w[j]])) >= 0.3
        )
    
    # 单个资产上限约束（避免过度集中，SPY 已有 0.7 上限，这里只约束 GLD 和 TLT）
    max_single_weight = 0.8
    for ticker in ["GLD", "TLT"]:
        if ticker in tickers:
            idx = tickers.index(ticker)
            ef.add_constraint(lambda w, i=idx, max_w=max_single_weight: w[i] <= max_w)

