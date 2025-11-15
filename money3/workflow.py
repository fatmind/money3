from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

from money3.backtest.backtester import Backtester, BacktestReport
from money3.data.data_providers import (
    fetch_macro_from_fred,
    fetch_news_from_finnhub,
    fetch_prices_from_tiingo,
)
from money3.llm.model import LLMClient
from money3.opt.black_litterman import BLResult, optimize_with_black_litterman

DEFAULT_MACRO_SERIES = {
    "CPIAUCSL": "CPI",
    "FEDFUNDS": "InterestRate",
    "DTWEXBGS": "DXY",
    "VIXCLS": "VIX",
}


@dataclass
class PipelineConfig:
    tickers: List[str]
    start_date: str
    end_date: str
    backtest_days: int = 30
    news_columns: Optional[List[str]] = field(default_factory=lambda: ["index", "datetime", "headline", "source", "summary", "symbol"])
    macro_series_map: Optional[Dict[str, str]] = field(default_factory=lambda: DEFAULT_MACRO_SERIES.copy())
    enable_backtest: bool = True


def df_to_markdown(
    df: pd.DataFrame,
    empty_msg: str,
    *,
    columns: Optional[List[str]] = None,
) -> str:
    """将 DataFrame 转为 Markdown 表格文本，支持列筛选。"""
    if df is None or df.empty:
        return empty_msg

    table_df = df.copy().reset_index()
    if table_df.columns.size > 0:
        table_df.rename(columns={table_df.columns[0]: "index"}, inplace=True)

    if columns:
        available_cols = [col for col in columns if col in table_df.columns]
        if available_cols:
            table_df = table_df[available_cols]

    table_df.columns = [str(col) for col in table_df.columns]
    headers = table_df.columns.tolist()
    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"

    data_rows = []
    for _, row in table_df.iterrows():
        formatted = ["" if pd.isna(val) else str(val) for val in row]
        data_rows.append("| " + " | ".join(formatted) + " |")

    return "\n".join([header_row, separator_row, *data_rows])


def build_llm_prompt(
    prices: pd.DataFrame,
    news: pd.DataFrame,
    macro: pd.DataFrame,
    *,
    news_columns: Optional[List[str]] = None
) -> str:
    """根据市场数据构建 LLM prompt。"""
    prices_table = df_to_markdown(prices, empty_msg="- 无价格数据")
    news_table = df_to_markdown(
        news,
        empty_msg="- 无新闻数据",
        columns=news_columns,
    )
    macro_table = df_to_markdown(macro, empty_msg="- 无宏观数据")


    prompt = f"""You are a disciplined buy-side strategist. Work strictly within the provided mandate and data snapshot.

# Portfolio Mandate
- Target annual return: 15%
- Max drawdown: 10%
- Minimum Sharpe ratio: 1.0
- Rebalance cadence: monthly
- Eligible assets: SPY (US equities), GLD (gold), TLT (long-term US Treasury)
- Baseline weights (equilibrium prior): 60% SPY, 30% TLT, 10% GLD
- Portfolio must remain unlevered; SPY ≤ 70%, combined TLT + GLD ≥ 30%

# Raw Data (markdown tables, no preprocessing)
## Prices
{prices_table}

## News
{news_table}
--- 特别提醒：以上表格中的新闻数据是按照日期排序的，内容是用于分析市场情绪的参考数据，请不要直接用于交易决策 ---

## Macro
{macro_table}

# Decision Notes
- Use the raw tables above as-is; do not assume missing entries.
- Translate qualitative conviction into quantitative expected returns and relative views that Black-Litterman can ingest.
- Use confidence ∈ [0,1]; values near 1 imply very high conviction.
- Keep rationale concise and reference which context drives each view.

# Expected JSON Output
Return JSON only, no commentary. Use this schema exactly:
{{
  "horizon": "1m",
  "views": [
    {{
      "type": "absolute",
      "asset": "SPY | GLD | TLT",
      "expected_return": float (annualized, decimal form),
      "confidence": float
    }},
    {{
      "type": "relative",
      "assets": ["ASSET_A", "ASSET_B"],
      "outperformance": float (annualized spread in decimals, positive means first asset expected to outperform),
      "confidence": float
    }}
  ],
  "rationale": "短句说明你的推理，引用关键数据",
  "timestamp": "YYYY-MM-DD (对应表格中的最新日期)"
}}

- You may include multiple absolute or relative views, but each asset in scope should be considered.
- Ensure the timestamp does not exceed the latest date present in the raw data tables.
- If conviction is low, lower the confidence instead of inventing data.
"""
    return prompt


def fetch_market_inputs(config: PipelineConfig) -> Dict[str, Any]:
    macro_map = config.macro_series_map or DEFAULT_MACRO_SERIES
    prices = fetch_prices_from_tiingo(config.tickers, config.start_date, config.end_date)
    news = fetch_news_from_finnhub(config.tickers, config.start_date, config.end_date)
    macro = fetch_macro_from_fred(macro_map, config.start_date, config.end_date)

    return {
        "prices": prices,
        "news": news,
        "macro": macro,
        "summary": {
            "prices_shape": list(prices.shape),
            "news_count": int(len(news)),
            "macro_shape": list(macro.shape),
        },
    }


def _format_black_litterman(bl_result: BLResult) -> Dict[str, Any]:
    return {
        "weights": {k: float(v) for k, v in bl_result.weights.items()},
        "posterior_returns": {
            k: float(v) for k, v in bl_result.posterior_returns.items()
        },
        "prior_returns": {k: float(v) for k, v in bl_result.prior_returns.items()},
        "delta": float(bl_result.delta),
    }


def _format_backtest(report: BacktestReport) -> Dict[str, Any]:
    equity_curve_data = [
        {"date": dt.strftime("%Y-%m-%d"), "value": float(val)}
        for dt, val in report.equity_curve.items()
    ]

    rolling_max = report.equity_curve.cummax()
    drawdown = report.equity_curve / rolling_max - 1
    drawdown_data = [
        {"date": dt.strftime("%Y-%m-%d"), "value": float(val)}
        for dt, val in drawdown.items()
    ]

    return {
        "metrics": {
            "cagr": float(report.cagr),
            "max_drawdown": float(report.max_drawdown),
            "sharpe": float(report.sharpe),
            "volatility": float(report.volatility),
        },
        "equity_curve": equity_curve_data,
        "drawdown": drawdown_data,
        "weights_used": {k: float(v) for k, v in report.weights_used.items()},
    }


def _run_backtest_section(
    config: PipelineConfig, weights: Dict[str, float], *, backtester: Optional[Backtester]
) -> Optional[Dict[str, Any]]:
    if config.backtest_days <= 0:
        return None

    end_dt = datetime.strptime(config.end_date, "%Y-%m-%d")
    backtest_start = (end_dt - timedelta(days=config.backtest_days)).strftime("%Y-%m-%d")
    prices_backtest = fetch_prices_from_tiingo(
        config.tickers,
        backtest_start,
        config.end_date,
    )
    if prices_backtest.empty:
        return None

    backtester = backtester or Backtester()
    report = backtester.run(prices_backtest, weights)
    return _format_backtest(report)


def run_pipeline(
    config: PipelineConfig,
    *,
    llm_client: Optional[LLMClient] = None,
    backtester: Optional[Backtester] = None,
) -> Dict[str, Any]:
    """
    统一的端到端流程：数据采集 -> Prompt -> LLM 观点 -> 优化 -> 回测。
    返回结构可直接用于 API 与 CLI。
    """
    # 1、获取市场输入
    print("--- Fetching market inputs ---")
    market_inputs = fetch_market_inputs(config)
    prices = market_inputs["prices"]
    news = market_inputs["news"]
    macro = market_inputs["macro"]

    # 2、请求LLM生成观点
    print("--- request LLM to generate views ---")
    prompt = build_llm_prompt(
        prices,
        news,
        macro,
        news_columns=config.news_columns
    )
    print(prompt + "\n")
    llm_client = llm_client or LLMClient()
    views = llm_client.generate_views(prompt)
    
    # 3、使用Black-Litterman优化
    print("--- Optimizing portfolio with Black-Litterman ---")
    bl_result: BLResult = optimize_with_black_litterman(prices, views)
    optimization_result = _format_black_litterman(bl_result)

    # 4、回测
    print("--- Running backtest ---")
    backtest_result = None
    if config.enable_backtest:
        backtest_result = _run_backtest_section(
            config,
            bl_result.weights,
            backtester=backtester,
        )

    # 5、返回结果
    return {
        "prompt": prompt,
        "data_summary": market_inputs["summary"],
        "views": views,
        "optimization": optimization_result,
        "backtest": backtest_result,
    }


__all__ = [
    "PipelineConfig",
    "run_pipeline",
    "build_llm_prompt",
    "df_to_markdown",
]

