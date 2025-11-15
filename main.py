from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd

from money3.data.data_providers import (
    fetch_prices_from_tiingo,
    fetch_news_from_finnhub,
    fetch_macro_from_fred,
)
from money3.llm.model import LLMClient
from money3.opt.black_litterman import BLResult, optimize_with_black_litterman
from money3.backtest.backtester import (
    Backtester,
    BacktestReport,
    render_drawdown,
    render_equity_curve,
    render_weights,
)


def build_llm_prompt(prices: pd.DataFrame, news: pd.DataFrame, macro: pd.DataFrame) -> str:
    """根据市场数据构建给 LLM 的 prompt。"""

    def _df_to_markdown(
        df: pd.DataFrame,
        empty_msg: str,
        *,
        columns: Optional[List[str]] = None,
    ) -> str:
        if df is None or df.empty:
            return empty_msg

        table_df = df.copy()
        table_df = table_df.reset_index()
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

    prices_table = _df_to_markdown(prices, empty_msg="- 无价格数据")
    news_table = _df_to_markdown(
        news,
        empty_msg="- 无新闻数据",
        columns=["index", "datetime", "headline", "source", "summary", "symbol"],
    )
    macro_table = _df_to_markdown(macro, empty_msg="- 无宏观数据")

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


def main() -> None:
    """主流程"""

    # 加载环境变量
    from dotenv import load_dotenv
    load_dotenv()

    # 开始结束时间
    end_time_str = "2025-10-30"  # 视图生成时点之前最后一个交易日
    start_time_str = (datetime.strptime(end_time_str, "%Y-%m-%d") - timedelta(days=30)).strftime("%Y-%m-%d")

    backtest_end_str = "2025-10-31"  # 回测截止日期
    backtest_start_str = (datetime.strptime(backtest_end_str, "%Y-%m-%d") - timedelta(days=30)).strftime("%Y-%m-%d")

    # 1. 数据采集
    print("--- 1. Fetching Data ---")
    tickers = ["SPY", "GLD", "TLT"]
    prices = fetch_prices_from_tiingo(tickers, start_time_str, end_time_str)
    news = fetch_news_from_finnhub(tickers, start_time_str, end_time_str)
    macro_series_map = {
        "CPIAUCSL": "CPI",
        "FEDFUNDS": "InterestRate",
        "DTWEXBGS": "DXY",
        "VIXCLS": "VIX",
    }
    macro = fetch_macro_from_fred(macro_series_map, start_time_str, end_time_str)

    print("Data fetching complete.")
    print(f"Prices: {prices.shape}, News: {news.shape}, Macro: {macro.shape}")


    # 2. 构建 Prompt 并调用 LLM
    print("\n--- 2. Generating LLM Views ---")
    prompt = build_llm_prompt(prices, news, macro)
    
    # 打印 prompt 预览
    print("\n====== LLM Prompt Preview ======")
    print(prompt)

    # 3. 生成 LLM Views
    llm_client = LLMClient()
    views = llm_client.generate_views(prompt)

    print("\nLLM Views Generated:")
    print(views)

    # 4. 优化组合
    print("\n--- 3. Black-Litterman Optimizer ---")
    bl_result: BLResult = optimize_with_black_litterman(prices, views)

    print("Optimal weights:")
    for asset, weight in bl_result.weights.items():
        print(f"  {asset}: {weight:.4%}")

    print("\nPosterior expected returns (annualised):")
    for asset, ret in bl_result.posterior_returns.items():
        print(f"  {asset}: {ret:.4f}")

    print("\nDiagnostic:")
    print(f"  Risk aversion (delta): {bl_result.delta:.4f}")
    print(f"  Views incorporated: {len(bl_result.Q)}")

    # 5. 回测 & 可视化
    # prices_backtest = fetch_prices_from_tiingo(tickers, backtest_start_str, backtest_end_str)
    # if prices_backtest.empty:
    #     raise ValueError("回测区间价格数据为空，请检查时间范围")
    # print(f"Prices (backtest horizon): {prices_backtest.shape}")
    # print("\n--- 4. Backtest & Visualization ---")
    # backtester = Backtester()
    # report: BacktestReport = backtester.run(prices_backtest, bl_result.weights)

    # print("Backtest metrics:")
    # print(f"  CAGR: {report.cagr:.2%}")
    # print(f"  Max Drawdown: {report.max_drawdown:.2%}")
    # print(f"  Sharpe: {report.sharpe:.2f}")
    # print(f"  Volatility: {report.volatility:.2%}")

    # print("====== Displaying charts ======")
    # render_equity_curve(report.equity_curve)
    # render_drawdown(report.equity_curve)
    # render_weights(report.weights_used)


if __name__ == "__main__":
    main()
