from src import __version__
from src.data.TingoData import TingoData, MarketData
from src.llm import LLMClient
from src.opt import Optimizer
from src.backtest.backtester import Backtester

import json


def _build_llm_prompt(market_data: MarketData) -> str:
    """构造一个丰富的 LLM 输入。"""
    
    prices_weekly = market_data.prices.resample('W').last().tail(30).to_json(orient='split')
    macro_weekly = market_data.macro.resample('W').last().tail(30).to_json(orient='split')
    news_summary = market_data.news.head(10)[['publishedDate', 'title', 'tags']].to_json(orient='records')

    prompt = f"""
    You are an investment advisor. Generate investment views for a Black-Litterman model based on the following market data.
    
    **Objective:** Target 15% annual return, max 10% drawdown, monthly rebalancing.
    **Tickers:** {list(market_data.prices.columns)}
    **Recent Prices (Weekly):** {prices_weekly}
    **Recent Macro Indicators (Weekly, CPI/FEDFUNDS/DXY/VIX):** {macro_weekly}
    **Recent News:** {news_summary}

    **Required Output (JSON only):**
    {{
      "horizon": "1m",
      "views": [
        {{ "type": "absolute", "asset": "SPY", "expected_return": 0.18, "confidence": 0.7 }},
        {{ "type": "relative", "assets": ["SPY", "TLT"], "outperformance": 0.05, "confidence": 0.6 }}
      ],
      "rationale": "Your reasoning here."
    }}
    """
    return prompt

def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()

    data_client = TingoData(["SPY", "GLD", "TLT"])
    market_data = data_client.fetch_market_data()
    
    llm_prompt = _build_llm_prompt(market_data)
    
    print("="*20 + " LLM Prompt Preview " + "="*20)
    print(llm_prompt[:500] + "\n...")
    print("="*58)

    llm = LLMClient()
    views = llm.generate_views(llm_prompt)

    # 注意：Optimizer 依赖于 DataClient 的接口，这里我们传入 TingoData 实例
    # 如果 Optimizer 内部有强依赖 OpenBB 的逻辑，需要进一步修改
    opt = Optimizer(llm, data_client) 
    result = opt.optimize(views)

    prices_bt = data_client.fetch_backtest_prices()
    backtester = Backtester(fee_rate=0.001)
    report = backtester.run(prices_bt, {k: float(v) for k, v in result.get("weights", {}).items()})

    print(f"\n--- money3 v{__version__} ---")
    print("Optimized Weights:", result.get("weights"))
    print("Views Used:", json.dumps(result.get("views_used"), indent=2))
    print("\n--- Backtest Report ---")
    print(f"CAGR: {report.cagr:.4f}")
    print(f"Max Drawdown: {report.max_drawdown:.4f}")
    print(f"Sharpe Ratio: {report.sharpe:.4f}")
    print(f"Volatility: {report.volatility:.4f}")


if __name__ == "__main__":
    main()
