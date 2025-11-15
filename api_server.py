"""
Portfolio Optimization API Server
提供 RESTful API 接口
"""

import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

from money3.backtest.backtester import Backtester, BacktestReport
from money3.data.data_providers import (
    fetch_macro_from_fred,
    fetch_news_from_finnhub,
    fetch_prices_from_tiingo,
)
from money3.llm.model import LLMClient
from money3.opt.black_litterman import BLResult, optimize_with_black_litterman

# 加载环境变量
load_dotenv()

app = Flask(__name__, static_folder='static')
CORS(app)  # 允许跨域请求


# ==================== Helper Functions ====================


def build_llm_prompt(
    prices: pd.DataFrame, news: pd.DataFrame, macro: pd.DataFrame
) -> str:
    """构建给 LLM 的 prompt"""

    def _df_to_markdown(df: pd.DataFrame, empty_msg: str) -> str:
        if df is None or df.empty:
            return empty_msg
        table_df = df.copy().reset_index()
        if table_df.columns.size > 0:
            table_df.rename(columns={table_df.columns[0]: "index"}, inplace=True)
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
    news_table = _df_to_markdown(news, empty_msg="- 无新闻数据")
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


# ==================== API Routes ====================


@app.route("/")
def index():
    """首页重定向到前端页面"""
    return send_from_directory('static', 'index.html')


@app.route("/static/<path:path>")
def serve_static(path):
    """提供静态文件"""
    return send_from_directory('static', path)


@app.route("/api/health", methods=["GET"])
def health_check():
    """健康检查"""
    return jsonify({"status": "ok", "message": "API server is running"})


@app.route("/api/optimize", methods=["POST"])
def optimize_portfolio():
    """
    投资组合优化主接口
    
    请求参数:
    {
        "tickers": ["SPY", "GLD", "TLT"],
        "start_date": "2025-09-01",
        "end_date": "2025-10-30",
        "backtest_days": 30
    }
    
    返回:
    {
        "status": "success",
        "data": {
            "views": {...},
            "optimization": {...},
            "backtest": {...}
        }
    }
    """
    try:
        # 解析请求参数
        data = request.get_json()
        tickers = data.get("tickers", ["SPY", "GLD", "TLT"])
        start_date = data.get("start_date")
        end_date = data.get("end_date")

        # 参数验证
        if not tickers or len(tickers) < 2:
            return jsonify({"status": "error", "message": "至少需要两个股票代码"}), 400

        if not start_date or not end_date:
            return (
                jsonify({"status": "error", "message": "开始日期和结束日期不能为空"}),
                400,
            )

        # Step 1: 获取数据
        print(f"[1/4] 获取数据: {tickers} from {start_date} to {end_date}")
        prices = fetch_prices_from_tiingo(tickers, start_date, end_date)
        news = fetch_news_from_finnhub(tickers, start_date, end_date)
        macro_series_map = {
            "CPIAUCSL": "CPI",
            "FEDFUNDS": "InterestRate",
            "DTWEXBGS": "DXY",
            "VIXCLS": "VIX",
        }
        macro = fetch_macro_from_fred(macro_series_map, start_date, end_date)

        data_summary = {
            "prices_shape": prices.shape,
            "news_count": len(news),
            "macro_shape": macro.shape,
        }
        print(f"数据获取完成: {data_summary}")

        # Step 2: 生成 Views
        print("[2/4] 生成投资观点...")
        prompt = build_llm_prompt(prices, news, macro)
        llm_client = LLMClient()
        views = llm_client.generate_views(prompt)
        print(f"Views 生成完成: {views}")

        # Step 3: 优化组合
        print("[3/4] 优化投资组合...")
        bl_result: BLResult = optimize_with_black_litterman(prices, views)

        optimization_result = {
            "weights": {k: float(v) for k, v in bl_result.weights.items()},
            "posterior_returns": {
                k: float(v) for k, v in bl_result.posterior_returns.items()
            },
            "prior_returns": {k: float(v) for k, v in bl_result.prior_returns.items()},
            "delta": float(bl_result.delta),
        }
        print(f"优化完成: {optimization_result['weights']}")

        # Step 4: 回测
        print("[4/4] 回测...")
        backtest_days = data.get("backtest_days", 30)
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        backtest_start = end_dt - timedelta(days=backtest_days)
        backtest_start_str = backtest_start.strftime("%Y-%m-%d")

        prices_backtest = fetch_prices_from_tiingo(
            tickers, backtest_start_str, end_date
        )

        backtest_result = None
        if not prices_backtest.empty:
            backtester = Backtester()
            report: BacktestReport = backtester.run(
                prices_backtest, bl_result.weights
            )

            # 转换为 JSON 可序列化格式
            equity_curve_data = [
                {
                    "date": dt.strftime("%Y-%m-%d"),
                    "value": float(val),
                }
                for dt, val in report.equity_curve.items()
            ]

            # 计算回撤数据
            rolling_max = report.equity_curve.cummax()
            drawdown = report.equity_curve / rolling_max - 1
            drawdown_data = [
                {
                    "date": dt.strftime("%Y-%m-%d"),
                    "value": float(val),
                }
                for dt, val in drawdown.items()
            ]

            backtest_result = {
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
            print(f"回测完成: CAGR={report.cagr:.2%}, Sharpe={report.sharpe:.2f}")

        # 返回完整结果
        return jsonify(
            {
                "status": "success",
                "data": {
                    "data_summary": data_summary,
                    "views": views,
                    "optimization": optimization_result,
                    "backtest": backtest_result,
                },
            }
        )

    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/test", methods=["GET"])
def test_with_sample_data():
    """使用示例数据进行测试（不调用外部 API）"""
    try:
        # 使用 bl_test_main.py 中的示例数据
        from money3.opt.bl_test_main import build_prices_df, build_views

        prices = build_prices_df()
        views = build_views()

        # 优化
        bl_result: BLResult = optimize_with_black_litterman(prices, views)

        result = {
            "weights": {k: float(v) for k, v in bl_result.weights.items()},
            "posterior_returns": {
                k: float(v) for k, v in bl_result.posterior_returns.items()
            },
        }

        return jsonify({"status": "success", "data": result})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    print("🚀 Starting Portfolio Optimization API Server...")
    print("📍 Server running at http://localhost:5000")
    print("📖 API Endpoints:")
    print("  - GET  /api/health")
    print("  - GET  /api/test")
    print("  - POST /api/optimize")
    app.run(host="0.0.0.0", port=5000, debug=True)

