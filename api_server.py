"""
Portfolio Optimization API Server
提供 RESTful API 接口
"""

import traceback

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

from money3.opt.black_litterman import BLResult, optimize_with_black_litterman
from money3.workflow import PipelineConfig, run_pipeline

# 加载环境变量
load_dotenv()

app = Flask(__name__, static_folder='static')
CORS(app)  # 允许跨域请求


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
        data = request.get_json() or {}
        tickers = data.get("tickers", ["SPY", "GLD", "TLT"])
        start_date = data.get("start_date")
        end_date = data.get("end_date")
        backtest_days=data.get("backtest_days")

        # 参数验证
        if not tickers or len(tickers) < 2:
            return jsonify({"status": "error", "message": "至少需要两个股票代码"}), 400

        if not start_date or not end_date:
            return (
                jsonify({"status": "error", "message": "开始日期和结束日期不能为空"}),
                400,
            )

        config = PipelineConfig(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            backtest_days=backtest_days
        )

        print(f"[Pipeline] running with {config}")
        result = run_pipeline(config)
        print(f"[Pipeline] summary: {result['data_summary']}")

        return jsonify({"status": "success", "data": result})

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

