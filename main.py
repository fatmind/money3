from datetime import datetime, timedelta

from money3.workflow import PipelineConfig, run_pipeline


def main() -> None:
    """主流程"""

    # 加载环境变量
    from dotenv import load_dotenv
    load_dotenv()

    # 开始结束时间
    end_time_str = "2025-10-15"  # 视图生成时点之前最后一个交易日
    start_time_str = (datetime.strptime(end_time_str, "%Y-%m-%d") - timedelta(days=15)).strftime("%Y-%m-%d")

    tickers = ["SPY", "GLD", "TLT"]

    config = PipelineConfig(
        tickers=tickers,
        start_date=start_time_str,
        end_date=end_time_str,
        backtest_days=15
    )

    # 运行pipeline
    result = run_pipeline(config)
    optimization = result["optimization"]

    # 打印结果
    print("Optimal weights:")
    for asset, weight in optimization["weights"].items():
        print(f"  {asset}: {weight:.4%}")

    print("\nPosterior expected returns (annualised):")
    for asset, ret in optimization["posterior_returns"].items():
        print(f"  {asset}: {ret:.4f}")

    print("\nDiagnostic:")
    print(f"  Risk aversion (delta): {optimization['delta']:.4f}")
    views_section = result["views"]
    print(f"  Views incorporated: {len(views_section.get('views', []))}")

    backtest = result["backtest"]
    if backtest:
        print("\n--- Backtest Metrics ---")
        metrics = backtest["metrics"]
        print(f"  CAGR: {metrics['cagr']:.2%}")
        print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"  Sharpe: {metrics['sharpe']:.2f}")
        print(f"  Volatility: {metrics['volatility']:.2%}")
    else:
        print("\n(No backtest data available for the selected window.)")


if __name__ == "__main__":
    main()
