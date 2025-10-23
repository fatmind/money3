from src import __version__
from src.data import DataClient
from src.llm import LLMClient
from src.opt import Optimizer
from src.backtest.backtester import Backtester


def main() -> None:
    
    # 读取 .env（若存在）
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
    
    # 链式依赖：data -> llm -> opt
    data = DataClient(["SPY", "GLD", "TLT"])  # 数据：1年行情+宏观；回测价
    llm = LLMClient()  # LLM：DeepSeek（无配置时固定 mock）
    opt = Optimizer(llm, data)  # 优化：Black-Litterman

    # 构造给 LLM 的纯文本上下文（简要提示，后续可替换为更丰富内容）
    llm_text = (
        "目标：年化15%、最大回撤10%、月度调仓。请输出 JSON views，包含绝对与相对观点。"
    )

    result = opt.optimize(llm_text)

    # 回测：2015-2025，月度近似再平衡，费率0.1%
    prices_bt = data.fetch_backtest_prices("2015-01-01", "2025-12-31")
    backtester = Backtester(fee_rate=0.001)
    report = backtester.run(prices_bt, {k: float(v) for k, v in result.get("weights", {}).items()})

    print(f"money3 {__version__}")
    print("Optimized Weights:", result.get("weights"))
    print("Views Used:", result.get("views_used"))
    print("CAGR:", round(report.cagr, 4))
    print("Max Drawdown:", round(report.max_drawdown, 4))
    print("Sharpe:", round(report.sharpe, 4))
    print("Volatility:", round(report.volatility, 4))


if __name__ == "__main__":
    main()
