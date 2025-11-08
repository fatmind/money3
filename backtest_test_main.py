"""单测：使用提供的价格数据和权重测试 backtester.run() 函数"""

import pandas as pd
from src.backtest.backtester import Backtester, BacktestReport


def build_prices_from_table() -> pd.DataFrame:
    """从提供的 markdown 表格构建价格 DataFrame，自动跳过空值行"""
    rows = [
        {"index": "2025-09-01 00:00:00", "SPY": None, "GLD": None, "TLT": None},
        {"index": "2025-09-02 00:00:00", "SPY": 638.5083883546, "GLD": 325.59, "TLT": 85.0252660992},
        {"index": "2025-09-03 00:00:00", "SPY": 641.9688411442, "GLD": 328.14, "TLT": 85.9586276562},
        {"index": "2025-09-04 00:00:00", "SPY": 647.334038841, "GLD": 326.69, "TLT": 86.6139666219},
        {"index": "2025-09-05 00:00:00", "SPY": 645.4592113931, "GLD": 331.05, "TLT": 87.9345739314},
        {"index": "2025-09-08 00:00:00", "SPY": 647.0448367347, "GLD": 334.82, "TLT": 89.1062405668},
        {"index": "2025-09-09 00:00:00", "SPY": 648.5407096985, "GLD": 334.06, "TLT": 88.5998422752},
        {"index": "2025-09-10 00:00:00", "SPY": 650.4155371465, "GLD": 335.26, "TLT": 89.1062405668},
        {"index": "2025-09-11 00:00:00", "SPY": 655.820624789, "GLD": 334.76, "TLT": 89.7020032629},
        {"index": "2025-09-12 00:00:00", "SPY": 655.6012300876, "GLD": 335.42, "TLT": 89.3147575104},
        {"index": "2025-09-15 00:00:00", "SPY": 659.0916003365, "GLD": 338.91, "TLT": 89.5232744541},
        {"index": "2025-09-16 00:00:00", "SPY": 658.1841040718, "GLD": 339.59, "TLT": 89.7119326411},
        {"index": "2025-09-17 00:00:00", "SPY": 657.3663601849, "GLD": 336.97, "TLT": 89.483556941},
        {"index": "2025-09-18 00:00:00", "SPY": 660.4378860039, "GLD": 335.62, "TLT": 88.5601247622},
        {"index": "2025-09-19 00:00:00", "SPY": 663.7, "GLD": 339.18, "TLT": 88.3913253316},
        {"index": "2025-09-22 00:00:00", "SPY": 666.84, "GLD": 345.05, "TLT": 88.0735852271},
        {"index": "2025-09-23 00:00:00", "SPY": 663.21, "GLD": 346.46, "TLT": 88.6892066796},
        {"index": "2025-09-24 00:00:00", "SPY": 661.1, "GLD": 343.32, "TLT": 88.3516078186},
        {"index": "2025-09-25 00:00:00", "SPY": 658.05, "GLD": 344.75, "TLT": 88.3516078186},
        {"index": "2025-09-26 00:00:00", "SPY": 661.82, "GLD": 346.74, "TLT": 88.2721727924},
        {"index": "2025-09-29 00:00:00", "SPY": 663.68, "GLD": 352.46, "TLT": 88.9970174059},
        {"index": "2025-09-30 00:00:00", "SPY": 666.18, "GLD": 355.47, "TLT": 88.738853571},
    ]
    df = pd.DataFrame(rows)
    df["index"] = pd.to_datetime(df["index"])
    df = df.set_index("index")
    # 过滤掉所有价格列都是空值的行（如 09-01）
    price_columns = ["SPY", "GLD", "TLT"]
    df = df.dropna(subset=price_columns, how="all")
    return df


def parse_weights_from_string(weights_str: dict) -> dict:
    """将权重字符串（如 '20.0000%'）转换为小数（如 0.2）"""
    weights = {}
    for asset, weight_str in weights_str.items():
        # 移除 '%' 并转换为 float，然后除以 100
        weight_float = float(weight_str.replace("%", "")) / 100.0
        weights[asset] = weight_float
    return weights


def main() -> None:
    """测试 backtester.run() 函数"""
    print("=== Backtest Test ===")
    
    # 1. 构建价格数据
    print("\n1. Building prices DataFrame...")
    prices_backtest = build_prices_from_table()
    print(f"   Prices shape: {prices_backtest.shape}")
    print(f"   Date range: {prices_backtest.index[0]} to {prices_backtest.index[-1]}")
    print(f"   Columns: {list(prices_backtest.columns)}")
    
    # 2. 解析权重数据
    print("\n2. Parsing weights...")
    weights_str = {"SPY": "20.0000%", "GLD": "80.0000%", "TLT": "0.0000%"}
    bl_result_weights = parse_weights_from_string(weights_str)
    print(f"   Weights: {bl_result_weights}")
    
    # 3. 运行回测
    print("\n3. Running backtest...")
    backtester = Backtester()
    report: BacktestReport = backtester.run(prices_backtest, bl_result_weights)
    
    # 4. 打印结果
    print("\n4. Backtest Results:")
    print(f"   CAGR: {report.cagr:.4%}")
    print(f"   Max Drawdown: {report.max_drawdown:.4%}")
    print(f"   Sharpe: {report.sharpe:.4f}")
    print(f"   Volatility: {report.volatility:.4%}")
    print(f"   Weights used: {report.weights_used}")
    print(f"   Equity curve length: {len(report.equity_curve)}")
    print(f"   Equity curve start: {report.equity_curve.iloc[0]:.6f}")
    print(f"   Equity curve end: {report.equity_curve.iloc[-1]:.6f}")
    
    print("\n=== Test Complete ===")


if __name__ == "__main__":
    main()

