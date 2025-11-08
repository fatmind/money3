from typing import Dict, List
import pandas as pd

# from src.opt.black_litterman import BLResult, optimize_with_black_litterman
from src.opt.black_litterman import BLResult, optimize_with_black_litterman


# 兼容两种运行方式：
# 1) python -m src.opt.bl_test_main  （包内相对导入）
# 2) python /path/to/src/opt/bl_test_main.py （脚本直接运行）
# try:
#     if __package__:  # 包内运行
#         from .black_litterman import BLResult, optimize_with_black_litterman  # type: ignore
#     else:
#         raise ImportError
# except Exception:
#     PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
#     if PROJECT_ROOT not in sys.path:
#         sys.path.insert(0, PROJECT_ROOT)
#     from src.opt.black_litterman import BLResult, optimize_with_black_litterman  # type: ignore


def build_prices_df() -> pd.DataFrame:
    rows: List[Dict[str, object]] = [
        # 原样使用你提供的有效行（跳过 2025-09-01 的空值行）
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
    return df


def build_views() -> dict:
    return {
        "horizon": "1m",
        "views": [
            {"type": "absolute", "asset": "GLD", "expected_return": 0.18, "confidence": 0.8},
            {"type": "relative", "assets": ["SPY", "TLT"], "outperformance": 0.04, "confidence": 0.5},
        ],
        "rationale": "Sample rationale",
        "timestamp": "2025-09-30",
    }


def check_and_print(result: BLResult) -> None:
    print("Weights:", {k: f"{v:.4%}" for k, v in result.weights.items()})
    print("Posterior returns:", {k: f"{float(v):.4f}" for k, v in result.posterior_returns.items()})
    print("P shape:", result.P.shape, "Q length:", len(result.Q))

    total_weight = sum(result.weights.values())
    non_negative = all(v >= 0.0 for v in result.weights.values())
    spy_ok = (result.weights.get("SPY", 0.0) <= 0.7 + 1e-8)
    print(f"Checks -> sum≈1: {abs(total_weight - 1.0) < 1e-6}, non_negative: {non_negative}, SPY<=0.7: {spy_ok}")


def main() -> None:
    prices = build_prices_df()
    views = build_views()
    result = optimize_with_black_litterman(prices, views)
    check_and_print(result)


if __name__ == "__main__":
    main()


