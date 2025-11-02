from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import os
import time
import pandas as pd
from openbb import obb  # type: ignore


@dataclass
class MarketData:
    prices_1y: pd.DataFrame  # 列为资产代码，行为日期，价格为收盘价
    macro_1y: pd.DataFrame  # 列为宏观指标，行为日期


class DataClient:
    """数据客户端（OpenBB）。

    - 1 年窗口：供 LLM 生成观点
    - 2015-2025：供回测
    """

    def __init__(self, tickers: Optional[List[str]] | None = None) -> None:
        self.tickers: List[str] = tickers or ["SPY", "GLD", "TLT"]

    # -------- 公共 API --------
    def fetch_recent_market_and_macro(self, start: str | None = None, end: str | None = None) -> MarketData:
        if not start or not end:
            start, end = self._one_year_window()
        prices = self._fetch_prices(self.tickers, start, end)
        macro = self._fetch_macro(start, end)
        return MarketData(prices_1y=prices, macro_1y=macro)

    def fetch_backtest_prices(self, start: str = "2015-01-01", end: str = "2025-12-31") -> pd.DataFrame:
        return self._fetch_prices(self.tickers, start, end)

    # -------- 具体实现 --------
    def _one_year_window(self) -> Tuple[str, str]:
        end_date = date.today()
        start_date = end_date - timedelta(days=365)
        return (start_date.isoformat(), end_date.isoformat())

    def _provider_try_order(self) -> List[str]:
        order: List[str] = []
        # 优先使用带密钥的可靠源，再退回免费源
        tiingo_token = os.getenv("OPENBB_TIINGO_TOKEN")
        if tiingo_token:
            order.append("tiingo")
        fmp_key = os.getenv("OPENBB_FMP_API_KEY")
        if fmp_key:
            order.append("fmp")
        polygon_key = os.getenv("OPENBB_POLYGON_API_KEY")
        if polygon_key:
            order.append("polygon")
        # 默认使用 yfinance
        order.append("yfinance")
        return order

    def _fetch_prices(self, tickers: List[str], start: str, end: str) -> pd.DataFrame:
        providers = self._provider_try_order()
        frames: List[pd.Series] = []
        for symbol in tickers:
            last_error_msgs: List[str] = []
            success_series: Optional[pd.Series] = None
            for provider in providers:
                # 指数退避重试：1s, 2s, 4s
                for attempt in range(3):
                    try:
                        data_obj = obb.equity.price.historical(
                            symbol=symbol,
                            start_date=start,
                            end_date=end,
                            provider=provider,
                            interval="1d",
                        )
                        df = self._to_dataframe(data_obj)
                        price_col = self._infer_price_column(df)
                        success_series = df[price_col].rename(symbol)
                        break
                    except Exception as e:
                        print(f"Error fetching {symbol} prices: {e}")
                        time.sleep(1.0 * (2 ** attempt))
                if success_series is not None:
                    break
            if success_series is None:
                detail = "; ".join(last_error_msgs)
                raise RuntimeError(
                    f"Failed to fetch {symbol} prices [{start} -> {end}] via providers {providers}. Details: {detail}. "
                    f"Tips: expand date range (avoid weekends/holidays) or set OPENBB_TIINGO_TOKEN in .env to enable 'tiingo'."
                )
            frames.append(success_series)

        prices = pd.concat(frames, axis=1).sort_index()
        prices = prices[~prices.index.duplicated(keep="first")]
        prices = prices.asfreq("B").ffill()  # 对齐到工作日，向前填充
        return prices

    def _fetch_macro(self, start: str, end: str) -> pd.DataFrame:
        # 选用 FRED 指标代码
        fred_series = {
            "CPI": "CPIAUCSL",  # CPI（美国城市居民消费价格指数，季调）
            "FEDFUNDS": "FEDFUNDS",  # 联邦基金利率（有效联邦基金利率）
            "DXY": "DTWEXBGS",  # 广义美元指数（BROAD, Goods-Services）
        }

        frames: List[pd.Series] = []
        for name, series_id in fred_series.items():
            data_obj = obb.economy.fred.series(
                series_id=series_id,
                start_date=start,
                end_date=end,
            )
            df = self._to_dataframe(data_obj)
            value_col = self._infer_value_column(df)
            frames.append(df[value_col].rename(name))

        # VIX 用指数价格，带提供商重试
        providers = self._provider_try_order()
        vix_series: Optional[pd.Series] = None
        last_errs: List[str] = []
        for provider in providers:
            for attempt in range(3):
                try:
                    vix_obj = obb.index.price.historical(
                        symbol="^VIX",
                        start_date=start,
                        end_date=end,
                        provider=provider,
                        interval="1d",
                    )
                    vix_df = self._to_dataframe(vix_obj)
                    vix_col = self._infer_price_column(vix_df)
                    vix_series = vix_df[vix_col].rename("VIX")
                    break
                except Exception as e:
                    last_errs.append(f"VIX:{provider}#{attempt+1}:{type(e).__name__}:{str(e)[:120]}")
                    time.sleep(1.0 * (2 ** attempt))
            if vix_series is not None:
                break
        if vix_series is None:
            detail = "; ".join(last_errs)
            raise RuntimeError(
                f"Failed to fetch VIX [{start} -> {end}] via providers {providers}. Details: {detail}."
            )
        frames.append(vix_series)

        macro = pd.concat(frames, axis=1).sort_index()
        macro = macro.asfreq("B").ffill()
        return macro

    # -------- 工具方法 --------
    @staticmethod
    def _to_dataframe(data_obj: object) -> pd.DataFrame:
        # 兼容不同返回体
        for attr in ("to_df", "to_dataframe", "to_pandas", "to_df_dict"):
            if hasattr(data_obj, attr):  # pragma: no branch
                candidate = getattr(data_obj, attr)
                try:
                    df = candidate()
                    if isinstance(df, pd.DataFrame):
                        return df
                    if isinstance(df, dict):
                        return pd.DataFrame(df)
                except Exception as e:
                    raise
        # 已经是 DataFrame
        if isinstance(data_obj, pd.DataFrame):
            return data_obj
        raise TypeError("无法将 OpenBB 返回结果转换为 DataFrame")

    @staticmethod
    def _infer_price_column(df: pd.DataFrame) -> str:
        for col in ["adj_close", "close", "Adj Close", "Close", "price", "Price"]:
            if col in df.columns:
                return col
        # 猜测数值列
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not numeric_cols:
            raise ValueError("找不到价格列")
        return numeric_cols[0]

    @staticmethod
    def _infer_value_column(df: pd.DataFrame) -> str:
        # FRED 往往只有一个值列
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not numeric_cols:
            raise ValueError("找不到数值列")
        return numeric_cols[0]

if __name__ == "__main__":
    # 加载 .env，启用 TIINGO/FMP 等密钥
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
    except Exception:
        raise Exception("Failed to load .env")

    client = DataClient()

    # 测试1：获取最近1年数据
    # print("=" * 50)
    # print("Test 1: Recent 1-year market data and macro")
    # print("=" * 50)
    # market = client.fetch_recent_market_and_macro()
    # print(f"\nPrices shape: {market.prices_1y.shape}")
    # print(f"Price columns: {list(market.prices_1y.columns)}")
    # print(f"First 5 price rows:\n{market.prices_1y.head()}")
    # print(f"\nMacro shape: {market.macro_1y.shape}")
    # print(f"Macro columns: {list(market.macro_1y.columns)}")
    # print(f"First 5 macro rows:\n{market.macro_1y.head()}")

    # 测试2：获取指定时间段数据（尽量避免仅1-2天，避开周末/假日）
    # print("\n" + "=" * 50)
    # print("Test 2: Custom date range (2023-12-05 to 2023-12-10)")
    # print("=" * 50)
    # market_custom = client.fetch_recent_market_and_macro("2023-12-05", "2023-12-10")
    # print(f"\nPrices shape: {market_custom.prices_1y.shape}")
    # print(f"First 5 rows:\n{market_custom.prices_1y.head()}")

    # 测试3：获取回测数据
    # print("\n" + "=" * 50)
    # print("Test 3: Backtest prices (2015-01-01 to 2025-12-31)")
    # print("=" * 50)
    # prices_bt = client.fetch_backtest_prices("2015-01-01", "2025-12-31")
    # print(f"\nPrices shape: {prices_bt.shape}")
    # print(f"Columns: {list(prices_bt.columns)}")
    # print(f"First 5 rows:\n{prices_bt.head()}")
    # print(f"Last 5 rows:\n{prices_bt.tail()}")

    from openbb import obb

    # Get historical price data for Apple (AAPL) using Tiingo
    aapl_prices = obb.equity.price.historical(
        symbol="AAPL",
        start_date="2024-01-01",
        end_date="2024-01-31",
        provider="tiingo"
    )

    # Print the resulting DataFrame
    aapl_info = aapl_prices.to_df()
    print(aapl_info)
