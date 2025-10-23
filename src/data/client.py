from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

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
    def fetch_recent_market_and_macro(self) -> MarketData:
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

    def _fetch_prices(self, tickers: List[str], start: str, end: str) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        for symbol in tickers:
            # ETF 价格（SPY/GLD/TLT 都是 ETF），优先用 yfinance 提供商
            data_obj = obb.etf.price.historical(
                symbol=symbol,
                start_date=start,
                end_date=end,
                provider="yfinance",
                interval="1d",
            )
            df = self._to_dataframe(data_obj)
            # 统一列名，取收盘价
            price_col = self._infer_price_column(df)
            series = df[price_col].rename(symbol)
            frames.append(series)

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

        # VIX 用指数价格
        vix_obj = obb.index.price.historical(
            symbol="^VIX",
            start_date=start,
            end_date=end,
            provider="yfinance",
            interval="1d",
        )
        vix_df = self._to_dataframe(vix_obj)
        vix_col = self._infer_price_column(vix_df)
        frames.append(vix_df[vix_col].rename("VIX"))

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
