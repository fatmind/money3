from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import yfinance as yf


@dataclass
class MarketData:
    prices: pd.DataFrame
    macro: pd.DataFrame
    news: pd.DataFrame


class TingoData:
    """数据客户端（Tiingo for prices/news, FRED for macro, yfinance for VIX）。"""

    def __init__(self, tickers: Optional[List[str]] = None) -> None:
        self.tickers: List[str] = tickers or ["SPY", "GLD", "TLT"]
        self.tiingo_api_key = os.getenv("TIINGO_TOKEN")
        self.fred_api_key = os.getenv("FRED_API_KEY")
        
        if not self.tiingo_api_key:
            raise ValueError("Missing TIINGO_TOKEN in environment/.env")
        if not self.fred_api_key:
            raise ValueError("Missing FRED_API_KEY in environment/.env")
            
        self.tiingo_session = self._init_tiingo_session()

    def _init_tiingo_session(self) -> requests.Session:
        session = requests.Session()
        session.headers.update({
            "Authorization": f"Token {self.tiingo_api_key}",
            "Content-Type": "application/json",
        })
        return session

    def fetch_market_data(self, start: str | None = None, end: str | None = None) -> MarketData:
        if not start or not end:
            start, end = self._one_year_window()
        
        prices = self._fetch_prices(self.tickers, start, end)
        vix = self._fetch_vix(start, end)
        
        macro_symbols = {"CPIAUCSL": "CPI", "FEDFUNDS": "FEDFUNDS", "DTWEXBGS": "DXY"}
        macro_series = self._fetch_fred_data(list(macro_symbols.keys()), start, end)
        
        macro_df = pd.DataFrame(macro_series).rename(columns=macro_symbols)
        macro_df["VIX"] = vix
        macro = macro_df.asfreq("B").ffill()

        news = self._fetch_news(self.tickers, start, end)
        
        return MarketData(prices=prices, macro=macro, news=news)

    def fetch_backtest_prices(self, start: str = "2015-01-01", end: str = "2025-12-31") -> pd.DataFrame:
        return self._fetch_prices(self.tickers, start, end)

    def _one_year_window(self) -> Tuple[str, str]:
        end_date = date.today()
        start_date = end_date - timedelta(days=365)
        return start_date.isoformat(), end_date.isoformat()

    def _fetch_prices(self, tickers: List[str], start: str, end: str) -> pd.DataFrame:
        frames = {}
        base_index = pd.date_range(start=start, end=end, freq='B')

        for symbol in tickers:
            series = pd.Series(dtype=float, index=base_index, name=symbol)
            for attempt in range(3):
                try:
                    url = f"https://api.tiingo.com/tiingo/daily/{symbol}/prices?startDate={start}&endDate={end}&format=json"
                    r = self.tiingo_session.get(url, timeout=15)
                    r.raise_for_status()
                    data = r.json()
                    if data:
                        df = pd.DataFrame(data)
                        df['date'] = pd.to_datetime(df['date'])
                        df = df.set_index('date')
                        series = df['adjClose'].tz_localize(None)
                    break 
                except Exception as e:
                    if attempt == 2:
                        print(f"Warning: Failed to fetch {symbol} after 3 attempts. Will be empty.")
                    time.sleep(1 * (2 ** attempt))
            frames[symbol] = series

        prices = pd.DataFrame(frames).sort_index()
        return prices.reindex(base_index).ffill()
        
    def _fetch_vix(self, start: str, end: str) -> pd.Series:
        try:
            vix_data = yf.download("^VIX", start=start, end=end, progress=False)
            if not vix_data.empty:
                return vix_data['Adj Close']
        except Exception as e:
            print(f"Warning: Failed to fetch VIX using yfinance: {e}")
        return pd.Series(dtype=float)

    def _fetch_fred_data(self, series_ids: List[str], start: str, end: str) -> Dict[str, pd.Series]:
        series_data = {}
        for series_id in series_ids:
            try:
                url = (
                    f"https://api.stlouisfed.org/fred/series/observations"
                    f"?series_id={series_id}&api_key={self.fred_api_key}&file_type=json"
                    f"&observation_start={start}&observation_end={end}"
                )
                r = requests.get(url, timeout=15)
                r.raise_for_status()
                data = r.json()

                if 'observations' in data and data['observations']:
                    df = pd.DataFrame(data['observations'])
                    df['date'] = pd.to_datetime(df['date'])
                    # FRED data can have '.' for missing values, coerce to numeric
                    df['value'] = pd.to_numeric(df['value'], errors='coerce')
                    df = df.set_index('date')
                    series_data[series_id] = df['value']
            except Exception as e:
                print(f"Warning: Failed to fetch FRED series {series_id}: {e}")
        return series_data
    
    def _fetch_news(self, tickers: List[str], start: str, end: str) -> pd.DataFrame:
        try:
            url = f"https://api.tiingo.com/tiingo/news?tickers={','.join(tickers)}&startDate={start}&endDate={end}&limit=100"
            r = self.tiingo_session.get(url, timeout=20)
            r.raise_for_status()
            news_data = r.json()
            if not news_data:
                return pd.DataFrame()
            
            df = pd.DataFrame(news_data)
            df['publishedDate'] = pd.to_datetime(df['publishedDate'])
            return df.sort_values('publishedDate', ascending=False)
        except Exception:
            return pd.DataFrame()


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    client = TingoData()
    
    print("=" * 50)
    print("Test: Fetching recent market data from Tiingo")
    print("=" * 50)
    market_data = client.fetch_market_data()

    print("\n--- Prices (head) ---")
    print(market_data.prices.head())

    print("\n--- Macro (head) ---")
    print(market_data.macro.head())

    print("\n--- News (head) ---")
    print(market_data.news.head())
    print(f"\nNews Shape: {market_data.news.shape}")
