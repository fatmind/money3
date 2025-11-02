import os
import time
from typing import Dict, List

import pandas as pd
import requests
import yfinance as yf
from urllib.request import urlopen
import json
import certifi
import finnhub

def fetch_prices_from_tiingo(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    从 Tiingo API 获取多个股票的历史日终价格。

    :param tickers: 股票代码列表, e.g., ["SPY", "GLD"]
    :param start_date: 开始日期 (YYYY-MM-DD)
    :param end_date: 结束日期 (YYYY-MM-DD)
    :return: 一个包含所有股票复权收盘价的 DataFrame
    """
    api_key = os.getenv("TIINGO_TOKEN")
    if not api_key:
        raise ValueError("Missing TIINGO_TOKEN in environment/.env")

    session = requests.Session()
    session.headers.update({
        "Authorization": f"Token {api_key}",
        "Content-Type": "application/json",
    })

    frames: Dict[str, pd.Series] = {}
    base_index = pd.date_range(start=start_date, end=end_date, freq='B')

    for symbol in tickers:
        series = pd.Series(dtype=float, index=base_index, name=symbol)
        try:
            url = f"https://api.tiingo.com/tiingo/daily/{symbol}/prices?startDate={start_date}&endDate={end_date}&format=json"
            r = session.get(url, timeout=15)
            r.raise_for_status()
            data = r.json()
            if data:
                df = pd.DataFrame(data)
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
                series = df['adjClose'].tz_localize(None)
        except Exception as e:
            print(f"Warning: An unexpected error occurred for {symbol}: {e}")
            
        frames[symbol] = series

    prices_df = pd.DataFrame(frames).sort_index()
    return prices_df.reindex(base_index).ffill()


def fetch_news_from_finnhub(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    从 Finnhub API 获取多个股票的公司新闻。

    :param tickers: 股票代码列表, e.g., ["SPY", "GLD"]
    :param start_date: 开始日期 (YYYY-MM-DD)
    :param end_date: 结束日期 (YYYY-MM-DD)
    :return: 包含新闻数据的 DataFrame
    """
    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        raise ValueError("Missing FINNHUB_API_KEY in environment/.env")

    finnhub_client = finnhub.Client(api_key=api_key)
    
    all_news = []
    for symbol in tickers:
        try:
            news_data = finnhub_client.company_news(symbol, _from=start_date, to=end_date)
            
            if news_data:
                for item in news_data:
                    item['symbol'] = symbol
                    all_news.append(item)
            else:
                print(f"----- No news found for {symbol} from Finnhub -----")
        except Exception as e:
            raise ValueError(f"Failed to fetch news for {symbol} from Finnhub: {e}")
        
    df = pd.DataFrame(all_news)
    df['datetime'] = pd.to_datetime(df['datetime'], unit='s', utc=True)
    return df.sort_values('datetime', ascending=False)


def fetch_macro_from_fred(series_map: Dict[str, str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    从 FRED API 获取多个宏观经济数据序列。

    :param series_map: 一个字典，键是 FRED Series ID, 值是期望的列名, e.g., {"CPIAUCSL": "CPI"}
    :param start_date: 开始日期 (YYYY-MM-DD)
    :param end_date: 结束日期 (YYYY-MM-DD)
    :return: 包含所有宏观数据的 DataFrame
    """
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise ValueError("Missing FRED_API_KEY in environment/.env")

    session = requests.Session()
    all_series = {}

    for series_id, col_name in series_map.items():
        try:
            url = (
                f"https://api.stlouisfed.org/fred/series/observations"
                f"?series_id={series_id}&api_key={api_key}&file_type=json"
                f"&observation_start={start_date}&observation_end={end_date}"
            )
            r = session.get(url, timeout=15)
            r.raise_for_status()
            data = r.json()

            if 'observations' in data and data['observations']:
                df = pd.DataFrame(data['observations'])
                df['date'] = pd.to_datetime(df['date'])
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
                df = df.set_index('date')
                all_series[col_name] = df['value']
        except Exception as e:
            print(f"----- Warning: Failed to fetch FRED series {series_id}: {e} -----")

    if not all_series:
        return pd.DataFrame()

    macro_df = pd.DataFrame(all_series).sort_index()
    return macro_df.asfreq('B').ffill()


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    
    test_tickers = ["SPY", "GLD", "TLT"]
    test_start = "2025-09-20"
    test_end = "2025-10-10"

    # --- 测试用例：价格 ---
    # print(f"Fetching prices for {test_tickers} from {test_start} to {test_end}")
    # prices = fetch_prices_from_tiingo(test_tickers, test_start, test_end)
    # print("\n--- Price DataFrame ---")
    # print(prices.to_string())

    # --- 测试用例：新闻 ---
    # print(f"\nFetching news for {test_tickers} from Finnhub")
    # news = fetch_news_from_finnhub(test_tickers, test_start, test_end)
    # print("\n--- News DataFrame ---")
    # if not news.empty:
    #     print(news[['datetime', 'symbol', 'headline']].head().to_string())
    # else:
    #     print("No news found.")
    
    # --- 测试用例：宏观数据 ---
    macro_series_to_fetch = {
        "CPIAUCSL": "CPI",
        "FEDFUNDS": "InterestRate",
        "DTWEXBGS": "DXY",
        "VIXCLS": "VIX"
    }
    print(f"\nFetching macro data from FRED for {list(macro_series_to_fetch.keys())}")
    macro_data = fetch_macro_from_fred(macro_series_to_fetch, test_start, test_end)
    print("\n--- Macro DataFrame ---")
    print(macro_data.to_string())
