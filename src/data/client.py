from typing import List


class DataClient:
    """最小数据客户端。"""

    def __init__(self, tickers: List[str] | None = None) -> None:
        self.tickers = tickers or ["SPY", "GLD", "TLT"]

    def fetch(self) -> dict:
        return {"tickers": self.tickers, "status": "ok"}
