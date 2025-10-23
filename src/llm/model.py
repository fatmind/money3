from typing import Dict, Any, List
from src.data import DataClient


class LLMClient:
    """最小 LLM 客户端，依赖 DataClient。"""

    def __init__(self, data_client: DataClient) -> None:
        self.data_client = data_client

    def generate_views(self) -> Dict[str, Any]:
        # 使用 data_client 的数据
        tickers = self.data_client.tickers
        return {
            "horizon": "1m",
            "views": [
                {"type": "absolute", "asset": tickers[0], "expected_return": 0.10, "confidence": 0.6}
            ],
            "status": "ok",
        }
