from typing import Dict, Any
from src.llm import LLMClient


class Optimizer:
    """最小优化器，依赖 LLMClient。"""

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    def optimize(self) -> Dict[str, Any]:
        # 使用 llm_client 的观点
        views = self.llm_client.generate_views()
        return {
            "weights": {"SPY": 0.6, "GLD": 0.1, "TLT": 0.3},
            "views_used": views,
            "status": "ok",
        }
