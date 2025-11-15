from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from openai import OpenAI

from money3.llm.json_util import parse_json


class LLMClient:
    """LLM 客户端（阿里云 DeepSeek API）。

    接收纯文本输入；若缺少配置则返回可控 mock。
    """

    def __init__(self) -> None:
        pass

    def generate_views(self, input_text: str) -> Dict[str, Any]:
        api_base = os.getenv("DEEPSEEK_API_BASE") or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("DASHSCOPE_API_KEY")

        if not api_key:
            raise Exception("Missing DEEPSEEK_API_KEY (or DASHSCOPE_API_KEY)")

        try:
            resp = self._call_deepseek(api_base, api_key, input_text)
            return resp
        except Exception as e:
            raise

    # ----- DeepSeek 调用 -----
    def _call_deepseek(self, api_base: str, api_key: str, input_text: str) -> Dict[str, Any]:
        client = OpenAI(api_key=api_key, base_url=api_base)
        completion = client.chat.completions.create(
            model="deepseek-v3.2-exp",
            messages=[
                {"role": "system", "content": "你是投资顾问，请仅输出 JSON，与 Black-Litterman 兼容的 views 结构"},
                {"role": "user", "content": input_text},
            ],
            temperature=0.3,
            stream=True,
            stream_options={"include_usage": True},
            extra_body={"enable_thinking": True},
        )
        # 流式处理响应
        reasoning_content = ""
        answer_content = ""
        is_answering = False

        for chunk in completion:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            # 收集思考内容
            if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
                if not is_answering:
                    print(delta.reasoning_content, end="", flush=True)
                reasoning_content += delta.reasoning_content

            # 收集回复内容
            if hasattr(delta, "content") and delta.content:
                if not is_answering:
                    print("\n", end="", flush=True)
                    is_answering = True
                print(delta.content, end="", flush=True)
                answer_content += delta.content

        print("\n" + "~" * 40 + "\n")  # 波浪线换行隔开
        
        return self._extract_json(answer_content)

    def _extract_json(self, content: str) -> Dict[str, Any]:
        """从多种格式中提取 JSON。

        支持：
        1. ```json\n{...}\n``` 代码块
        2. 混合文本 + JSON
        3. 纯 JSON
        
        使用智能解析器，自动处理格式错误。
        """
        content = content.strip()

        # 格式1：markdown JSON 代码块 ```json...```
        if "```json" in content:
            start = content.find("```json") + len("```json")
            end = content.find("```", start)
            if end > start:
                json_str = content[start:end].strip()
                try:
                    return parse_json(json_str)
                except Exception:
                    pass

        # 格式2/3：混合文本 + JSON 或纯 JSON
        # 找第一个 { 和最后一个 }
        first_brace = content.find("{")
        last_brace = content.rfind("}")
        if first_brace >= 0 and last_brace > first_brace:
            json_str = content[first_brace : last_brace + 1]
            try:
                return parse_json(json_str)
            except Exception:
                pass

        # 都失败，直接尝试解析整个 content
        try:
            return parse_json(content)
        except Exception:
            raise Exception(f"Failed to parse JSON from content: {content[:100]}")


if __name__ == "__main__":
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
    client = LLMClient()
    test_text = (
        "目标：年化15%、最大回撤10%、月度调仓。请仅输出 JSON，与 Black-Litterman 兼容的 views 结构。"
    )
    result = client.generate_views(test_text)
    print(json.dumps(result, ensure_ascii=False, indent=2))
