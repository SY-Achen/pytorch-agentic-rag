"""rag_agent/utils/router.py — 启发式多模型路由引擎
核心逻辑：关键词匹配 → 模型映射 → Fallback 降级，零 LLM 开销。"""
import re
from typing import Optional


# ── 路由规则表 ────────────────────────────────────────
ROUTES = {
    "code_query": ["def ", "import ", "class ", "return", "if __name__"],
    "comparison": ["对比", "比较", "区别", "vs", "versus"],
    "debugging":   ["报错", "异常", "错误", "error", "traceback", "bug"],
    "explain":     ["为什么", "怎么工作", "原理", "how does", "what is"],
}

MODEL_MAP = {
    "code_query":      "qwen2.5-7b",       # 代码→小模型
    "comparison":      "glm-4-plus",       # 对比→大模型（需推理）
    "debugging":       "glm-4-plus",       # 调试→大模型（需强推理）
    "explain":         "qwen2.5-7b",       # 解释→小模型足够
    "default":         "qwen2.5-7b",       # 兜底
}


def route(question: str, available_models: list[str] | None = None) -> str:
    """启发式路由器：根据问题内容选择最优模型。"""
    q_lower = question.lower()

    # 1. 尝试关键词匹配
    for task_type, keywords in ROUTES.items():
        if any(re.search(r"\b" + kw.replace(" ", r"\s+") + r"\b", q_lower) or kw in q_lower for kw in keywords):
            return MODEL_MAP[task_type]

    # 2. Fallback：选最大可用模型
    if available_models:
        return available_models[-1]
    return MODEL_MAP["default"]


if __name__ == "__main__":
    tests = [
        ("def fetch_data():", "code"),
        ("GLM-4 和 Qwen2.5 有什么区别？", "compare"),
        ("Traceback (most recent call last): TypeError", "debug"),
        ("为什么卡尔曼滤波要用协方差矩阵？", "explain"),
        ("今天天气怎么样", "unknown"),
    ]

    print("=== Router Test ===")
    for q, expected in tests:
        model = route(q, ["qwen2.5-7b", "glm-4-plus"])
        print(f"[{expected}] '{q[:30]}...' → {model}")
