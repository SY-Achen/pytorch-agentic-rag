"""
Enterprise Tracing Middleware
Purpose: Intercept LangGraph execution steps and output structured JSON logs.
"""
import json
import time
import logging
from typing import Any
from langchain_core.callbacks import BaseCallbackHandler
from uuid import UUID

class EnterpriseTraceHandler(BaseCallbackHandler):
    """极简的企业级日志处理器"""
    
    def __init__(self):
        self.start_time = None
        # ponytail: 使用 Python 标准 logging 模块，避免引入第三方重型框架
        self.logger = logging.getLogger(__name__)
        
    def on_chain_start(self, serialized: dict, inputs: dict, *, run_id: UUID, **kwargs: Any) -> None:
        self.start_time = time.time()
        node_name = serialized.get('name', 'unknown_node')
        self.logger.info(f"⬇️ START | Node: {node_name}")

    def on_chain_end(self, outputs: dict, *, run_id: UUID, **kwargs: Any) -> None:
        elapsed = round(time.time() - self.start_time, 3)
        status = {"latency_s": elapsed, "status": "success"}
        self.logger.info(json.dumps(status, ensure_ascii=False))

    def on_tool_error(self, error: Exception, *, run_id: UUID, **kwargs: Any) -> None:
        # 重点记录工具层的崩溃原因，这是调试的关键
        self.logger.error(f"🚨 TOOL ERROR DETECTED: {error}", exc_info=True)
        
    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        self.logger.error(f"⛔ MODEL FAILURE: {error}")
