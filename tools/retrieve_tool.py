"""Retrieve tool with tenacity retry — wraps retrieve_logic._do_retrieve."""
import os
import json
from typing import Optional
from langchain_core.tools import tool
from tenacity import retry, stop_after_attempt, wait_exponential, \
    retry_if_exception_type
from tools.retrieve_logic import _do_retrieve

_MAX_RETRIES = int(os.environ.get("RETRIEVE_MAX_RETRIES", "3"))


@tool
@retry(
    stop=stop_after_attempt(_MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    before_sleep=lambda retry_state: print(
        f"[RETRY] retrieve failed, attempt {retry_state.attempt_number}/{_MAX_RETRIES}"),
)
def retrieve(query: str, filter_json: Optional[str] = None) -> str:
    """Search the knowledge base and return relevant document passages for a given question.

    Args:
        query: The user's question.
        filter_json: Optional JSON string of ChromaDB metadata filter for RBAC / dept / time_decay.
            Example: '{"$and":[{"visibility":{"$gte":"internal"}},{"dept":"hr"}]}'
            If None, searches the entire collection.
    """
    metadata_filter = json.loads(filter_json) if filter_json else None
    return _do_retrieve(query, metadata_filter=metadata_filter)
