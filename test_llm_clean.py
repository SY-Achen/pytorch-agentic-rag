"""LLM-assisted cleaning + feedback-driven chunking with fallback."""
import server


class _FakeClean:
    def __init__(self, payload): self.payload = payload

    async def __call__(self, messages, temperature=0.7):
        return self.payload


def test_clean_uses_blocks_from_llm(monkeypatch):
    payload = '{"cleaned": "标题A\\n内容A\\n标题B\\n内容B", "blocks": [{"heading": "A", "content": "内容A"}, {"heading": "B", "content": "内容B"}]}'
    monkeypatch.setattr(server, "async_deepseek", _FakeClean(payload))
    chunks = server._smart_chunk("原始很乱很长的文本。原始很乱很长的文本。原始很乱很长的文本。", 600, 120)
    joined = "\n".join(chunks)
    assert "内容A" in joined and "内容B" in joined
    assert len(chunks) >= 2  # blocks split into separate chunks


def test_clean_falls_back_on_llm_error(monkeypatch):
    class _Boom:
        async def __call__(self, messages, temperature=0.7):
            raise ConnectionError("clean model unavailable")

    monkeypatch.setattr(server, "async_deepseek", _Boom())
    text = "第一段。" * 30 + "\n\n" + "第二段。" * 30
    chunks = server._smart_chunk(text, 200, 50)
    # fallback splitter still returns non-empty chunks covering the text
    assert chunks
    assert sum(len(c) for c in chunks) > 100


def test_clean_invalid_json_falls_back(monkeypatch):
    monkeypatch.setattr(server, "async_deepseek", _FakeClean("not json at all"))
    chunks = server._smart_chunk("普通文本。" * 40, 200, 50)
    assert chunks