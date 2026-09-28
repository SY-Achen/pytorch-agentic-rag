"""Image caption bridging: extract -> vision caption -> confidence filter -> text bridge."""
import pytest
import server


@pytest.fixture(autouse=True)
def _offline_chunking(monkeypatch):
    """Caption tests must not call the real document-cleaning LLM."""
    monkeypatch.setattr(server, "_smart_chunk", server._simple_split)


class _FakeVision:
    def __init__(self, responses): self.responses = list(responses); self.calls = 0

    def __call__(self, image_bytes):
        self.calls += 1
        return self.responses.pop(0)


def test_high_confidence_caption_bridged(monkeypatch):
    monkeypatch.setattr(server, "VISION_API_KEY", "test-key")
    fake = _FakeVision([("内存层次结构图：寄存器/L1/L2/主存", 0.9)])
    monkeypatch.setattr(server, "_caption_image", fake)
    text = "计算机组成原理笔记。\n"
    chunks = server._capsule_images(text, [b"\x89PNGfake1"], "notes.docx", 600, 120)
    assert "内存层次结构图" in "\n".join(chunks)
    assert fake.calls == 1


def test_low_confidence_image_dropped(monkeypatch):
    monkeypatch.setattr(server, "VISION_API_KEY", "test-key")
    fake = _FakeVision([("模糊不清的图片", 0.12)])
    monkeypatch.setattr(server, "_caption_image", fake)
    chunks = server._capsule_images("普通文本。", [b"\x89PNGfake1"], "n.docx", 600, 120)
    assert "模糊不清" not in "\n".join(chunks)


def test_no_images_returns_original_chunks():
    chunks = server._capsule_images("第一段。" * 20, [], "n.docx", 150, 30)
    assert chunks


def test_vision_crash_degrades_to_text_only(monkeypatch):
    def boom(image_bytes):
        raise ConnectionError("vision api down")

    monkeypatch.setattr(server, "_caption_image", boom)
    chunks = server._capsule_images("没有图片也能入库。" * 10, [b"x1", b"x2"], "n.docx", 150, 30)
    assert chunks