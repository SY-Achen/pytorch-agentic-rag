"""Fetch PyTorch official documentation (.md sources) from GitHub raw."""
import json
import urllib.request
from pathlib import Path

UA = {"User-Agent": "Mozilla/5.0"}
BASE = "https://raw.githubusercontent.com/pytorch/pytorch/main/docs/source/"

# Hand-picked core docs — enough context for a real PyTorch QA chatbot.
PAGES = [
    "nn", "tensors", "optim", "data", "autograd", "amp",
    "torch", "nn.init", "nn.modules.module", "torch.nn.functional",
]


def fetch(name: str) -> str:
    req = urllib.request.Request(BASE + name + ".md", headers=UA)
    return urllib.request.urlopen(req, timeout=30).read().decode("utf-8")


def main(out: str = "data"):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    meta = []
    for name in PAGES:
        try:
            text = fetch(name)
            if len(text) < 500:
                print(f"  skip {name} ({len(text)}c)"); continue
            (out / f"{name}.md").write_text(text, encoding="utf-8")
            meta.append({"source": BASE + name + ".md", "title": name, "chars": len(text)})
            print(f"  ok  {name}  {len(text)}c")
        except Exception as e:
            print(f"FAIL {name}: {e}")
    (out / "index.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TOTAL {len(meta)} docs")


if __name__ == "__main__":
    main()