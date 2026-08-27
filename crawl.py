#!/usr/bin/env python
"""Scrape PyTorch docs (stable) pages to raw markdown-ish text files."""
import argparse, re, requests
from pathlib import Path
from bs4 import BeautifulSoup

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0)"}
BASE = "https://pytorch.org/docs/stable/"

# A hand-picked map of useful PyTorch doc pages (name -> file). Expand as needed.
PAGES = {
    "tensors": "tensors.html",
    "autograd": "autograd.html",
    "nn": "nn.html",
    "nn.modules.conv": "generated/torch.nn.Conv2d.html",
    "optim": "optim.html",
    "data": "data.html",
    "torch.nn.functional": "nn.functional.html",
    "tensor_attributes": "tensor_attributes.html",
    "torch.compile": "compile.html",
    "distributions": "distributions.html",
    "sparse": "sparse.html",
    "fft": "fft.html",
    "linalg": "linalg.html",
    "cpp_extension": "cpp_extension.html",
    "cuda": "cuda.html",
}


def clean(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    main = soup.find("div", {"class": "container"}) or soup
    for tag in main(["script", "style", "nav", "header", "footer", "noscript"]):
        tag.decompose()
    text = main.get_text("\n")
    # collapse blanks, drop nav-chatter lines
    lines = [l.strip() for l in text.splitlines()]
    lines = [l for l in lines if l and not l.startswith(("Previous", "Next", "On this page"))]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data", help="output dir")
    ap.add_argument("--limit", type=int, default=0, help="max pages to fetch (0=all)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    n = 0
    for name, file in PAGES.items():
        if a.limit and n >= a.limit:
            break
        url = BASE + file
        try:
            html = requests.get(url, headers=UA, timeout=30).text
            text = clean(html)
            if len(text) < 200:
                print(f"  skip {name} (too small)"); continue
            (out / f"{name}.txt").write_text(text, encoding="utf-8")
            print(f"  ok  {name}  {len(text)} chars")
            n += 1
        except Exception as e:
            print(f"FAIL {name}: {type(e).__name__} {e}")


if __name__ == "__main__":
    main()