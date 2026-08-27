"""CLI for the PyTorch Agentic RAG chatbot.

Usage:
    python cli.py init --key YOUR_DEEPSEEK_KEY        # once: save key
    python cli.py "How do I create a tensor on GPU?"  # ask a question
"""
import os
import sys
from pathlib import Path

KEY_FILE = Path(".deepseek_key")


def get_key() -> str:
    k = os.environ.get("DEEPSEEK_API_KEY")
    if k:
        return k
    if KEY_FILE.exists():
        return KEY_FILE.read_text().strip()
    sys.exit("No API key. Run: python cli.py init --key sk-...")


def main():
    args = sys.argv[1:]
    if args and args[0] == "init":
        key = args[args.index("--key") + 1] if "--key" in args else input("DeepSeek key: ")
        KEY_FILE.write_text(key.strip())
        print("✓ key saved (add .deepseek_key to .gitignore)")
        return
    import agent
    agent.set_llm(get_key())
    q = " ".join(args)
    if not q:
        sys.exit("Usage: python cli.py \"your question\"")
    print("\n--- answer ---")
    print(agent.ask(q))


if __name__ == "__main__":
    main()