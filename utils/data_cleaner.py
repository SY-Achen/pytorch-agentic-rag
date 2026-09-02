"""
rag_agent/utils/data_cleaner.py — 企业文档清洗管线
策略：正则（快/便宜）→ 切块 → LLM（准/贵）分级处理
"""
import re
from pathlib import Path


# 🔧 第一层：零成本过滤
NOISE_PATTERNS = [
    r"^第\s*\d+\s*页",              # 中文页码
    r"^Page\s+\d+\b",               # 英文页码 (含 "Page 1 of 10")
    r"^\w+\s+\d+(?:\.\d+)*",         # 章节标题 (如 "Section 2.1")
    r"(Copyright|©)\s*.*$",         # 版权声明
    r"^\.{3,}$",                    # 分隔线 .../---
]

def strip_noise(text: str) -> str:
    """逐行丢弃匹配噪音模式的片段"""
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if any(re.match(p, stripped) for p in NOISE_PATTERNS):
            continue
        lines.append(line)
    return "\n".join(lines)

def normalize_whitespace(text: str) -> str:
    """修复 PDF 抓取后的典型错位：合并单行，保留段落"""
    text = re.sub(r'[ \t]*(\r?\n)[ \t]+', ' ', text)  # 断行合并
    text = re.sub(r'(\n\n){3,}', '\n\n\n', text)      # 最多留一个空段
    return text.strip()

# --- Pipeline 编排 ---

def clean(raw_text: str) -> str:
    """一级清洗：正则 → 去噪 → 标准化"""
    text = strip_noise(raw_text)
    return normalize_whitespace(text)

def prepare_for_llm(raw_text: str) -> list[str]:
    """二级准备：先粗清再分块，减少喂给模型的垃圾"""
    cleaned = clean(raw_text)
    
    # 按段落切片，每段控制在 800 字内
    chunks = cleaned.split("\n\n")
    result = []
    buffer = ""
    for chunk in chunks:
        if len(buffer) + len(chunk) > 800:
            if buffer:
                result.append(buffer.strip())
            buffer = chunk
        else:
            buffer += "\n\n" + chunk
    if buffer.strip():
        result.append(buffer.strip())
    return result

if __name__ == "__main__":
    sample = """
       1.3 启动流程
        
       检查电源指示灯是否亮起。若显示红色，请执行紧急停机。
       
       Copyright © 2026 ABC Corp. 机密文件
       .........
       第 5 页 / 共 42 页
       
       重启后等待 30 秒确认日志输出正常。
    """
    
    print("=== 一级清洗 ===")
    print(clean(sample))
    print("\n=== 二级分块 ===")
    for i, c in enumerate(prepare_for_llm(sample)):
        print(f"\n[Chunk {i}]\n{c}")
