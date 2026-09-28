"""PII (Personally Identifiable Information) guardrail.

Intercepts user input before it reaches the LLM. Detects Chinese mobile
numbers, mainland ID card numbers, and email addresses via regex.
"""
import re

# Chinese mobile: 1[3-9]X{10}
RE_MOBILE = re.compile(r"1[3-9]\d{9}")
# Mainland ID card: 18 digits (last char can be X)
RE_ID_CARD = re.compile(r"\b[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]\b")
# Email address
RE_EMAIL = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")

SENSITIVE_PATTERNS = [
    ("手机号", RE_MOBILE),
    ("身份证号", RE_ID_CARD),
    ("邮箱", RE_EMAIL),
]


def detect_pii(text: str) -> list[dict]:
    """Return list of detected PII items."""
    findings = []
    for label, pattern in SENSITIVE_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            findings.append({"type": label, "matches": matches})
    return findings


def sanitize(text: str) -> tuple[str, bool]:
    """Redact PII from text. Returns (cleaned_text, was_redacted)."""
    found = detect_pii(text)
    if not found:
        return text, False

    redacted = text
    for item in found:
        placeholder = f"[已脱敏:{item['type']}]"
        for match in item["matches"]:
            redacted = redacted.replace(match, placeholder)

    return redacted, True
