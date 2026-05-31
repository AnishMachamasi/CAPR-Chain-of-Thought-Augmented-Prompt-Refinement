import re


def extract_answer(raw: str, question_type: str) -> str:
    text = raw.strip()

    # 1. Explicit tag
    tag = re.search(r'FINAL ANSWER\s*:\s*(.+)', text, re.IGNORECASE)
    if tag:
        return tag.group(1).strip().split()[0]

    # 2a. Multiple-choice  (A / B / C / D)
    if question_type == "knowledge":
        m = re.search(r'\b([A-D])\b', text)
        if m:
            return m.group(1)

    # 2b. Yes / No
    if question_type == "reasoning":
        if re.search(r'\byes\b', text, re.IGNORECASE):
            return "Yes"
        if re.search(r'\bno\b', text, re.IGNORECASE):
            return "No"

    # 2c. Numeric – last standalone number in the response
    numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', text)
    if numbers:
        return numbers[-1]

    # 3. Fallback
    return text[:60].strip()


def answers_match(predicted: str, expected: str) -> bool:
    p, e = predicted.strip().lower(), expected.strip().lower()
    if p == e:
        return True
    try:
        return abs(float(p) - float(e)) < 0.01
    except ValueError:
        return False
