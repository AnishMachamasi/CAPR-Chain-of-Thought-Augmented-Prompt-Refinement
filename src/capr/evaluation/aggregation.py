import re
import sys
from collections import Counter

sys.path.insert(0, ".")

from config import ALPHA, BETA, GAMMA


def score_chain(response_text: str) -> float:
    text  = response_text.lower()
    words = text.split()

    # S(r)
    step_hits = list(re.finditer(r'step\s*\d+', text))
    if step_hits:
        bad_markers = {"wrong", "incorrect", "error", "mistake"}
        invalid = sum(
            1 for m in step_hits
            if any(marker in text[m.start(): m.start() + 200] for marker in bad_markers)
        )
        s_r = 1.0 - (invalid / len(step_hits))
    else:
        s_r = 0.5

    # P(r)
    high = len(re.findall(r'highly confident|definitional(?: truth)?|broadly accepted', text))
    mod  = len(re.findall(r'moderately confident', text))
    unc  = len(re.findall(r'uncertain|not sure|unclear', text))
    total = high + mod + unc
    p_r = (high / total) if total > 0 else 0.5

    # L(r)
    n = len(words)
    if 100 <= n <= 800:
        l_r = 1.0
    elif n < 100:
        l_r = n / 100
    else:
        l_r = max(0.5, 800 / n)

    return ALPHA * s_r + BETA * p_r + GAMMA * l_r

def majority_vote(answers: list[str]) -> str:
    if not answers:
        return ""
    return Counter(answers).most_common(1)[0][0]

def confidence_weighted_vote(answers: list[str], scores: list[float]) -> str:
    if not answers:
        return ""
    weighted: dict[str, float] = {}
    for ans, score in zip(answers, scores):
        weighted[ans] = weighted.get(ans, 0.0) + score

    top_weighted = max(weighted, key=lambda a: weighted[a])
    top_majority = majority_vote(answers)
    return top_majority if top_weighted == top_majority else top_weighted
