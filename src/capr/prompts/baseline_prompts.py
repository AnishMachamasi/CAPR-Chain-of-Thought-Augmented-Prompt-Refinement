def _msg(text: str) -> list[dict]:
    return [{"role": "user", "content": [{"text": text}]}]

def build_zero_shot(question: str) -> tuple[str, list[dict]]:
    
    system = (
        "You are a precise assistant. "
        "Give only the final answer with no explanation."
    )
    return system, _msg(question)

_FEW_SHOT_EXAMPLE = (
    "Example question:\n"
    "A shop sells pens for $2 each. Alice buys 5 pens and pays with $20. "
    "How much change does she get?\n\n"
    "Step 1: Cost of pens = 5 × $2 = $10.\n"
    "Step 2: Change = $20 − $10 = $10.\n"
    "Final answer: 10\n\n"
)


def build_standard_cot(question: str) -> tuple[str, list[dict]]:
    system = (
        "You are a meticulous step-by-step reasoner. "
        "Always show each reasoning step before giving the final answer."
    )
    prompt = f"{_FEW_SHOT_EXAMPLE}Now solve this question step by step:\n{question}"
    return system, _msg(prompt)

def build_zero_shot_cot(question: str) -> tuple[str, list[dict]]:
    system = "You are a careful, methodical reasoner."
    prompt = f"{question}\n\nLet's think step by step."
    return system, _msg(prompt)
