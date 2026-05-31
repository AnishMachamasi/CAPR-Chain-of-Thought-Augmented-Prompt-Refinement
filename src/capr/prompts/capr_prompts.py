CAPR_SYSTEM: str = (
    "You are a rigorous, step-by-step reasoner who always verifies your work "
    "before giving a final answer."
)


def _msg(text: str) -> list[dict]:
    return [{"role": "user", "content": [{"text": text}]}]


def stage1_decomposition(question: str) -> list[dict]:
    prompt = (
        "Before solving the problem below, complete these steps first:\n"
        "  1. List every sub-question you must answer, in the order "
        "they must be answered.\n"
        "  2. Label each sub-question with its reasoning type: "
        "arithmetic, look-up, analogical, definitional, or logical.\n"
        "  3. Solve each sub-question in sequence.\n"
        "  4. State the final answer on its own line as: FINAL ANSWER: <answer>\n\n"
        f"Problem:\n{question}"
    )
    return _msg(prompt)


def stage2_self_critique(question: str, stage1_response: str) -> list[dict]:
    prompt = (
        f"Original problem:\n{question}\n\n"
        f"Your previous reasoning:\n{stage1_response}\n\n"
        "Review EACH numbered step of the reasoning above.\n"
        "For every step, state whether it is:\n"
        "  (a) logically valid\n"
        "  (b) arithmetically correct (where applicable)\n"
        "  (c) consistent with all earlier steps and the problem statement\n"
        "  (d) necessary for solving the problem\n\n"
        "If any step fails a criterion, identify the error and write a corrected "
        "version. Then re-derive the final answer from the corrected chain.\n"
        "End with: FINAL ANSWER: <answer>"
    )
    return _msg(prompt)

def stage3_contextual_anchoring(question: str, stage2_response: str) -> list[dict]:
    prompt = (
        f"Original problem:\n{question}\n\n"
        f"Current reasoning chain:\n{stage2_response}\n\n"
        "List the 3–5 most important factual premises your reasoning depends on.\n"
        "For each premise state:\n"
        "  (a) its type: definitional truth | broadly accepted empirical fact | "
        "claim specific to the problem statement\n"
        "  (b) your confidence: HIGHLY CONFIDENT | MODERATELY CONFIDENT | UNCERTAIN\n\n"
        "Flag any UNCERTAIN premise and revise the reasoning to reduce dependence "
        "on it if possible.\n"
        "End with: FINAL ANSWER: <answer>"
    )
    return _msg(prompt)

def stage4_aggregation_chain(question: str, attempt_index: int) -> list[dict]:
    """
    Generate one independent reasoning chain for Stage-4 aggregation.
    Called with temperature=0.7 to ensure diversity across chains.

    attempt_index: 0-based index of this extra chain
                   (attempt 1 = Stage-3 output; extras start at attempt 2)
    """
    prompt = (
        "Solve the following problem carefully, showing all reasoning steps.\n"
        "At the very end write: FINAL ANSWER: <answer>\n\n"
        f"Problem:\n{question}\n\n"
        f"(Independent attempt {attempt_index + 2})"
    )
    return _msg(prompt)
