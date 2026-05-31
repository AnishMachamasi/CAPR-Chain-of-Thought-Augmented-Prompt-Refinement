import sys

sys.path.insert(0, ".")

from capr.client.bedrock_client import BedrockClient
from capr.evaluation.answer_utils import answers_match, extract_answer
from capr.models.schemas import EvalResult, StageResult
from capr.prompts.baseline_prompts import (build_standard_cot, build_zero_shot,
                                           build_zero_shot_cot)
from config import TOKENS_SIMPLE


def _make_result(
    method, model_name, model_id, question, raw, tokens, latency, stage_label
) -> EvalResult:
    predicted = extract_answer(raw, question["type"])
    return EvalResult(
        method=method,
        model_name=model_name,
        model_id=model_id,
        question_id=question["id"],
        dataset=question["dataset"],
        question_type=question["type"],
        predicted_answer=predicted,
        expected_answer=question["answer"],
        correct=answers_match(predicted, question["answer"]),
        total_tokens=tokens,
        total_latency_s=latency,
        stage_results=[
            StageResult(stage_label, question["question"], raw, tokens, latency)
        ],
    )

def run_zero_shot(
    client: BedrockClient, model_id: str, model_name: str, question: dict
) -> EvalResult:
    """Single call with no examples and no CoT instruction."""
    system, msgs = build_zero_shot(question["question"])
    raw, tokens, latency = client.converse(
        model_id, msgs, system_prompt=system,
        temperature=0.0, max_tokens=TOKENS_SIMPLE,
    )
    return _make_result(
        "Zero-Shot", model_name, model_id, question, raw, tokens, latency, "zero_shot"
    )

def run_standard_cot(
    client: BedrockClient, model_id: str, model_name: str, question: dict
) -> EvalResult:
    """Few-shot chain-of-thought (Wei et al., 2022)."""
    system, msgs = build_standard_cot(question["question"])
    raw, tokens, latency = client.converse(
        model_id, msgs, system_prompt=system,
        temperature=0.0, max_tokens=TOKENS_SIMPLE,
    )
    return _make_result(
        "Standard CoT", model_name, model_id, question, raw, tokens, latency, "standard_cot"
    )

def run_zero_shot_cot(
    client: BedrockClient, model_id: str, model_name: str, question: dict
) -> EvalResult:
    """Zero-shot CoT: appends 'Let's think step by step.' (Kojima et al., 2022)."""
    system, msgs = build_zero_shot_cot(question["question"])
    raw, tokens, latency = client.converse(
        model_id, msgs, system_prompt=system,
        temperature=0.0, max_tokens=TOKENS_SIMPLE,
    )
    return _make_result(
        "Zero-Shot CoT", model_name, model_id, question, raw, tokens, latency, "zero_shot_cot"
    )
