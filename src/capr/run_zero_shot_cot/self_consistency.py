import sys
import time

sys.path.insert(0, ".")

from capr.client.bedrock_client import BedrockClient
from capr.evaluation.aggregation import majority_vote
from capr.evaluation.answer_utils import answers_match, extract_answer
from capr.models.schemas import EvalResult, StageResult
from capr.prompts.baseline_prompts import build_standard_cot
from config import API_PAUSE, SC_K, TOKENS_SIMPLE


def run_self_consistency(
    client:     BedrockClient,
    model_id:   str,
    model_name: str,
    question:   dict,
    k:          int = SC_K,
) -> EvalResult:
    system, base_msgs = build_standard_cot(question["question"])

    answers:       list[str]         = []
    stage_results: list[StageResult] = []
    total_tokens   = 0
    total_latency  = 0.0

    for i in range(k):
        raw, tokens, latency = client.converse(
            model_id, base_msgs,
            system_prompt=system,
            temperature=0.7,
            max_tokens=TOKENS_SIMPLE,
        )
        answers.append(extract_answer(raw, question["type"]))
        stage_results.append(StageResult(
            stage_name=f"sc_chain_{i + 1}",
            prompt_sent=base_msgs[0]["content"][0]["text"],
            raw_response=raw,
            tokens_used=tokens,
            latency_s=latency,
        ))
        total_tokens  += tokens
        total_latency += latency
        time.sleep(API_PAUSE)

    predicted = majority_vote(answers)

    return EvalResult(
        method="Self-Consistency",
        model_name=model_name,
        model_id=model_id,
        question_id=question["id"],
        dataset=question["dataset"],
        question_type=question["type"],
        predicted_answer=predicted,
        expected_answer=question["answer"],
        correct=answers_match(predicted, question["answer"]),
        total_tokens=total_tokens,
        total_latency_s=total_latency,
        stage_results=stage_results,
    )
