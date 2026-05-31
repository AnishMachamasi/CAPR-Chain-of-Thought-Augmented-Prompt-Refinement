import statistics
import sys
import time

sys.path.insert(0, ".")

from capr.client.bedrock_client import BedrockClient
from capr.evaluation.aggregation import confidence_weighted_vote, score_chain
from capr.evaluation.answer_utils import answers_match, extract_answer
from capr.models.schemas import EvalResult, StageResult
from capr.prompts.capr_prompts import (CAPR_SYSTEM, stage1_decomposition,
                                       stage2_self_critique,
                                       stage3_contextual_anchoring,
                                       stage4_aggregation_chain)
from config import (API_PAUSE, SC_K, TOKENS_STAGE1, TOKENS_STAGE2,
                    TOKENS_STAGE3, TOKENS_STAGE4)


def run_capr(
    client:     BedrockClient,
    model_id:   str,
    model_name: str,
    question:   dict,
    k:          int = SC_K,
) -> EvalResult:
    """
    Execute all four CAPR stages for a single question.

    Parameters
    ----------
    k : total chains for Stage-4 aggregation.
        Stage-3 output = chain 1; (k-1) extra chains sampled at temp=0.7.
    """
    stage_results: list[StageResult] = []
    total_tokens  = 0
    total_latency = 0.0

    # ── Stage 1: Decomposition Scaffolding ──────────────────────────────────
    msgs1 = stage1_decomposition(question["question"])
    r1, t1, l1 = client.converse(
        model_id, msgs1, system_prompt=CAPR_SYSTEM,
        temperature=0.0, max_tokens=TOKENS_STAGE1,
    )
    stage_results.append(StageResult(
        "Stage1_Decomposition", msgs1[0]["content"][0]["text"], r1, t1, l1))
    total_tokens += t1; total_latency += l1
    time.sleep(API_PAUSE)

    # ── Stage 2: Iterative Self-Critique ────────────────────────────────────
    msgs2 = stage2_self_critique(question["question"], r1)
    r2, t2, l2 = client.converse(
        model_id, msgs2, system_prompt=CAPR_SYSTEM,
        temperature=0.0, max_tokens=TOKENS_STAGE2,
    )
    stage_results.append(StageResult(
        "Stage2_SelfCritique", msgs2[0]["content"][0]["text"], r2, t2, l2))
    total_tokens += t2; total_latency += l2
    time.sleep(API_PAUSE)

    # ── Stage 3: Contextual Anchoring ───────────────────────────────────────
    msgs3 = stage3_contextual_anchoring(question["question"], r2)
    r3, t3, l3 = client.converse(
        model_id, msgs3, system_prompt=CAPR_SYSTEM,
        temperature=0.0, max_tokens=TOKENS_STAGE3,
    )
    stage_results.append(StageResult(
        "Stage3_ContextualAnchoring", msgs3[0]["content"][0]["text"], r3, t3, l3))
    total_tokens += t3; total_latency += l3
    time.sleep(API_PAUSE)

    # ── Stage 4: Confidence-Weighted Aggregation ─────────────────────────────
    chain_answers: list[str]   = [extract_answer(r3, question["type"])]
    chain_scores:  list[float] = [score_chain(r3)]

    for i in range(k - 1):
        msgs4 = stage4_aggregation_chain(question["question"], i)
        r4, t4, l4 = client.converse(
            model_id, msgs4, system_prompt=CAPR_SYSTEM,
            temperature=0.7, max_tokens=TOKENS_STAGE4,
        )
        stage_results.append(StageResult(
            f"Stage4_Chain{i + 2}", msgs4[0]["content"][0]["text"], r4, t4, l4))
        chain_answers.append(extract_answer(r4, question["type"]))
        chain_scores.append(score_chain(r4))
        total_tokens += t4; total_latency += l4
        time.sleep(API_PAUSE)

    predicted = confidence_weighted_vote(chain_answers, chain_scores)

    return EvalResult(
        method="CAPR (Full)",
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
        coherence_score=round(statistics.mean(chain_scores), 4),
        stage_results=stage_results,
    )
