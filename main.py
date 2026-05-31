import sys
import time

sys.path.insert(0, "src")

from capr.client.bedrock_client import BedrockClient
from capr.evaluation.reporter import (METHODS_ORDERED, print_results_table,
                                      save_results)
from capr.methods.baselines import (run_standard_cot, run_zero_shot,
                                    run_zero_shot_cot)
from capr.models.schemas import EvalResult
from capr.run_zero_shot_cot.capr_runner import run_capr
from capr.run_zero_shot_cot.self_consistency import run_self_consistency
from config import API_PAUSE, MODELS, RESULTS_JSON, SC_K
from data.questions import QUESTIONS

METHOD_RUNNERS: dict = {
    "Zero-Shot":        run_zero_shot,
    "Standard CoT":     run_standard_cot,
    "Zero-Shot CoT":    run_zero_shot_cot,
    "Self-Consistency": run_self_consistency,
    "CAPR (Full)":      run_capr,
}


def main() -> None:
    sep = "=" * 80
    print(sep)
    print("  CAPR – Chain-of-Thought Augmented Prompt Refinement")
    print("  Benchmarking on Amazon Bedrock")
    print(sep)
    print(f"  Models    : {list(MODELS.keys())}")
    print(f"  Methods   : {METHODS_ORDERED}")
    print(f"  Questions : {len(QUESTIONS)}  |  SC / CAPR chains k={SC_K}")
    print(sep + "\n")

    client = BedrockClient()
    all_results: list[EvalResult] = []

    for model_name, model_id in MODELS.items():
        print(f"\n{'─' * 60}")
        print(f"  Model : {model_name}  ({model_id})")
        print(f"{'─' * 60}")

        for q in QUESTIONS:
            print(f"\n  ▸ [{q['id']}]  {q['question'][:65]}...")

            for method_name, runner in METHOD_RUNNERS.items():
                try:
                    result: EvalResult = runner(client, model_id, model_name, q)
                    icon = "✓" if result.correct else "✗"
                    print(
                        f"    [{method_name:<22}] {icon}  "
                        f"pred={result.predicted_answer!r:<8}  "
                        f"exp={result.expected_answer!r:<8}  "
                        f"tokens={result.total_tokens:>5}  "
                        f"latency={result.total_latency_s:.1f}s"
                    )
                    all_results.append(result)

                except RuntimeError as exc:
                    print(f"    [{method_name:<22}] ERROR: {exc}")
                except Exception as exc:
                    print(f"    [{method_name:<22}] UNEXPECTED: {exc}")

                time.sleep(API_PAUSE)

    dataset_names = list({q["dataset"] for q in QUESTIONS})
    model_names   = list(MODELS.keys())

    print_results_table(all_results, model_names, dataset_names)
    save_results(all_results, RESULTS_JSON)


if __name__ == "__main__":
    main()
