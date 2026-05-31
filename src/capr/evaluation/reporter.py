import json
import os
import statistics
from dataclasses import asdict

from capr.models.schemas import EvalResult

METHODS_ORDERED: list[str] = [
    "Zero-Shot",
    "Standard CoT",
    "Zero-Shot CoT",
    "Self-Consistency",
    "CAPR (Full)",
]

def print_results_table(
    all_results:   list[EvalResult],
    model_names:   list[str],
    dataset_names: list[str],
) -> None:
    """
    Print three sections to stdout:
      1. Overall accuracy (%) by method × model
      2. Per-dataset accuracy breakdown
      3. Average tokens and latency per method
    """
    col_w = 14

    def _acc(method: str, model: str, ds: str | None) -> str:
        subset = [
            r for r in all_results
            if r.method == method
            and r.model_name == model
            and (ds is None or r.dataset == ds)
        ]
        if not subset:
            return f"{'N/A':>{col_w}}"
        pct = 100 * sum(r.correct for r in subset) / len(subset)
        return f"{pct:>{col_w}.1f}"

    def _table(title: str, ds: str | None) -> None:
        print(f"\n{title}")
        header = f"{'Method':<22}" + "".join(f"{m:>{col_w}}" for m in model_names)
        print("─" * len(header))
        print(header)
        print("─" * len(header))
        for method in METHODS_ORDERED:
            row = f"{method:<22}" + "".join(_acc(method, m, ds) for m in model_names)
            print(row)
        print("─" * len(header))

    sep = "=" * 80
    print(f"\n{sep}")
    print("  CAPR BENCHMARK RESULTS")
    print(sep)
    _table("  ── Overall Accuracy (%) ──", None)
    for ds in sorted(dataset_names):
        _table(f"  ── {ds} ──", ds)

    print(f"\n{sep}")
    print("  Average resource usage per question by method")
    print(sep)
    for method in METHODS_ORDERED:
        subset = [r for r in all_results if r.method == method]
        if subset:
            avg_tok = statistics.mean(r.total_tokens    for r in subset)
            avg_lat = statistics.mean(r.total_latency_s for r in subset)
            print(f"  {method:<22}  tokens: {avg_tok:>7.0f}   latency: {avg_lat:>5.2f}s")
    print(sep + "\n")

def save_results(all_results: list[EvalResult], path: str) -> None:
    """Serialise all results (including nested StageResult lists) to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    print(f"  Results saved → {path}")
