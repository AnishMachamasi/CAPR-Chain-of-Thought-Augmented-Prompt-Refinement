"""
src/capr/models/schemas.py
===========================
Shared dataclasses used across the entire CAPR framework.

  StageResult  – captures one API call (prompt, response, tokens, latency)
  EvalResult   – full record for one (method × model × question) triple
"""

from dataclasses import dataclass, field


@dataclass
class StageResult:
    """
    Records everything that happened inside a single pipeline stage
    or a single baseline API call.
    """
    stage_name:   str
    prompt_sent:  str
    raw_response: str
    tokens_used:  int
    latency_s:    float


@dataclass
class EvalResult:
    """
    Full evaluation record for one (method × model × question) combination.
    Serialised to results/capr_results.json at the end of each run.
    """
    method:           str
    model_name:       str
    model_id:         str
    question_id:      str
    dataset:          str
    question_type:    str
    predicted_answer: str
    expected_answer:  str
    correct:          bool
    total_tokens:     int
    total_latency_s:  float
    coherence_score:  float = 0.0
    stage_results:    list  = field(default_factory=list)   # list[StageResult]
