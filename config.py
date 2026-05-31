AWS_REGION: str = "us-east-1"

MODELS: dict[str, str] = {
    "Nova-Micro":    "amazon.nova-micro-v1:0",
    "Nova-Lite":     "amazon.nova-lite-v1:0",
    "Nova-Pro":      "amazon.nova-pro-v1:0",
    "Claude-Haiku":  "anthropic.claude-3-haiku-20240307-v1:0",
    "Claude-Sonnet": "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "Llama3-8B":     "meta.llama3-8b-instruct-v1:0",
    "Mistral-Large": "mistral.mistral-large-2402-v1:0",
}

SC_K: int = 3          # chains for Self-Consistency and CAPR Stage 4

ALPHA: float = 0.5     # step-validity  S(r)
BETA:  float = 0.3     # premise-confidence  P(r)
GAMMA: float = 0.2     # length-normalisation  L(r)

API_PAUSE:   float = 1.0    # seconds between consecutive Bedrock calls
MAX_RETRIES: int   = 3      # retries on throttling / transient errors

TOKENS_STAGE1: int = 768    # Decomposition Scaffolding
TOKENS_STAGE2: int = 768    # Iterative Self-Critique
TOKENS_STAGE3: int = 512    # Contextual Anchoring
TOKENS_STAGE4: int = 512    # Aggregation chains
TOKENS_SIMPLE: int = 256    # Baselines (Zero-Shot, CoT, Zero-Shot CoT)

RESULTS_DIR:  str = "results"
RESULTS_JSON: str = "results/capr_results.json"
