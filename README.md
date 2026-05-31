# CAPR – Chain-of-Thought Augmented Prompt Refinement

> **CAPR: A Structured Prompt Optimisation Framework for Enhancing Accuracy and Reliability in Large Language Models**  
> Group 04 · Master of Data Science · Charles Darwin University · 2026

---

## Overview

CAPR is a four-stage prompt refinement framework that systematically improves LLM reasoning accuracy. Unlike standard Chain-of-Thought prompting, CAPR forces the model to decompose problems, self-critique its own reasoning, anchor factual premises, and aggregate multiple chains using a confidence-weighted vote.

The implementation runs entirely via the **Amazon Bedrock Converse API** and benchmarks five prompting methods across seven LLMs on three datasets (GSM8K, MMLU, BIG-Bench Hard).

---

## Project Structure

```
capr/
├── main.py                             # Entry point – benchmarking loop
├── config.py                           # All hyperparameters and settings
├── requirements.txt                    # Python dependencies
├── pyproject.toml                      # Poetry project config
│
├── data/
│   └── questions.py                    # 13 benchmark questions (GSM8K / MMLU / BIG-Bench Hard)
│
├── src/capr/
│   ├── client/
│   │   └── bedrock_client.py           # boto3 Converse API wrapper + retry logic
│   │
│   ├── prompts/
│   │   ├── capr_prompts.py             # CAPR Stages 1–4 prompt builders
│   │   └── baseline_prompts.py         # Zero-Shot / CoT / Zero-Shot CoT templates
│   │
│   ├── run_zero_shot_cot/
│   │   ├── capr_runner.py              # Full CAPR 4-stage pipeline runner
│   │   └── self_consistency.py         # Self-Consistency baseline
│   │
│   ├── evaluation/
│   │   ├── aggregation.py              # score_chain + confidence_weighted_vote
│   │   ├── answer_utils.py             # Answer extraction and matching
│   │   └── reporter.py                 # Results table + JSON export
│   │
│   └── models/
│       └── schemas.py                  # StageResult / EvalResult dataclasses
│
├── results/
│   └── capr_results.json               # Auto-generated after each run
│
└── tests/
    ├── test_answer_utils.py            # Unit tests for extraction and matching
    └── test_aggregation.py             # Unit tests for scoring and voting
```

---

## Environment Setup

### Prerequisites

| Requirement | Version |
|---|---|
| Python | ≥ 3.10 |
| boto3 | ≥ 1.34.0 |
| AWS Account | with Bedrock model access |

### Installation

```bash
# Clone or extract the repository
cd capr/

# Install dependencies
pip install -r requirements.txt

# Or using Poetry
poetry install
```

### AWS Credentials

CAPR calls Amazon Bedrock via boto3. Provide credentials using **one** of these methods:

**Option 1 – Environment variables (recommended for CI/scripts)**
```bash
export AWS_ACCESS_KEY_ID=YOUR_ACCESS_KEY_ID
export AWS_SECRET_ACCESS_KEY=YOUR_SECRET_ACCESS_KEY
export AWS_DEFAULT_REGION=us-east-1
```

**Option 2 – AWS CLI**
```bash
aws configure
# Enter your Access Key ID, Secret Access Key, region (us-east-1), and output format
```

**Option 3 – IAM Role** (if running on EC2 / SageMaker, credentials are auto-detected)

### Enable Bedrock Model Access

Before running, you must request access to each model in the AWS Console:

1. Go to **AWS Console → Amazon Bedrock → Model access**
2. Click **Manage model access**
3. Enable the models you want to use (see table below)
4. Wait for access to be granted (usually instant for Amazon models; may take minutes for third-party)

---

## Supported Models

| Alias | Model ID | Provider |
|---|---|---|
| Nova-Micro | `amazon.nova-micro-v1:0` | Amazon |
| Nova-Lite | `amazon.nova-lite-v1:0` | Amazon |
| Nova-Pro | `amazon.nova-pro-v1:0` | Amazon |
| Claude-Haiku | `anthropic.claude-3-haiku-20240307-v1:0` | Anthropic |
| Claude-Sonnet | `anthropic.claude-3-5-sonnet-20241022-v2:0` | Anthropic |
| Llama3-8B | `meta.llama3-8b-instruct-v1:0` | Meta |
| Mistral-Large | `mistral.mistral-large-2402-v1:0` | Mistral AI |

To disable a model, comment it out in `config.py → MODELS`.

---

## Running the Benchmark

```bash
# Run the full benchmark (all methods × all models × all questions)
python main.py

# Run unit tests
python -m pytest tests/ -v
```

The benchmark evaluates all five methods on every model and prints live `✓/✗` results per call, followed by summary accuracy tables.

---

## Parameter Settings

All tunable settings are centralised in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `AWS_REGION` | `us-east-1` | Bedrock inference region |
| `SC_K` | `3` | Number of reasoning chains for Self-Consistency and CAPR Stage 4 |
| `ALPHA` | `0.5` | Coherence weight for step-validity S(r) |
| `BETA` | `0.3` | Coherence weight for premise-confidence P(r) |
| `GAMMA` | `0.2` | Coherence weight for length-normalisation L(r) |
| `API_PAUSE` | `1.0 s` | Delay between consecutive Bedrock calls |
| `MAX_RETRIES` | `3` | Retries on throttling / transient errors |
| `TOKENS_STAGE1` | `768` | Token budget for Stage 1 (Decomposition) |
| `TOKENS_STAGE2` | `768` | Token budget for Stage 2 (Self-Critique) |
| `TOKENS_STAGE3` | `512` | Token budget for Stage 3 (Contextual Anchoring) |
| `TOKENS_STAGE4` | `512` | Token budget for Stage 4 (Aggregation chains) |
| `TOKENS_SIMPLE` | `256` | Token budget for baseline methods |

> **Note:** `ALPHA + BETA + GAMMA` must always equal `1.0`.

---

## Prompting Methods

| # | Method | Description |
|---|---|---|
| 1 | **Zero-Shot** | No examples, no CoT; direct answer |
| 2 | **Standard CoT** | One worked example + "step-by-step" instruction |
| 3 | **Zero-Shot CoT** | "Let's think step by step" (Kojima et al., 2022) |
| 4 | **Self-Consistency** | k=3 chains with majority vote (Wang et al., 2022) |
| 5 | **CAPR (Full)** | 4-stage pipeline (this paper) |

### CAPR Pipeline

```
Question
  │
  ▼  Stage 1 – Decomposition Scaffolding       (temperature = 0)
  │  Enumerate sub-goals with reasoning-type labels before solving
  │
  ▼  Stage 2 – Iterative Self-Critique          (temperature = 0)
  │  Audit every step: logical validity, arithmetic, consistency, necessity
  │
  ▼  Stage 3 – Contextual Anchoring             (temperature = 0)
  │  Surface factual premises, rate confidence, revise UNCERTAIN ones
  │
  ▼  Stage 4 – Confidence-Weighted Aggregation  (temperature = 0.7)
     C(r) = α·S(r) + β·P(r) + γ·L(r)
     Confidence-weighted vote across k chains → FINAL ANSWER
```

---

## Datasets

| Dataset | Type | Questions | Description |
|---|---|---|---|
| GSM8K | Arithmetic | 4 | Grade-school math word problems |
| MMLU | Knowledge | 5 | Multi-domain multiple-choice questions |
| BIG-Bench Hard | Reasoning | 4 | Compositional and logical reasoning |

---

## Output

**Console** – live `✓/✗` per API call, then printed tables showing:
- Overall accuracy (%) by method × model
- Per-dataset breakdown (GSM8K / MMLU / BIG-Bench Hard)
- Average token usage and latency per method

**`results/capr_results.json`** – machine-readable record of every `EvalResult`, including all stage prompts, raw model responses, token counts, coherence scores, and latencies for offline analysis.

---

## Authors

**Group 04 – PRT840 IT Thesis, Charles Darwin University (2026)**

| Name | Student ID |
|---|---|
| Anish Machamasi |S389151 |
| Anjan Shrestha | S389022|
| Sanil Khadka | S390105|
| Sujan Sharma | S388975|


