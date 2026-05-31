"""
data/questions.py
=================
Benchmark questions covering the three dataset families from the paper:

  GSM8K          – grade-school arithmetic / math word problems
  MMLU           – multi-domain knowledge (multiple-choice)
  BIG-Bench Hard – compositional / logical reasoning

Each entry is a dict with keys:
  id       : unique string identifier
  dataset  : "GSM8K" | "MMLU" | "BIG-Bench Hard"
  type     : "arithmetic" | "knowledge" | "reasoning"
  question : full question text sent to the model
  answer   : expected answer (always stored as a string)
"""

QUESTIONS: list[dict] = [

    # =========================================================================
    # GSM8K  –  Arithmetic / Math Word Problems
    # =========================================================================
    {
        "id":       "gsm8k_1",
        "dataset":  "GSM8K",
        "type":     "arithmetic",
        "question": (
            "A store sells apples for $1.50 each and oranges for $2.00 each. "
            "Maria buys 4 apples and 3 oranges, then gives the cashier $15. "
            "How much change does she receive? Give only the numeric dollar amount."
        ),
        "answer": "3.0",
    },
    {
        "id":       "gsm8k_2",
        "dataset":  "GSM8K",
        "type":     "arithmetic",
        "question": (
            "A train travels at 60 mph for the first 2 hours, "
            "then at 80 mph for the next 3 hours. "
            "What is the average speed for the entire journey? "
            "Give only the numeric answer in mph."
        ),
        "answer": "72.0",
    },
    {
        "id":       "gsm8k_3",
        "dataset":  "GSM8K",
        "type":     "arithmetic",
        "question": (
            "A recipe requires 2.5 cups of flour to make 24 cookies. "
            "How many cups of flour are needed to make 60 cookies? "
            "Round to 2 decimal places and give only the numeric answer."
        ),
        "answer": "6.25",
    },
    {
        "id":       "gsm8k_4",
        "dataset":  "GSM8K",
        "type":     "arithmetic",
        "question": (
            "A water tank holds 500 litres and is currently 40% full. "
            "Water is pumped in at 25 litres per minute. "
            "How many minutes until the tank is completely full? "
            "Give only the numeric answer."
        ),
        "answer": "12.0",
    },
    {
        "id":       "gsm8k_5",
        "dataset":  "GSM8K",
        "type":     "arithmetic",
        "question": (
            "A car depreciates by 15% each year. "
            "If the car is worth $20,000 today, what will it be worth after 2 years? "
            "Round to 2 decimal places and give only the numeric answer."
        ),
        "answer": "14450.0",
    },

    # =========================================================================
    # MMLU  –  Multi-Domain Knowledge (Multiple Choice)
    # =========================================================================
    {
        "id":       "mmlu_1",
        "dataset":  "MMLU",
        "type":     "knowledge",
        "question": (
            "Which statement best describes the relationship between "
            "machine learning and artificial intelligence?\n"
            "(A) They are synonymous terms\n"
            "(B) Machine learning is a subset of AI\n"
            "(C) AI is a subset of machine learning\n"
            "(D) They are completely unrelated fields\n"
            "Answer with the letter only, e.g. A"
        ),
        "answer": "B",
    },
    {
        "id":       "mmlu_2",
        "dataset":  "MMLU",
        "type":     "knowledge",
        "question": (
            "Which of the following best describes 'hallucination' "
            "in large language models?\n"
            "(A) The model refuses to answer a question\n"
            "(B) The model generates confident but factually incorrect information\n"
            "(C) The model repeats the same sentence endlessly\n"
            "(D) The model produces very slow responses\n"
            "Answer with the letter only, e.g. A"
        ),
        "answer": "B",
    },
    {
        "id":       "mmlu_3",
        "dataset":  "MMLU",
        "type":     "knowledge",
        "question": (
            "In neural networks, what does 'backpropagation' refer to?\n"
            "(A) A forward pass through the network\n"
            "(B) An algorithm for updating weights using gradient descent\n"
            "(C) Removing neurons to reduce model size\n"
            "(D) Adding extra layers to increase model capacity\n"
            "Answer with the letter only, e.g. A"
        ),
        "answer": "B",
    },
    {
        "id":       "mmlu_4",
        "dataset":  "MMLU",
        "type":     "knowledge",
        "question": (
            "Which prompting technique elicits multi-step reasoning "
            "without providing any worked examples?\n"
            "(A) Few-shot prompting\n"
            "(B) Standard CoT prompting\n"
            "(C) Zero-Shot CoT prompting\n"
            "(D) Retrieval-Augmented Generation\n"
            "Answer with the letter only, e.g. A"
        ),
        "answer": "C",
    },

    # =========================================================================
    # BIG-Bench Hard  –  Compositional / Logical Reasoning
    # =========================================================================
    {
        "id":       "bbhard_1",
        "dataset":  "BIG-Bench Hard",
        "type":     "reasoning",
        "question": (
            "All bloops are razzles. All razzles are lazzles. "
            "Are all bloops definitely lazzles? Answer Yes or No only."
        ),
        "answer": "Yes",
    },
    {
        "id":       "bbhard_2",
        "dataset":  "BIG-Bench Hard",
        "type":     "reasoning",
        "question": (
            "A bat and a ball together cost $1.10. "
            "The bat costs exactly $1.00 more than the ball. "
            "How much does the ball cost in cents? Give only the numeric answer."
        ),
        "answer": "5",
    },
    {
        "id":       "bbhard_3",
        "dataset":  "BIG-Bench Hard",
        "type":     "reasoning",
        "question": (
            "You have a 3-litre jug and a 5-litre jug with no markings. "
            "What is the minimum number of pouring steps needed to measure "
            "exactly 4 litres? Give only the integer answer."
        ),
        "answer": "6",
    },
    {
        "id":       "bbhard_4",
        "dataset":  "BIG-Bench Hard",
        "type":     "reasoning",
        "question": (
            "If it takes 5 machines 5 minutes to make 5 widgets, "
            "how many minutes would it take 100 machines to make 100 widgets? "
            "Give only the numeric answer."
        ),
        "answer": "5",
    },
]
