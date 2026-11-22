
from pathlib import Path
from functools import lru_cache

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# =====================================================
# 1. LOCAL MODEL CONFIGURATION
# =====================================================

APP_DIR = Path(__file__).resolve().parent
BASE_DIR = APP_DIR.parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "Qwen2.5-1.5B-Instruct"
)


# =====================================================
# 2. LOAD QWEN MODEL ONCE
# =====================================================

@lru_cache(maxsize=1)
def load_llm():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Qwen model directory not found: {MODEL_PATH}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        dtype=torch.float16,
        device_map="auto",
        local_files_only=True
    )

    model.eval()

    return tokenizer, model


# =====================================================
# 3. FORMAT RETRIEVED EVIDENCE
# =====================================================

def format_evidence(evidence_df, max_passages=5):

    if evidence_df is None or evidence_df.empty:
        return "No supporting research passages were retrieved."

    evidence_blocks = []

    for _, row in evidence_df.head(max_passages).iterrows():

        chunk_id = row.get("Chunk_ID", "Unknown")
        article_id = row.get("Article_ID", "Unknown")
        title = row.get("Title", "Unknown")
        pages = row.get("Pages", "Unknown")

        text = row.get("Text", "")

        if not isinstance(text, str) or not text.strip():
            text = row.get("Text_retrieved", "")

        if not isinstance(text, str):
            text = ""

        block = f"""
Chunk ID: {chunk_id}
Article ID: {article_id}
Title: {title}
Pages: {pages}

Passage:
{text}
"""

        evidence_blocks.append(block)

    return "\n".join(evidence_blocks)


# =====================================================
# 4. GENERATE ANSWER WITH QWEN
# =====================================================

def generate_answer(
    question,
    evidence_df=None,
    use_retrieval=False,
    max_new_tokens=400
):

    tokenizer, model = load_llm()

    if use_retrieval:

        evidence = format_evidence(evidence_df)

        instruction = """
Answer the question using only the supplied research evidence.

Rules:
1. Use the evidence rather than unsupported assumptions.
2. Cite supporting passages using their Chunk IDs.
3. If the evidence is insufficient, clearly say so.
4. Never invent research findings or citations.
5. Give a clear and informative answer.
"""

    else:

        evidence = "No research passages were supplied."

        instruction = """
Answer the question using your general knowledge.

Rules:
1. Answer clearly and accurately.
2. Do not invent research-paper citations.
3. Do not claim that an answer is supported by retrieved documents.
4. If you are uncertain, acknowledge the uncertainty.
"""

    user_prompt = f"""
{instruction}

QUESTION:
{question}

RESEARCH EVIDENCE:
{evidence}
"""

    messages = [
        {
            "role": "system",
            "content": (
                "You are a research assistant specializing "
                "in phishing and cybersecurity research."
            )
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.inference_mode():

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_tokens = outputs[
        0, inputs["input_ids"].shape[1]:
    ]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return answer.strip()
