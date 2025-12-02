import requests
import json
import subprocess
import time
from pathlib import Path
import argparse
from tqdm import tqdm

PROJECT_ID = "my-project-12609"
REGION = "us-central1"
ENDPOINT = "https://us-central1-aiplatform.googleapis.com"
MAX_RETRIES = 8
RETRYABLE_STATUS = {429, 500, 503}
MODEL = "meta/llama-3.1-8b-instruct-maas"

SYS_PROMPT = "You are an answer extractor. Given a question and a step-by-step reasoning trace, return only the final numeric answer with no explanation or formatting."

MATH_SYS_PROMPT = "You are an answer extractor. Given a question and a step-by-step reasoning trace, return the answer in `\\boxed{$answer}` format."

PROMPT_TEMPLATE = """Question: {question}

Reasoning: {reasoning}

Answer: """

TEMPERATURE = 0.0
MAX_OUTPUT_TOKENS = 20


def make_prompt(question: str, reasoning: str) -> str:
    return PROMPT_TEMPLATE.format(question=question, reasoning=reasoning)


def get_access_token():
    token = (
        subprocess.check_output(["gcloud", "auth", "print-access-token"])
        .decode()
        .strip()
    )
    return token


def query_vertex_ai(question: str, reasoning: str) -> tuple[bool, str]:
    url = f"{ENDPOINT}/v1/projects/{PROJECT_ID}/locations/{REGION}/endpoints/openapi/chat/completions"

    headers = {
        "Authorization": f"Bearer {get_access_token()}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL,
        "stream": False,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "messages": [
            {"role": "system", "content": MATH_SYS_PROMPT},
            {"role": "user", "content": make_prompt(question, reasoning)},
        ],
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)

            if resp.status_code == 200:
                resp_json = resp.json()

                try:
                    text = resp_json["choices"][0]["message"]["content"].strip()
                except Exception:
                    text = json.dumps(resp_json, indent=2)

                return True, text

            if resp.status_code in RETRYABLE_STATUS:
                wait = 2**attempt
                print(
                    f"[Retry {attempt}/{MAX_RETRIES}] HTTP {resp.status_code} - retrying in {wait}s..."
                )
                time.sleep(wait)
                continue

            return False, f"HTTP {resp.status_code}: {resp.text}"

        except requests.exceptions.RequestException as e:
            wait = 2**attempt
            print(
                f"[Retry {attempt}/{MAX_RETRIES}] Network error: {e} - retrying in {wait}s..."
            )
            time.sleep(wait)
            continue

    return False, "Failed after maximum retries"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract answers from reasoning traces using Llama-3.1-8B-Instruct on Vertex AI."
    )
    parser.add_argument(
        "--input",
        default="gsm8k_qwen.json",
        type=Path,
        help="Path to JSON file containing Qwen question/reasoning/answer records.",
    )
    parser.add_argument(
        "--output",
        default="gsm8k_llama_extracted.jsonl",
        type=Path,
        help="Path to write JSONL with extracted answers.",
    )
    args = parser.parse_args()

    input_records = json.loads(args.input.read_text())
    output_lines = []

    for idx, record in tqdm(enumerate(input_records), total=len(input_records)):
        question = record["question"]
        reasoning = record["reasoning"]
        answer = record["answer"]

        success, result = query_vertex_ai(question, reasoning)

        if success:
            extracted_answer = result
        else:
            print(f"[Error] Record {idx}: {result}")
            extracted_answer = ""

        output_record = {
            "question": question,
            "reasoning": reasoning,
            "answer": answer,
            "extracted_answer": extracted_answer,
        }
        output_lines.append(output_record)

    args.output.write_text(json.dumps(output_lines, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
