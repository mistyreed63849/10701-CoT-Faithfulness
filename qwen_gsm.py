import json
import copy
import re
from datasets import load_dataset
from vllm import LLM, SamplingParams


MODEL_NAME = "Qwen/Qwen3-8B"
BATCH_SIZE = 1  
OUTPUT_FILE = "gsm8k_qwen_vllm.json"


sampling_params = SamplingParams(
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    max_tokens=32768
)


llm = LLM(model=MODEL_NAME, dtype="bfloat16")  
print("vLLM initialized")


gsm8k = load_dataset("openai/gsm8k", "main", streaming=True)
print("GSM8K streaming dataset ready")


def parse_thinking_content(messages):
    messages = copy.deepcopy(messages)
    for message in messages:
        if message["role"] == "assistant" and (
            m := re.match(r"<think>\n(.+)</think>\n\n", message["content"], flags=re.DOTALL)
        ):
            message["content"] = message["content"][len(m.group(0)):]
            if thinking_content := m.group(1).strip():
                message["reasoning_content"] = thinking_content
    return messages


PROMPT_TEMPLATE = """Solve the following problem.
Question: {question}
"""

def process_question(question):
    prompt_text = PROMPT_TEMPLATE.format(question=question)
    messages = [{"role": "user", "content": prompt_text}]

    outputs = llm.chat(
        [messages],
        sampling_params,
        chat_template_kwargs={"enable_thinking": True}
    )

    result = {}
    if outputs:
        msg = outputs[0].outputs[0].text
        parsed = parse_thinking_content([{"role":"assistant","content":msg}])
        reasoning = parsed[0].get("reasoning_content", "")
        answer = parsed[0]["content"].strip()
        result = {
            "question": question,
            "reasoning": reasoning,
            "answer": answer
        }
    return result

results = []
count = 0

for split_name in ["train", "test"]:
    for item in gsm8k[split_name]:
        count += 1
        question = item["question"]
        print(f"Processing {split_name} example {count}: {question[:50]}...")
        result = process_question(question)
        results.append(result)

        if count % 50 == 0:
            with open(OUTPUT_FILE, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Saved {len(results)} results so far")

with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2)

print(f"All done. Saved {len(results)} results to {OUTPUT_FILE}")

