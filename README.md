# 10701-CoT-Faithfulness
Repo for the Project "Measuring Chain-of-Thought Faithfulness for Thinking LLMs"

## Extractor Step

With gcloud already logged in, run
```
python llama_answer_extractor.py
```
to generate `gsm8k_llama_extracted.jsonl`.

In this file, the field `extracted_answer` contains the answer extracted from Qwen-3’s reasoning.