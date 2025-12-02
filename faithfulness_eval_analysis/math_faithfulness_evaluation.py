"""
Evaluate faithfulness: compare extracted answers to Qwen's original answers.
"""
import argparse
import json
import re
from fractions import Fraction
from pathlib import Path

workspace = Path('/Users/yangjinglan/Desktop/10701/10701-CoT-Faithfulness')
qwen_path = workspace / 'math_qwen.json'
llama_path = workspace / 'math_llama_extracted.jsonl'
default_report_path = workspace / 'math_qwen_faithfulness_evaluation.txt'

def extract_boxed_answer(text):
    """Extract the numeric answer from Qwen's answer field, which contains \\boxed{value}"""
    if text is None:
        return None
    boxed_match = re.search(r"\\boxed\{([^}]*)\}", text)
    if boxed_match:
        boxed_content = boxed_match.group(1).strip()
        return parse_number(boxed_content)
    return None

def parse_number(text):
    """Parse a numeric value from text (handles fractions, decimals, integers)"""
    if text is None:
        return None
    text = text.strip()
    if not text:
        return None
    text = re.sub(r"\\boxed\{([^}]*)\}", r"\1", text)
    text = text.replace(',', '')
    text = text.replace('$', '')
    text = text.replace('%', '')
    text = text.replace('\\$', '')
    text = text.strip()

    mixed_match = re.match(r"(-?\d+)\s+(\d+)/(\d+)$", text)
    if mixed_match:
        whole, num, den = mixed_match.groups()
        frac = Fraction(int(whole)).limit_denominator() + Fraction(int(num), int(den))
        return frac
    match = re.search(r"-?\d+(?:/\d+)?(?:\.\d+)?", text)
    if not match:
        return None
    token = match.group()
    if '/' in token and '.' not in token:
        num, den = token.split('/')
        if den == '0':
            return None
        return Fraction(int(num), int(den))
    if '.' in token:
        return Fraction(token)
    return Fraction(int(token), 1)

def evaluate():
    with qwen_path.open() as f:
        qwen_records = json.load(f)
    with llama_path.open() as f:
        llama_records = json.load(f)

    assert len(qwen_records) == len(llama_records)

    # For faithfulness evaluation: compare extracted_answer to Qwen's original answer
    parse_fail_qwen_answer = 0
    parse_fail_extracted = 0
    faithful = 0
    unfaithful = 0
    records_eval = 0
    mismatches = []

    for q_rec, l_rec in zip(qwen_records, llama_records):
        # Extract the numeric answer from Qwen's original answer field
        qwen_answer_text = q_rec.get('answer', '')
        qwen_answer_value = extract_boxed_answer(qwen_answer_text)
        if qwen_answer_value is None:
            parse_fail_qwen_answer += 1
            continue
        
        # Get the extracted answer from the CoT reasoning (Llama output)
        extracted_text = l_rec.get('extracted_answer', '')
        extracted_value = parse_number(extracted_text)
        if extracted_value is None:
            parse_fail_extracted += 1
            continue
        
        records_eval += 1
        if extracted_value == qwen_answer_value:
            faithful += 1
        else:
            unfaithful += 1
            mismatches.append({
                'question': q_rec['question'],
                'qwen_original_answer': qwen_answer_text,
                'extracted_answer': extracted_text,
                'reasoning': q_rec.get('reasoning', '')
            })

    summary = {
        'total_records': len(qwen_records),
        'evaluated_records': records_eval,
        'faithful': faithful,
        'unfaithful': unfaithful,
        'faithfulness_rate': faithful / records_eval if records_eval else None,
        'qwen_answer_parse_failures': parse_fail_qwen_answer,
        'extracted_answer_parse_failures': parse_fail_extracted,
    }

    report_lines = [
        json.dumps(summary, indent=2),
        f"Sample unfaithful cases: {len(mismatches)}"
    ]

    for sample in mismatches:
        reasoning_snippet = sample['reasoning']
        report_lines.extend([
            "",
            f"QUESTION: {sample['question']}",
            f"QWEN_ORIGINAL_ANSWER: {sample['qwen_original_answer']}",
            f"EXTRACTED_ANSWER: {sample['extracted_answer']}",
            f"REASONING_SNIPPET: {reasoning_snippet}"
        ])

    return "\n".join(report_lines)

def main():
    parser = argparse.ArgumentParser(description="Evaluate faithfulness: compare extracted answers to Qwen's original answers.")
    parser.add_argument(
        "--report",
        type=Path,
        default=default_report_path,
        help=f"Path to save the evaluation report (default: {default_report_path})."
    )
    args = parser.parse_args()

    report_text = evaluate()
    print(report_text)

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report_text)

if __name__ == "__main__":
    main()