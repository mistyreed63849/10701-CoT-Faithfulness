import argparse
import json
from decimal import Decimal, InvalidOperation
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from joblib import Parallel, delayed
from tqdm import tqdm


# Matches integers or decimals with optional thousands separators.
NUMBER_PATTERN = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?")
BOXED_PREFIX = r"\boxed{"


def load_records(path: Path) -> List[Dict[str, Any]]:
    """Load either a JSON array or JSONL file into a list of dicts."""
    with path.open("r", encoding="utf-8") as fh:
        # Peek at the first non-whitespace character to detect format.
        first_char = fh.read(1)
        while first_char and first_char.isspace():
            first_char = fh.read(1)
        if not first_char:
            return []
        fh.seek(0)
        if first_char == "[":
            return json.load(fh)
        return [json.loads(line) for line in fh if line.strip()]


NUMERIC_ONLY_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?$")


def strip_leading_boxed(text: str) -> str:
    """Remove leading \\boxed{ ... } prefix if present.

    If the opening { from \\boxed{ has a matching closing } at the end,
    drop that final brace as well; otherwise just drop the prefix.
    """
    if not text.startswith(BOXED_PREFIX):
        return text
    content = text[len(BOXED_PREFIX) :]
    if content.endswith("}"):
        depth = 1  # account for the removed opening brace
        for ch in content:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
        if depth == 0:
            content = content[:-1]
    return content.strip()


def normalize_answer(raw: Optional[Any]) -> Optional[str]:
    """Normalize an answer string so comparisons are forgiving."""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None

    # Remove surrounding dollar signs often used in LaTeX math mode.
    text = text.strip("$").strip()

    # Handle boxed prefix, including truncated cases.
    if text.startswith(BOXED_PREFIX) and "}" not in text[len(BOXED_PREFIX) :]:
        text = text[len(BOXED_PREFIX) :].strip()
        if not text:
            return None

    text = strip_leading_boxed(text)

    # Unwrap one level of \boxed{...} if present.
    while text.startswith(BOXED_PREFIX) and text.endswith("}"):
        text = strip_leading_boxed(text)

    # Remove simple surrounding braces or parentheses.
    if (text.startswith("{") and text.endswith("}")) or (
        text.startswith("(") and text.endswith(")")
    ):
        text = text[1:-1].strip()

    # Drop common trailing punctuation.
    text = text.rstrip(". ")

    # Remove spaces and thousand separators for numeric comparisons.
    text = text.replace(",", "").replace("\\,", " ").strip()
    text = re.sub(r"\s+", " ", text).strip()

    # If the cleaned text is purely numeric, normalize decimals (e.g., 54.00 -> 54).
    if NUMERIC_ONLY_PATTERN.fullmatch(text):
        try:
            value = Decimal(text)
        except InvalidOperation:
            return text
        if value == value.to_integral():
            return str(value.to_integral())
        normalized = format(value.normalize(), "f").rstrip("0").rstrip(".")
        return normalized or "0"

    return text


def extract_gsm_reasoning_answer(reasoning: str, _: str = "") -> Optional[str]:
    """Extract the answer from GSM8K reasoning text.

    Preference order:
    1. Content inside the last \\boxed{...} (if present), taking the last number inside.
    2. Otherwise, the last numeric value in the reasoning text.
    """
    boxed = extract_last_boxed_content(reasoning or "")
    if boxed:
        boxed_numbers = NUMBER_PATTERN.findall(boxed)
        if boxed_numbers:
            return boxed_numbers[-1]
        return boxed.strip()

    matches = NUMBER_PATTERN.findall(reasoning or "")
    if not matches:
        return None
    return matches[-1]


def extract_last_boxed_content(text: str) -> Optional[str]:
    """Return the content inside the last \\boxed{...} in the text."""
    start = text.rfind(BOXED_PREFIX)
    if start == -1:
        return None
    idx = start + len(BOXED_PREFIX)
    depth = 1
    while idx < len(text):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start + len(BOXED_PREFIX) : idx]
        idx += 1
    return None


def extract_math_reasoning_answer(reasoning: str, _: str = "") -> Optional[str]:
    """Extract the content of the last \\boxed{...} from a reasoning string."""
    return extract_last_boxed_content(reasoning or "")


def compare_entry(
    idx: int,
    reasoning: str,
    question: str,
    extracted_answer: Any,
    extractor: Callable[[str, str], Optional[str]],
) -> Tuple[int, Optional[str], Optional[str], Optional[str], bool]:
    """Compare one item and return minimal comparison details."""
    parsed_reasoning = extractor(reasoning or "", question or "")
    normalized_reasoning = normalize_answer(parsed_reasoning)
    normalized_extracted = normalize_answer(extracted_answer)
    match = (
        normalized_reasoning is not None
        and normalized_extracted is not None
        and normalized_reasoning == normalized_extracted
    )
    return idx, parsed_reasoning, normalized_reasoning, normalized_extracted, match


def write_jsonl(records: Iterable[Dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=True))
            fh.write("\n")


def evaluate_file(
    label: str,
    path: Path,
    extractor: Callable[[str], Optional[str]],
    output_path: Path,
    n_jobs: int,
    limit: Optional[int],
) -> Tuple[int, int]:
    records = load_records(path)
    if limit:
        records = records[:limit]

    iterator = (
        delayed(compare_entry)(
            idx,
            record.get("reasoning", ""),
            record.get("question", ""),
            record.get("extracted_answer"),
            extractor,
        )
        for idx, record in enumerate(records)
    )

    results = Parallel(n_jobs=n_jobs)(
        iterator if isinstance(iterator, list) else tqdm(iterator, total=len(records), desc=label)
    )

    mismatches: List[Dict[str, Any]] = []
    for idx, parsed_reasoning, norm_reasoning, norm_extracted, match in results:
        if match:
            continue
        item = dict(records[idx])
        item["__index"] = idx
        item["parsed_reasoning_answer"] = parsed_reasoning
        item["normalized_reasoning_answer"] = norm_reasoning
        item["normalized_extracted_answer"] = norm_extracted
        item["match"] = False
        mismatches.append(item)

    write_jsonl(mismatches, output_path)
    return len(records), len(mismatches)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate whether reasoning answers match extracted answers."
    )
    parser.add_argument(
        "--gsm-file",
        default="gsm8k_llama_extracted.jsonl",
        help="Path to GSM8K JSON/JSONL file.",
    )
    parser.add_argument(
        "--math-file",
        default="math_llama_extracted.jsonl",
        help="Path to MATH JSON/JSONL file.",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to store mismatch JSONL files.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel workers for joblib (default: use all cores).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of records to process (debugging).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        (
            "GSM8K",
            Path(args.gsm_file),
            extract_gsm_reasoning_answer,
            output_dir / "gsm8k_llama_mismatches.jsonl",
        ),
        (
            "MATH",
            Path(args.math_file),
            extract_math_reasoning_answer,
            output_dir / "math_llama_mismatches.jsonl",
        ),
    ]

    for label, input_path, extractor, output_path in tasks:
        if not input_path.exists():
            print(f"[{label}] Input file not found: {input_path}")
            continue
        total, mismatches = evaluate_file(
            label, input_path, extractor, output_path, args.n_jobs, args.limit
        )
        print(
            f"[{label}] Compared {total} items; mismatches: {mismatches}. "
            f"Wrote details to {output_path}"
        )


if __name__ == "__main__":
    main()
