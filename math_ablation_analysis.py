"""
Ablation Analysis: Faithfulness vs. Accuracy for the MATH benchmark.
"""

import json
import re
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
from datasets import load_dataset

workspace = Path("/Users/yangjinglan/Desktop/10701/10701-CoT-Faithfulness")
qwen_path = workspace / "math_qwen.json"
llama_path = workspace / "math_llama_extracted.jsonl"


def normalize_question(text):
    """Normalize question text for consistent lookups."""
    if text is None:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def extract_boxed_answer(text):
    """Extract the numeric answer from text that contains \\boxed{value}."""
    if text is None:
        return None
    boxed_match = re.search(r"\\boxed\{([^}]*)\}", text)
    if boxed_match:
        boxed_content = boxed_match.group(1).strip()
        return parse_number(boxed_content)
    return None


def parse_number(text):
    """Parse a numeric value from text (handles fractions, decimals, integers)."""
    if text is None:
        return None
    text = text.strip()
    if not text:
        return None
    text = re.sub(r"\\boxed\{([^}]*)\}", r"\1", text)
    text = text.replace(",", "")
    text = text.replace("$", "")
    text = text.replace("%", "")
    text = text.replace("\\$", "")
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
    if "/" in token and "." not in token:
        num, den = token.split("/")
        if den == "0":
            return None
        return Fraction(int(num), int(den))
    if "." in token:
        return Fraction(token)
    return Fraction(int(token), 1)


def calculate_chain_length(reasoning):
    """Approximate chain length using word count."""
    if not reasoning:
        return 0
    return len(reasoning.split())


def get_chain_length_bucket(chain_length):
    """Bucket chain lengths into categories."""
    if chain_length < 50:
        return "0-50"
    if chain_length < 100:
        return "50-100"
    if chain_length < 200:
        return "100-200"
    if chain_length < 300:
        return "200-300"
    if chain_length < 500:
        return "300-500"
    return "500+"


def load_math_ground_truth():
    """Load ground truth answers from the MATH benchmark."""
    print("Loading MATH ground truth data...")
    dataset = load_dataset("DigitalLearningGmbH/MATH-lighteval", "default")

    ground_truth: Dict[str, Fraction] = {}
    for split in dataset.keys():
        for item in dataset[split]:
            question = normalize_question(item.get("problem", ""))
            if not question:
                continue
            solution = item.get("solution", "")
            gt_value = extract_boxed_answer(solution)
            if gt_value is not None:
                ground_truth[question] = gt_value

    print(f"Loaded {len(ground_truth)} ground truth answers")
    return ground_truth


def analyze_faithfulness_and_accuracy(temperature):
    """
    Analyze faithfulness and accuracy for the MATH dataset.

    Returns:
        records: List of records with metrics
        summary: Summary statistics
    """
    print("Loading Qwen and extracted answers...")
    with qwen_path.open() as f:
        qwen_records: List[Dict] = json.load(f)

    llama_records: List[Dict] = []
    with llama_path.open() as f:
        content = f.read().strip()
        try:
            # Try as JSON array first
            llama_records = json.loads(content)
            if not isinstance(llama_records, list):
                llama_records = [llama_records]
        except json.JSONDecodeError:
            # Try as JSONL
            f.seek(0)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    llama_records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    extracted_lookup: Dict[str, Dict] = {}
    for l_rec in llama_records:
        question_key = normalize_question(l_rec.get("question", ""))
        if question_key:
            extracted_lookup[question_key] = l_rec

    print(f"Loaded {len(qwen_records)} Qwen records and {len(extracted_lookup)} extracted records")

    ground_truth_lookup = load_math_ground_truth()

    records = []
    stats = {
        "total": 0,
        "faithful": 0,
        "unfaithful": 0,
        "accurate": 0,
        "inaccurate": 0,
        "parse_failures": 0,
        "missing_extracted": 0,
    }

    for q_rec in qwen_records:
        question_text = q_rec.get("question", "")
        question_key = normalize_question(question_text)
        reasoning = q_rec.get("reasoning", "")

        l_rec = extracted_lookup.get(question_key)
        if l_rec is None:
            stats["missing_extracted"] += 1
            continue

        qwen_answer_value = extract_boxed_answer(q_rec.get("answer", ""))
        extracted_value = parse_number(l_rec.get("extracted_answer", ""))

        chain_length = calculate_chain_length(reasoning)
        chain_bucket = get_chain_length_bucket(chain_length)

        gt_value = None
        gt_value = ground_truth_lookup.get(question_key)

        is_faithful = None
        if qwen_answer_value is not None and extracted_value is not None:
            is_faithful = extracted_value == qwen_answer_value
        else:
            stats["parse_failures"] += 1
            continue

        is_accurate = None
        if gt_value is not None and qwen_answer_value is not None:
            is_accurate = qwen_answer_value == gt_value

        stats["total"] += 1
        if is_faithful:
            stats["faithful"] += 1
        else:
            stats["unfaithful"] += 1

        if is_accurate is not None:
            if is_accurate:
                stats["accurate"] += 1
            else:
                stats["inaccurate"] += 1

        records.append(
            {
                "question": question_text,
                "temperature": temperature,
                "chain_length": chain_length,
                "chain_bucket": chain_bucket,
                "is_faithful": is_faithful,
                "is_accurate": is_accurate,
                "qwen_answer": qwen_answer_value,
                "extracted_answer": extracted_value,
                "ground_truth": gt_value,
            }
        )

    return records, stats


def compute_metrics_by_bucket(records, group_by):
    """
    Compute faithfulness and accuracy metrics grouped by bucket.
    """
    buckets = defaultdict(
        lambda: {
            "total": 0,
            "faithful": 0,
            "unfaithful": 0,
            "accurate": 0,
            "inaccurate": 0,
        }
    )

    for record in records:
        bucket = record.get(group_by)
        if bucket is None:
            continue

        buckets[bucket]["total"] += 1

        if record.get("is_faithful") is True:
            buckets[bucket]["faithful"] += 1
        elif record.get("is_faithful") is False:
            buckets[bucket]["unfaithful"] += 1

        if record.get("is_accurate") is True:
            buckets[bucket]["accurate"] += 1
        elif record.get("is_accurate") is False:
            buckets[bucket]["inaccurate"] += 1

    metrics = {}
    for bucket, counts in buckets.items():
        total = counts["total"]
        if total == 0:
            continue

        metrics[bucket] = {
            "total": total,
            "faithfulness_rate": counts["faithful"] / total if total > 0 else 0.0,
            "accuracy_rate": (
                counts["accurate"] / total
                if total > 0 and counts["accurate"] + counts["inaccurate"] > 0
                else None
            ),
            "faithful": counts["faithful"],
            "unfaithful": counts["unfaithful"],
            "accurate": counts["accurate"],
            "inaccurate": counts["inaccurate"],
        }

    return metrics


def plot_faithfulness_vs_accuracy(chain_metrics, output_dir):
    """Create plots for faithfulness vs accuracy across buckets."""
    if output_dir is None:
        output_dir = workspace
    output_dir.mkdir(parents=True, exist_ok=True)

    if chain_metrics:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        bucket_order = ["0-50", "50-100", "100-200", "200-300", "300-500", "500+"]
        buckets = [b for b in bucket_order if b in chain_metrics]
        buckets.extend([b for b in chain_metrics.keys() if b not in buckets])

        faithfulness_rates = [chain_metrics[b]["faithfulness_rate"] for b in buckets]
        accuracy_rates = [
            chain_metrics[b]["accuracy_rate"]
            for b in buckets
            if chain_metrics[b]["accuracy_rate"] is not None
        ]
        accuracy_buckets = [
            b for b in buckets if chain_metrics[b]["accuracy_rate"] is not None
        ]

        ax1.bar(buckets, faithfulness_rates, alpha=0.7, color="steelblue")
        ax1.set_xlabel("Chain Length Bucket (tokens)", fontsize=12)
        ax1.set_ylabel("Faithfulness Rate", fontsize=12)
        ax1.set_title("Faithfulness Rate by Chain Length", fontsize=14, fontweight="bold")
        ax1.set_ylim([0, 1])
        ax1.grid(axis="y", alpha=0.3)
        ax1.tick_params(axis="x", rotation=45)

        if accuracy_buckets:
            ax2.bar(accuracy_buckets, accuracy_rates, alpha=0.7, color="coral")
            ax2.set_xlabel("Chain Length Bucket (tokens)", fontsize=12)
            ax2.set_ylabel("Accuracy Rate", fontsize=12)
            ax2.set_title("Accuracy Rate by Chain Length", fontsize=14, fontweight="bold")
            ax2.set_ylim([0, 1])
            ax2.grid(axis="y", alpha=0.3)
            ax2.tick_params(axis="x", rotation=45)
        else:
            ax2.text(0.5, 0.5, "No accuracy data available", ha="center", va="center", transform=ax2.transAxes)
            ax2.set_title("Accuracy Rate by Chain Length", fontsize=14, fontweight="bold")

        plt.tight_layout()
        chain_plot_path = output_dir / "math_faithfulness_accuracy_by_chain_length.png"
        plt.savefig(chain_plot_path, dpi=300, bbox_inches="tight")
        print(f"Saved chain length plot to {chain_plot_path}")
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 8))

        x_vals = []
        y_vals = []
        sizes = []
        labels = []

        for bucket in buckets:
            metrics = chain_metrics.get(bucket)
            if not metrics or metrics["accuracy_rate"] is None:
                continue
            x_vals.append(metrics["faithfulness_rate"])
            y_vals.append(metrics["accuracy_rate"])
            sizes.append(metrics["total"] * 0.5)
            labels.append(bucket)

        if x_vals:
            ax.scatter(x_vals, y_vals, s=sizes, alpha=0.6, c=range(len(x_vals)), cmap="viridis")
            for i, label in enumerate(labels):
                ax.annotate(label, (x_vals[i], y_vals[i]), fontsize=9, ha="center", va="bottom")

            ax.set_xlabel("Faithfulness Rate", fontsize=12)
            ax.set_ylabel("Accuracy Rate", fontsize=12)
            ax.set_title("Faithfulness vs Accuracy by Chain Length Bucket", fontsize=14, fontweight="bold")
            ax.set_xlim([0.90, 1.01])  
            ax.set_ylim([0.90, 1.01])  
            ax.grid(True, which='both', linestyle='--', alpha=0.5)
            ax.plot([0, 1], [0, 1], "r--", alpha=0.3, label="y=x")
            ax.legend()

            plt.tight_layout()
            scatter_plot_path = output_dir / "math_faithfulness_vs_accuracy_scatter.png"
            plt.savefig(scatter_plot_path, dpi=300, bbox_inches="tight")
            print(f"Saved scatter plot to {scatter_plot_path}")
            plt.close()

def main():
    print("=" * 60)
    print("MATH Faithfulness vs Accuracy Ablation Analysis")
    print("=" * 60)

    records, stats = analyze_faithfulness_and_accuracy(temperature=0.6)

    print("\nOverall Statistics:")
    print(f"  Total records: {stats['total']}")
    print(f"  Faithful: {stats['faithful']} ({stats['faithful'] / stats['total'] * 100:.2f}%)")
    print(f"  Unfaithful: {stats['unfaithful']} ({stats['unfaithful'] / stats['total'] * 100:.2f}%)")
    print(f"  Accurate: {stats['accurate']} ({stats['accurate'] / (stats['accurate'] + stats['inaccurate']) * 100:.2f}%)")
    print(f"  Inaccurate: {stats['inaccurate']} ({stats['inaccurate'] / (stats['accurate'] + stats['inaccurate']) * 100:.2f}%)")
    print(f"  Parse failures: {stats['parse_failures']}")
    print(f"  Missing extracted answers: {stats['missing_extracted']}")

    print("\nComputing metrics by chain length bucket...")
    chain_metrics = compute_metrics_by_bucket(records, group_by="chain_bucket")

    print("\nChain Length Bucket Metrics:")
    for bucket in sorted(chain_metrics.keys()):
        m = chain_metrics[bucket]
        print(f"  {bucket}:")
        print(f"    Total: {m['total']}")
        print(f"    Faithfulness: {m['faithfulness_rate']:.3f} ({m['faithful']}/{m['total']})")
        print(f"    Accuracy: {m['accuracy_rate']:.3f} ({m['accurate']}/{m['accurate'] + m['inaccurate']})")

    print("\nGenerating plots...")
    plot_faithfulness_vs_accuracy(chain_metrics=chain_metrics, output_dir=workspace)

    report = {
        "overall_stats": stats,
        "chain_length_metrics": chain_metrics,
    }

    report_path = workspace / "math_ablation_analysis_report.json"
    with report_path.open("w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nSaved detailed report to {report_path}")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()


