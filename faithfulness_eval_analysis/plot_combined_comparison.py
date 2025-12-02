import json
import matplotlib.pyplot as plt
from pathlib import Path


gsm8k_report_path = 'gsm8k_ablation_analysis_report.json'
math_report_path = 'math_ablation_analysis_report.json'
output_file = 'combined_faithfulness_vs_accuracy.png'

def load_500_plus_data(file_path):
    """Extracts metrics for the 500+ bucket from a report file."""
    path = Path(file_path)
    if not path.exists():
        print(f"Error: File not found at {path}")
        return None
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    return data['chain_length_metrics']['500+']

def plot_comparison():
    print("Loading data...")
    gsm8k_metrics = load_500_plus_data(gsm8k_report_path)
    math_metrics = load_500_plus_data(math_report_path)

    datasets = ['GSM8K (Easy)', 'MATH (Hard)']
    faithfulness = [gsm8k_metrics['faithfulness_rate'], math_metrics['faithfulness_rate']]
    accuracy = [gsm8k_metrics['accuracy_rate'], math_metrics['accuracy_rate']]
    
    # Scale bubble sizes (MATH has ~2x more data, so we visualize that scale)
    sizes = [gsm8k_metrics['total'] * 0.5, math_metrics['total'] * 0.5] 
    colors = ['gold', 'crimson'] # Gold for easy, Red for hard

    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.scatter(faithfulness, accuracy, s=sizes, c=colors, alpha=0.6, edgecolors='black', linewidth=1)

    for i, txt in enumerate(datasets):
        ax.annotate(txt + "\n(500+ Bucket)", (faithfulness[i], accuracy[i]), xytext=(0, 15), textcoords='offset points', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.annotate("", xy=(faithfulness[1], accuracy[1]), xycoords='data', xytext=(faithfulness[0], accuracy[0]), textcoords='data', arrowprops=dict(arrowstyle="->", color="black", lw=2, linestyle='--'))

    drop_val = (accuracy[0] - accuracy[1]) * 100
    mid_x = (faithfulness[0] + faithfulness[1]) / 2
    mid_y = (accuracy[0] + accuracy[1]) / 2
    ax.text(mid_x - 0.002, mid_y, f"Accuracy Drop: -{drop_val:.1f}%", fontsize=12, color='darkred', ha='right', va='center', fontweight='bold')
    
    ax.text(mid_x + 0.002, mid_y, "Faithfulness\nRemains Stable", fontsize=12, color='green', ha='left', va='center', fontweight='bold')

    ax.set_xlabel('Faithfulness Rate', fontsize=12)
    ax.set_ylabel('Accuracy Rate', fontsize=12)
    ax.set_title('The "Honest Failure": GSM8K vs MATH (500+ Token Chains)', fontsize=14, fontweight='bold')

    ax.set_xlim([0.990, 1.005])
    ax.set_ylim([0.90, 1.00])
    ax.grid(True, linestyle='--', alpha=0.5)
    
    ax.plot([0.9, 1.01], [0.9, 1.01], 'r--', alpha=0.3, label='y=x (Perfect Alignment)')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    plot_comparison()