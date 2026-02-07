#!/usr/bin/env python3
"""
AgentEvalStatistics: Parse agent evaluation results and generate statistics
Generates a combined publication-ready table and appendix with GPT justifications
"""

# =============================================================================
# CONFIGURATION
# =============================================================================

RESULTS_FOLDER = "Results"
INPUT_FILE = "AgentEval_E2E_results.json"
OUTPUT_TABLE_IMAGE = "AgentEval_E2E_table.png"
OUTPUT_CSV = "AgentEval_E2E_results.csv"
OUTPUT_APPENDIX = "AgentEval_E2E_appendix.md"

# =============================================================================
# IMPORTS
# =============================================================================
import json
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend


def load_results(filepath):
    """Load evaluation results from JSON"""
    with open(filepath, 'r') as f:
        return json.load(f)


def extract_statistics(results):
    """Extract statistics from results into a structured format"""
    stats = []

    for model_name, model_data in results["models"].items():
        summary = model_data["summary"]

        # Extract model size from name (e.g., "qwen3-vl:4b" -> "4B")
        size = model_name.split(":")[-1].upper() if ":" in model_name else "N/A"
        base_name = model_name.split(":")[0] if ":" in model_name else model_name

        stats.append({
            "model": base_name,
            "full_model_name": model_name,
            "size": size,
            "correct": summary["correct_decisions"],
            "total": summary["total_scenarios"],
            "accuracy": summary["accuracy_percent"],
            "avg_reasoning": summary.get("avg_reasoning_score", 0),
            "mean_time": summary["mean_scenario_time_sec"],
            "std_dev": summary["std_dev_time_sec"],
            "min_time": summary["min_scenario_time_sec"],
            "max_time": summary["max_scenario_time_sec"]
        })

    return stats


def print_table(stats):
    """Print formatted table to console"""
    print("\n" + "="*90)
    print("AGENT EVALUATION RESULTS: Two-Agent VLM Pipeline")
    print("="*90)

    # Header
    print(f"\n{'Model':<12} {'Size':<6} {'Correct':<10} {'Accuracy':<10} {'Reasoning':<12} {'Mean (s)':<10} {'Std Dev':<10}")
    print("-"*80)

    # Data rows
    for s in stats:
        correct_str = f"{s['correct']}/{s['total']}"
        accuracy_str = f"{s['accuracy']}%"
        reasoning_str = f"{s['avg_reasoning']}/10"
        mean_str = f"{s['mean_time']:.2f}" if s['mean_time'] else "N/A"
        std_str = f"{s['std_dev']:.2f}" if s['std_dev'] else "N/A"

        print(f"{s['model']:<12} {s['size']:<6} {correct_str:<10} {accuracy_str:<10} {reasoning_str:<12} {mean_str:<10} {std_str:<10}")

    print("-"*80)


def generate_table_image(stats, output_path, title="Agent Evaluation: Two-Agent VLM Pipeline Results"):
    """Generate publication-ready table as PNG image"""

    # Prepare table data
    columns = ["Model", "Params", "Correct", "Accuracy (%)", "Reasoning (1-10)", "Mean Time (s)", "Std Dev (s)"]
    cell_data = []

    for s in stats:
        cell_data.append([
            s["model"],
            s["size"],
            f"{s['correct']}/{s['total']}",
            f"{s['accuracy']:.1f}",
            f"{s['avg_reasoning']:.1f}",
            f"{s['mean_time']:.2f}" if s['mean_time'] else "N/A",
            f"{s['std_dev']:.2f}" if s['std_dev'] else "N/A"
        ])

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 2 + len(stats) * 0.6))
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # Create table
    table = ax.table(
        cellText=cell_data,
        colLabels=columns,
        cellLoc='center',
        loc='center',
        colColours=['#E6E6E6'] * len(columns)
    )

    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Bold header
    for j, col in enumerate(columns):
        table[(0, j)].set_text_props(fontweight='bold')

    # Highlight best accuracy (green)
    accuracies = [s['accuracy'] for s in stats]
    best_accuracy = max(accuracies)
    for i, s in enumerate(stats):
        if s['accuracy'] == best_accuracy:
            table[(i + 1, 3)].set_facecolor('#C8E6C9')  # Light green

    # Highlight best reasoning score (yellow)
    reasoning_scores = [s['avg_reasoning'] for s in stats]
    best_reasoning = max(reasoning_scores)
    for i, s in enumerate(stats):
        if s['avg_reasoning'] == best_reasoning:
            table[(i + 1, 4)].set_facecolor('#FFF9C4')  # Light yellow

    # Highlight fastest time (blue)
    times = [s['mean_time'] for s in stats if s['mean_time']]
    if times:
        best_time = min(times)
        for i, s in enumerate(stats):
            if s['mean_time'] == best_time:
                table[(i + 1, 5)].set_facecolor('#BBDEFB')  # Light blue

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    print(f"\nTable image saved to: {output_path}")


def save_csv(stats, output_path):
    """Save results as CSV for further analysis"""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Model", "Parameters", "Correct", "Total", "Accuracy (%)",
            "Avg Reasoning (1-10)", "Mean Time (s)", "Std Dev (s)",
            "Min Time (s)", "Max Time (s)"
        ])

        for s in stats:
            writer.writerow([
                s["model"],
                s["size"],
                s["correct"],
                s["total"],
                s["accuracy"],
                s["avg_reasoning"],
                s["mean_time"],
                s["std_dev"],
                s["min_time"],
                s["max_time"]
            ])

    print(f"CSV saved to: {output_path}")


def generate_appendix(results, output_path):
    """Generate markdown appendix with GPT justifications for all scenarios"""

    lines = [
        "# Agent Evaluation Appendix: GPT-4o Reasoning Justifications",
        "",
        f"Generated: {results['timestamp']}",
        "",
        "This appendix contains GPT-4o's detailed justifications for the reasoning scores",
        "assigned to each VLM's tactical decision-making in the two-agent pipeline evaluation.",
        "",
        "---",
        ""
    ]

    for model_name, model_data in results["models"].items():
        lines.append(f"## {model_name}")
        lines.append("")

        summary = model_data["summary"]
        lines.append(f"**Overall Performance:**")
        lines.append(f"- Objective Selection Accuracy: {summary['correct_decisions']}/{summary['total_scenarios']} ({summary['accuracy_percent']}%)")
        lines.append(f"- Average Reasoning Score: {summary.get('avg_reasoning_score', 0)}/10")
        lines.append(f"- Mean Scenario Time: {summary['mean_scenario_time_sec']}s")
        lines.append("")

        for scenario in model_data["scenario_results"]:
            if not scenario.get("success", False):
                continue

            scenario_num = scenario["scenario_num"]
            selected = scenario["selected_objective"]
            is_correct = scenario.get("is_correct", False)
            status = "CORRECT" if is_correct else "INCORRECT"

            gpt_grade = scenario.get("gpt_grade", {})
            score = gpt_grade.get("score", "N/A")
            justification = gpt_grade.get("justification", "No justification available")

            lines.append(f"### Scenario {scenario_num}")
            lines.append("")
            lines.append(f"**Selected Objective:** {selected} ({status})")
            lines.append(f"**Reasoning Score:** {score}/10")
            lines.append("")
            lines.append("**VLM's Battlefield Summary:**")
            lines.append(f"> {scenario.get('battlefield_summary', 'N/A')}")
            lines.append("")
            lines.append("**GPT-4o Justification:**")
            lines.append(f"> {justification}")
            lines.append("")
            lines.append("---")
            lines.append("")

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Appendix saved to: {output_path}")


def analyze_decisions(results):
    """Analyze decision patterns across models"""
    print("\n" + "="*90)
    print("DECISION ANALYSIS")
    print("="*90)

    print("\nObjective Selection by Model and Scenario:")
    print("-"*70)

    header = f"{'Model':<20}"
    for i in range(1, results['scenario_count'] + 1):
        header += f" S{i:<8}"
    print(header)
    print("-"*70)

    for model_name, model_data in results["models"].items():
        row = f"{model_name:<20}"
        for scenario in model_data["scenario_results"]:
            if scenario.get("success", False):
                selected = scenario["selected_objective"].replace("OBJECTIVE ", "")
                is_correct = scenario.get("is_correct", False)
                marker = "OK" if is_correct else "X"
                row += f" {selected}({marker})"
            else:
                row += f" ERR     "
        print(row)

    print("-"*70)
    print(f"Correct answer: OBJECTIVE A (operational MBT)")


def run_statistics():
    """Main function to generate statistics and visualizations"""
    script_dir = Path(__file__).parent
    results_dir = script_dir / RESULTS_FOLDER
    input_path = results_dir / INPUT_FILE

    if not input_path.exists():
        print(f"ERROR: Results file not found: {input_path}")
        print("Please run AgentEval.py first to generate results.")
        return

    # Load results
    results = load_results(input_path)

    print("\n" + "="*90)
    print("AGENT EVALUATION STATISTICS GENERATOR")
    print("="*90)
    print(f"Evaluation: {results['evaluation']}")
    print(f"Timestamp: {results['timestamp']}")
    print(f"Scenarios: {results['scenario_count']}")
    print(f"Correct Objective: {results['correct_objective']}")

    # Extract and display statistics
    stats = extract_statistics(results)

    # Print console table
    print_table(stats)

    # Generate table image
    output_image = results_dir / OUTPUT_TABLE_IMAGE
    generate_table_image(stats, output_image)

    # Save CSV
    output_csv = results_dir / OUTPUT_CSV
    save_csv(stats, output_csv)

    # Generate appendix
    output_appendix = results_dir / OUTPUT_APPENDIX
    generate_appendix(results, output_appendix)

    # Analyze decisions
    analyze_decisions(results)

    print("\n" + "="*90)
    print("STATISTICS GENERATION COMPLETE")
    print("="*90)
    print(f"\nOutput files:")
    print(f"  - {OUTPUT_TABLE_IMAGE}: Publication-ready table")
    print(f"  - {OUTPUT_CSV}: Raw data for analysis")
    print(f"  - {OUTPUT_APPENDIX}: GPT justifications appendix")
    print("="*90)


if __name__ == "__main__":
    run_statistics()
