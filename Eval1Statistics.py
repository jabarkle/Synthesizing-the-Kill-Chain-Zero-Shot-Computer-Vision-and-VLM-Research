#!/usr/bin/env python3
"""
Eval1Statistics: Parse evaluation results and generate statistics table
Generates a publication-ready table for academic papers
"""

# =============================================================================
# CONFIGURATION
# =============================================================================

RESULTS_FOLDER = "Results"
INPUT_FILE = "Eval1_results.json"
OUTPUT_TABLE_IMAGE = "Eval1_table.png"
OUTPUT_CSV = "Eval1_results.csv"

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

        # Calculate std dev from individual results
        inference_times = [
            r["inference_time_sec"]
            for r in model_data["results"]
            if r["success"] and r["inference_time_sec"]
        ]

        if len(inference_times) > 1:
            mean_time = sum(inference_times) / len(inference_times)
            variance = sum((t - mean_time) ** 2 for t in inference_times) / (len(inference_times) - 1)
            std_dev = variance ** 0.5
        else:
            std_dev = 0.0

        stats.append({
            "model": base_name,
            "size": size,
            "correct": summary["correct_answers"],
            "total": summary["total_images"],
            "accuracy": summary["accuracy_percent"],
            "mean_time": summary["mean_inference_time_sec"],
            "std_dev": round(std_dev, 3),
            "min_time": summary["min_inference_time_sec"],
            "max_time": summary["max_inference_time_sec"]
        })

    return stats


def print_table(stats, expected_answer):
    """Print formatted table to console"""
    print("\n" + "="*80)
    print("EVAL 1 RESULTS: False Positive Filtering")
    print(f"Expected Answer: {expected_answer}")
    print("="*80)

    # Header
    print(f"\n{'Model':<12} {'Size':<6} {'Correct':<10} {'Accuracy':<10} {'Mean (s)':<10} {'Std Dev':<10}")
    print("-"*68)

    # Data rows
    for s in stats:
        correct_str = f"{s['correct']}/{s['total']}"
        accuracy_str = f"{s['accuracy']}%"
        mean_str = f"{s['mean_time']:.3f}" if s['mean_time'] else "N/A"
        std_str = f"{s['std_dev']:.3f}" if s['std_dev'] else "N/A"

        print(f"{s['model']:<12} {s['size']:<6} {correct_str:<10} {accuracy_str:<10} {mean_str:<10} {std_str:<10}")

    print("-"*68)


def generate_table_image(stats, output_path, title="Eval 1: False Positive Filtering Results"):
    """Generate publication-ready table as PNG image"""

    # Prepare table data
    columns = ["Model", "Parameters", "Correct", "Accuracy (%)", "Mean Time (s)", "Std Dev (s)"]
    cell_data = []

    for s in stats:
        cell_data.append([
            s["model"],
            s["size"],
            f"{s['correct']}/{s['total']}",
            f"{s['accuracy']:.1f}",
            f"{s['mean_time']:.3f}" if s['mean_time'] else "N/A",
            f"{s['std_dev']:.3f}" if s['std_dev'] else "N/A"
        ])

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 2 + len(stats) * 0.5))
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

    # Highlight best accuracy
    accuracies = [s['accuracy'] for s in stats]
    best_accuracy = max(accuracies)
    for i, s in enumerate(stats):
        if s['accuracy'] == best_accuracy:
            table[(i + 1, 3)].set_facecolor('#C8E6C9')  # Light green

    # Highlight fastest time
    times = [s['mean_time'] for s in stats if s['mean_time']]
    if times:
        best_time = min(times)
        for i, s in enumerate(stats):
            if s['mean_time'] == best_time:
                table[(i + 1, 4)].set_facecolor('#BBDEFB')  # Light blue

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    print(f"\nTable image saved to: {output_path}")


def save_csv(stats, output_path):
    """Save results as CSV for further analysis"""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Parameters", "Correct", "Total", "Accuracy (%)",
                        "Mean Time (s)", "Std Dev (s)", "Min Time (s)", "Max Time (s)"])

        for s in stats:
            writer.writerow([
                s["model"],
                s["size"],
                s["correct"],
                s["total"],
                s["accuracy"],
                s["mean_time"],
                s["std_dev"],
                s["min_time"],
                s["max_time"]
            ])

    print(f"CSV saved to: {output_path}")


def analyze_per_image(results):
    """Analyze which images were incorrectly classified by each model"""
    print("\n" + "="*80)
    print("PER-IMAGE ANALYSIS")
    print("="*80)

    # Get all image names
    first_model = list(results["models"].keys())[0]
    image_names = [r["image_name"] for r in results["models"][first_model]["results"]]

    incorrect_by_image = {img: [] for img in image_names}

    for model_name, model_data in results["models"].items():
        for result in model_data["results"]:
            if result["success"]:
                answer = result["parsed_response"].get("answer", "").upper()
                if answer != "DENY":
                    incorrect_by_image[result["image_name"]].append(model_name)

    # Report images that confused models
    problem_images = {k: v for k, v in incorrect_by_image.items() if v}

    if problem_images:
        print("\nImages incorrectly classified (false positives):")
        print("-"*60)
        for image, models in sorted(problem_images.items()):
            print(f"  {image}: {', '.join(models)}")
    else:
        print("\nAll images correctly classified by all models.")

    return problem_images


def run_statistics():
    """Main function to generate statistics and visualizations"""
    script_dir = Path(__file__).parent
    results_dir = script_dir / RESULTS_FOLDER
    input_path = results_dir / INPUT_FILE

    if not input_path.exists():
        print(f"ERROR: Results file not found: {input_path}")
        print("Please run Eval1.py first to generate results.")
        return

    # Load results
    results = load_results(input_path)

    print("\n" + "="*80)
    print("EVAL 1 STATISTICS GENERATOR")
    print("="*80)
    print(f"Evaluation: {results['evaluation']}")
    print(f"Timestamp: {results['timestamp']}")
    print(f"Total Images: {results['total_images']}")
    print(f"Expected Answer: {results['expected_answer']}")

    # Extract and display statistics
    stats = extract_statistics(results)

    # Print console table
    print_table(stats, results['expected_answer'])

    # Generate table image
    output_image = results_dir / OUTPUT_TABLE_IMAGE
    generate_table_image(stats, output_image)

    # Save CSV
    output_csv = results_dir / OUTPUT_CSV
    save_csv(stats, output_csv)

    # Per-image analysis
    analyze_per_image(results)

    print("\n" + "="*80)
    print("STATISTICS GENERATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    run_statistics()
