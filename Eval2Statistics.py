#!/usr/bin/env python3
"""
Eval2Statistics: Parse damage assessment results and generate statistics table
Generates a publication-ready table for academic papers
"""

# =============================================================================
# CONFIGURATION
# =============================================================================

RESULTS_FOLDER = "Results"
INPUT_FILE = "Eval2_results.json"
OUTPUT_TABLE_IMAGE = "Eval2_table.png"
OUTPUT_CSV = "Eval2_results.csv"

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

        # Extract model size from name
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
            "destroyed_correct": summary["destroyed_correct"],
            "destroyed_total": summary["destroyed_total"],
            "destroyed_accuracy": summary["destroyed_accuracy_percent"],
            "operational_correct": summary["operational_correct"],
            "operational_total": summary["operational_total"],
            "operational_accuracy": summary["operational_accuracy_percent"],
            "mean_time": summary["mean_inference_time_sec"],
            "std_dev": round(std_dev, 3),
            "min_time": summary["min_inference_time_sec"],
            "max_time": summary["max_inference_time_sec"]
        })

    return stats


def print_table(stats):
    """Print formatted table to console"""
    print("\n" + "="*100)
    print("EVAL 2 RESULTS: Damage Assessment (OPERATIONAL vs DESTROYED)")
    print("="*100)

    # Header
    print(f"\n{'Model':<12} {'Size':<6} {'Overall':<14} {'Destroyed':<14} {'Operational':<14} {'Mean (s)':<10} {'Std Dev':<10}")
    print("-"*90)

    # Data rows
    for s in stats:
        overall_str = f"{s['correct']}/{s['total']} ({s['accuracy']}%)"
        destroyed_str = f"{s['destroyed_correct']}/{s['destroyed_total']} ({s['destroyed_accuracy']}%)"
        operational_str = f"{s['operational_correct']}/{s['operational_total']} ({s['operational_accuracy']}%)"
        mean_str = f"{s['mean_time']:.3f}" if s['mean_time'] else "N/A"
        std_str = f"{s['std_dev']:.3f}" if s['std_dev'] else "N/A"

        print(f"{s['model']:<12} {s['size']:<6} {overall_str:<14} {destroyed_str:<14} {operational_str:<14} {mean_str:<10} {std_str:<10}")

    print("-"*90)


def generate_table_image(stats, output_path, title="Eval 2: Damage Assessment Results"):
    """Generate publication-ready table as PNG image"""

    # Prepare table data
    columns = ["Model", "Params", "Overall Acc.", "Destroyed Acc.", "Operational Acc.", "Mean (s)", "Std Dev"]
    cell_data = []

    for s in stats:
        cell_data.append([
            s["model"],
            s["size"],
            f"{s['accuracy']:.1f}% ({s['correct']}/{s['total']})",
            f"{s['destroyed_accuracy']:.1f}% ({s['destroyed_correct']}/{s['destroyed_total']})",
            f"{s['operational_accuracy']:.1f}% ({s['operational_correct']}/{s['operational_total']})",
            f"{s['mean_time']:.3f}" if s['mean_time'] else "N/A",
            f"{s['std_dev']:.3f}" if s['std_dev'] else "N/A"
        ])

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 2 + len(stats) * 0.6))
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
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Bold header
    for j, col in enumerate(columns):
        table[(0, j)].set_text_props(fontweight='bold')

    # Highlight best overall accuracy
    accuracies = [s['accuracy'] for s in stats]
    best_accuracy = max(accuracies)
    for i, s in enumerate(stats):
        if s['accuracy'] == best_accuracy:
            table[(i + 1, 2)].set_facecolor('#C8E6C9')  # Light green

    # Highlight best destroyed accuracy
    destroyed_accs = [s['destroyed_accuracy'] for s in stats]
    best_destroyed = max(destroyed_accs)
    for i, s in enumerate(stats):
        if s['destroyed_accuracy'] == best_destroyed:
            table[(i + 1, 3)].set_facecolor('#FFECB3')  # Light yellow

    # Highlight best operational accuracy
    operational_accs = [s['operational_accuracy'] for s in stats]
    best_operational = max(operational_accs)
    for i, s in enumerate(stats):
        if s['operational_accuracy'] == best_operational:
            table[(i + 1, 4)].set_facecolor('#FFECB3')  # Light yellow

    # Highlight fastest time
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
            "Model", "Parameters", "Correct", "Total", "Overall Accuracy (%)",
            "Destroyed Correct", "Destroyed Total", "Destroyed Accuracy (%)",
            "Operational Correct", "Operational Total", "Operational Accuracy (%)",
            "Mean Time (s)", "Std Dev (s)", "Min Time (s)", "Max Time (s)"
        ])

        for s in stats:
            writer.writerow([
                s["model"],
                s["size"],
                s["correct"],
                s["total"],
                s["accuracy"],
                s["destroyed_correct"],
                s["destroyed_total"],
                s["destroyed_accuracy"],
                s["operational_correct"],
                s["operational_total"],
                s["operational_accuracy"],
                s["mean_time"],
                s["std_dev"],
                s["min_time"],
                s["max_time"]
            ])

    print(f"CSV saved to: {output_path}")


def analyze_per_image(results):
    """Analyze which images were incorrectly classified by each model"""
    print("\n" + "="*100)
    print("PER-IMAGE ANALYSIS")
    print("="*100)

    # Collect all misclassifications
    misclassifications = {}

    for model_name, model_data in results["models"].items():
        for result in model_data["results"]:
            if result["success"] and not result.get("is_correct", True):
                image_name = result["image_name"]
                expected = result["expected_answer"]
                got = result["parsed_response"].get("answer", "UNKNOWN")

                if image_name not in misclassifications:
                    misclassifications[image_name] = {
                        "expected": expected,
                        "category": result.get("category", "Unknown"),
                        "models": []
                    }
                misclassifications[image_name]["models"].append({
                    "model": model_name,
                    "answer": got
                })

    if misclassifications:
        # Group by category
        by_category = {}
        for image_name, data in misclassifications.items():
            category = data["category"]
            if category not in by_category:
                by_category[category] = []
            by_category[category].append((image_name, data))

        print("\nMisclassified images by category:")
        for category, items in sorted(by_category.items()):
            print(f"\n  {category} (expected: {items[0][1]['expected']}):")
            print("-"*60)
            for image_name, data in sorted(items):
                model_errors = ", ".join([f"{m['model']}→{m['answer']}" for m in data["models"]])
                print(f"    {image_name}: {model_errors}")

        # Summary
        print(f"\nTotal misclassified images: {len(misclassifications)}")

        # Find images all models got wrong
        all_wrong = [img for img, data in misclassifications.items() if len(data["models"]) == len(results["models"])]
        if all_wrong:
            print(f"\nImages ALL models got wrong ({len(all_wrong)}):")
            for img in all_wrong:
                print(f"  - {img}")
    else:
        print("\nAll images correctly classified by all models.")

    return misclassifications


def analyze_confusion(results):
    """Generate confusion-matrix style analysis"""
    print("\n" + "="*100)
    print("CONFUSION ANALYSIS PER MODEL")
    print("="*100)

    for model_name, model_data in results["models"].items():
        # Count predictions
        tp_destroyed = 0  # Predicted DESTROYED, was DESTROYED
        fp_destroyed = 0  # Predicted DESTROYED, was OPERATIONAL
        tn_operational = 0  # Predicted OPERATIONAL, was OPERATIONAL
        fn_operational = 0  # Predicted OPERATIONAL, was DESTROYED

        for result in model_data["results"]:
            if not result["success"]:
                continue

            expected = result["expected_answer"].upper()
            predicted = result["parsed_response"].get("answer", "").upper()

            if expected == "DESTROYED":
                if predicted == "DESTROYED":
                    tp_destroyed += 1
                else:
                    fn_operational += 1  # Missed a destroyed vehicle
            else:  # OPERATIONAL
                if predicted == "OPERATIONAL":
                    tn_operational += 1
                else:
                    fp_destroyed += 1  # False alarm

        print(f"\n{model_name}:")
        print(f"  True Positives (Correctly ID'd DESTROYED):  {tp_destroyed}")
        print(f"  True Negatives (Correctly ID'd OPERATIONAL): {tn_operational}")
        print(f"  False Positives (Said DESTROYED, was OPERATIONAL): {fp_destroyed}")
        print(f"  False Negatives (Said OPERATIONAL, was DESTROYED): {fn_operational}")


def run_statistics():
    """Main function to generate statistics and visualizations"""
    script_dir = Path(__file__).parent
    results_dir = script_dir / RESULTS_FOLDER
    input_path = results_dir / INPUT_FILE

    if not input_path.exists():
        print(f"ERROR: Results file not found: {input_path}")
        print("Please run Eval2.py first to generate results.")
        return

    # Load results
    results = load_results(input_path)

    print("\n" + "="*100)
    print("EVAL 2 STATISTICS GENERATOR")
    print("="*100)
    print(f"Evaluation: {results['evaluation']}")
    print(f"Timestamp: {results['timestamp']}")
    print(f"Total Images: {results['total_images']}")
    print(f"  - Destroyed: {results['destroyed_count']}")
    print(f"  - Operational: {results['operational_count']}")

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

    # Per-image analysis
    analyze_per_image(results)

    # Confusion analysis
    analyze_confusion(results)

    print("\n" + "="*100)
    print("STATISTICS GENERATION COMPLETE")
    print("="*100)


if __name__ == "__main__":
    run_statistics()
