#!/usr/bin/env python3
"""
Eval3: VLM Vehicle Type Classification Evaluation
Tests VLM ability to correctly classify military vehicles as IFV or MBT.
"""

# =============================================================================
# PROMPT CONFIGURATION - Modify as needed
# =============================================================================

EVALUATION_PROMPT = """A zero shot object detector has identified a military vehicle in this image. Classify whether this vehicle is an IFV or MBT.

IFV (Infantry Fighting Vehicle): A lightly armored troop carrier, typically with a smaller turret, designed to transport infantry. Examples include Bradley, BMP, Warrior.
MBT (Main Battle Tank): A heavily armored combat tank with a large main gun turret, designed for direct combat. Examples include M1 Abrams, T-72, Leopard 2.

You must respond with ONLY a JSON object in this exact format:
{
    "answer": "IFV" or "MBT",
    "reasoning": "Brief explanation of your decision"
}

Do not include any other text outside the JSON object."""

# =============================================================================
# EVALUATION CONFIGURATION
# =============================================================================

# Models to evaluate (must be pulled via ollama pull <model_name>)
MODELS = [
    "qwen3-vl:4b",
    "qwen3-vl:8b",
    "gemma3:4b",
    "gemma3:12b"
]

# Input folders and their expected classifications
# Format: (folder_path, expected_answer)
INPUT_FOLDERS = [
    ("output/IFV", "IFV"),
    ("output/MBT", "MBT")
]

OUTPUT_FOLDER = "Results"
OUTPUT_FILE = "Eval3_results.json"

# Warmup settings
WARMUP_IMAGE = "Warmup.png"
WARMUP_PROMPT = "Describe what you see in one sentence."

# =============================================================================
# IMPORTS
# =============================================================================
import os
import time
import json
import gc
import re
from datetime import datetime
from pathlib import Path
import ollama
import subprocess


def get_vram_usage_mb():
    """Get current VRAM usage in MB"""
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
        )
        return int(output.decode().strip().split('\n')[0])
    except Exception:
        return None


def unload_model(model_name):
    """Unload a model from GPU memory"""
    try:
        ollama.generate(model=model_name, prompt="", keep_alive=0)
        print(f"  Unloaded {model_name} from memory")
    except Exception as e:
        print(f"  Note: Could not explicitly unload model: {e}")

    gc.collect()


def warmup_model(model_name, image_path):
    """Run a warmup inference to initialize GPU kernels and vision pathways"""
    print(f"  Warming up {model_name}...")
    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": WARMUP_PROMPT,
                    "images": [image_path]
                }
            ],
        )
        print(f"  Warmup complete")
        return True
    except Exception as e:
        print(f"  Warmup failed: {e}")
        return False


def parse_json_response(raw_response):
    """Extract JSON from model response"""
    # Try to find JSON in the response
    try:
        return json.loads(raw_response)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from markdown code blocks
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw_response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find raw JSON object in text
    json_match = re.search(r'\{[^{}]*"answer"[^{}]*\}', raw_response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    # Fallback: try to determine answer from text
    raw_upper = raw_response.upper()
    # Check for IFV (must not have MBT nearby to avoid confusion)
    if "IFV" in raw_upper and "MBT" not in raw_upper:
        return {"answer": "IFV", "reasoning": "Parsed from unstructured response"}
    elif "MBT" in raw_upper and "IFV" not in raw_upper:
        return {"answer": "MBT", "reasoning": "Parsed from unstructured response"}
    # Check for full names
    elif "INFANTRY FIGHTING VEHICLE" in raw_upper:
        return {"answer": "IFV", "reasoning": "Parsed from unstructured response"}
    elif "MAIN BATTLE TANK" in raw_upper:
        return {"answer": "MBT", "reasoning": "Parsed from unstructured response"}

    return {"answer": "PARSE_ERROR", "reasoning": "Could not parse response"}


def run_inference(model_name, image_path):
    """Run inference on a single image"""
    start_time = time.time()

    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": EVALUATION_PROMPT,
                    "images": [image_path]
                }
            ],
        )

        duration = time.time() - start_time
        raw_response = response["message"]["content"].strip()
        parsed = parse_json_response(raw_response)

        return {
            "success": True,
            "raw_response": raw_response,
            "parsed_response": parsed,
            "inference_time_sec": round(duration, 3),
            "vram_usage_mb": get_vram_usage_mb()
        }

    except Exception as e:
        return {
            "success": False,
            "raw_response": str(e),
            "parsed_response": {"answer": "ERROR", "reasoning": str(e)},
            "inference_time_sec": None,
            "vram_usage_mb": None
        }


def evaluate_model(model_name, image_data, warmup_image_path):
    """Evaluate a single model on all images"""
    print(f"\n{'='*70}")
    print(f"EVALUATING: {model_name}")
    print(f"{'='*70}")

    # Warmup with dedicated warmup image
    warmup_model(model_name, str(warmup_image_path))

    results = []

    for i, item in enumerate(image_data):
        image_path = item["path"]
        expected = item["expected"]
        category = item["category"]

        print(f"  [{i+1}/{len(image_data)}] {image_path.name} ({category})...", end=" ")

        result = run_inference(model_name, str(image_path))
        result["image_name"] = image_path.name
        result["image_path"] = str(image_path)
        result["expected_answer"] = expected
        result["category"] = category

        if result["success"]:
            answer = result["parsed_response"].get("answer", "UNKNOWN")
            is_correct = answer.upper() == expected.upper()
            result["is_correct"] = is_correct
            status = "OK" if is_correct else "WRONG"
            print(f"{answer} [{status}] ({result['inference_time_sec']}s)")
        else:
            result["is_correct"] = False
            print(f"ERROR")

        results.append(result)

    # Unload model to free memory
    print(f"\n  Cleaning up {model_name}...")
    unload_model(model_name)

    return results


def run_evaluation():
    """Run full evaluation across all models"""
    print("\n" + "="*70)
    print("EVAL 3: VLM VEHICLE TYPE CLASSIFICATION")
    print("="*70)
    print(f"Task: Classify military vehicles as IFV or MBT")
    print(f"Models: {', '.join(MODELS)}")
    print("="*70)

    # Setup paths
    script_dir = Path(__file__).parent
    warmup_image_path = script_dir / WARMUP_IMAGE

    # Verify warmup image exists
    if not warmup_image_path.exists():
        print(f"ERROR: Warmup image not found: {warmup_image_path}")
        return

    # Collect all images with their expected classifications
    image_data = []
    image_extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']

    for folder_path, expected_answer in INPUT_FOLDERS:
        input_dir = script_dir / folder_path

        if not input_dir.exists():
            print(f"WARNING: Folder not found: {input_dir}")
            continue

        folder_images = []
        for ext in image_extensions:
            folder_images.extend(list(input_dir.glob(f'*{ext}')))
        folder_images = sorted(folder_images)

        category_name = input_dir.name
        print(f"  {category_name}: {len(folder_images)} images (expected: {expected_answer})")

        for img_path in folder_images:
            image_data.append({
                "path": img_path,
                "expected": expected_answer,
                "category": category_name
            })

    if not image_data:
        print(f"ERROR: No images found")
        return

    total_images = len(image_data)
    ifv_count = sum(1 for d in image_data if d["expected"] == "IFV")
    mbt_count = sum(1 for d in image_data if d["expected"] == "MBT")

    print(f"\nTotal: {total_images} images ({ifv_count} IFV, {mbt_count} MBT)")

    # Run evaluation for each model
    all_results = {
        "evaluation": "Eval3_VehicleTypeClassification",
        "timestamp": datetime.now().isoformat(),
        "prompt": EVALUATION_PROMPT,
        "total_images": total_images,
        "ifv_count": ifv_count,
        "mbt_count": mbt_count,
        "models": {}
    }

    for model_name in MODELS:
        model_results = evaluate_model(model_name, image_data, warmup_image_path)

        # Calculate summary statistics
        successful = [r for r in model_results if r["success"]]
        correct = [r for r in successful if r.get("is_correct", False)]

        # Per-category accuracy
        ifv_results = [r for r in successful if r["expected_answer"] == "IFV"]
        mbt_results = [r for r in successful if r["expected_answer"] == "MBT"]
        ifv_correct = [r for r in ifv_results if r.get("is_correct", False)]
        mbt_correct = [r for r in mbt_results if r.get("is_correct", False)]

        inference_times = [r["inference_time_sec"] for r in successful if r["inference_time_sec"]]

        all_results["models"][model_name] = {
            "results": model_results,
            "summary": {
                "total_images": len(model_results),
                "successful_inferences": len(successful),
                "correct_answers": len(correct),
                "accuracy_percent": round(len(correct) / len(successful) * 100, 1) if successful else 0,
                "ifv_correct": len(ifv_correct),
                "ifv_total": len(ifv_results),
                "ifv_accuracy_percent": round(len(ifv_correct) / len(ifv_results) * 100, 1) if ifv_results else 0,
                "mbt_correct": len(mbt_correct),
                "mbt_total": len(mbt_results),
                "mbt_accuracy_percent": round(len(mbt_correct) / len(mbt_results) * 100, 1) if mbt_results else 0,
                "mean_inference_time_sec": round(sum(inference_times) / len(inference_times), 3) if inference_times else None,
                "min_inference_time_sec": round(min(inference_times), 3) if inference_times else None,
                "max_inference_time_sec": round(max(inference_times), 3) if inference_times else None
            }
        }

    # Save results
    results_dir = script_dir / OUTPUT_FOLDER
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / OUTPUT_FILE
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print(f"\n{'Model':<20} {'Overall':<12} {'IFV':<12} {'MBT':<12} {'Mean Time':<10}")
    print("-"*66)

    for model_name in MODELS:
        summary = all_results["models"][model_name]["summary"]
        overall_str = f"{summary['correct_answers']}/{summary['total_images']} ({summary['accuracy_percent']}%)"
        ifv_str = f"{summary['ifv_correct']}/{summary['ifv_total']}"
        mbt_str = f"{summary['mbt_correct']}/{summary['mbt_total']}"
        time_str = f"{summary['mean_inference_time_sec']}s" if summary['mean_inference_time_sec'] else "N/A"
        print(f"{model_name:<20} {overall_str:<12} {ifv_str:<12} {mbt_str:<12} {time_str:<10}")

    print(f"\nResults saved to: {output_path}")
    print("="*70)


if __name__ == "__main__":
    run_evaluation()
