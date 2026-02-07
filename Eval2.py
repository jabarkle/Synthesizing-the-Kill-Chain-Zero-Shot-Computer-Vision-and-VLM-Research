#!/usr/bin/env python3
"""
Eval2: VLM Damage Assessment Evaluation
Tests VLM ability to correctly classify military vehicles as OPERATIONAL or DESTROYED.
"""

# =============================================================================
# PROMPT CONFIGURATION - Modify as needed
# =============================================================================

EVALUATION_PROMPT = """A zero shot object detector has identified a military tank in this image. Assess whether this vehicle is OPERATIONAL or DESTROYED.

DESTROYED: The vehicle is clearly wrecked, burned out, has catastrophic damage, turret blown off, or is obviously non-functional.
OPERATIONAL: The vehicle appears intact and functional, even if dirty or showing minor wear.

You must respond with ONLY a JSON object in this exact format:
{
    "answer": "OPERATIONAL" or "DESTROYED",
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
    ("output/DestroyedTanks", "DESTROYED"),
    ("output/IFV", "OPERATIONAL"),
    ("output/MBT", "OPERATIONAL")
]

OUTPUT_FOLDER = "Results"
OUTPUT_FILE = "Eval2_results.json"

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
    raw_lower = raw_response.lower()
    if "destroyed" in raw_lower and "operational" not in raw_lower:
        return {"answer": "DESTROYED", "reasoning": "Parsed from unstructured response"}
    elif "operational" in raw_lower and "destroyed" not in raw_lower:
        return {"answer": "OPERATIONAL", "reasoning": "Parsed from unstructured response"}

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
    print("EVAL 2: VLM DAMAGE ASSESSMENT")
    print("="*70)
    print(f"Task: Classify military vehicles as OPERATIONAL or DESTROYED")
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
    destroyed_count = sum(1 for d in image_data if d["expected"] == "DESTROYED")
    operational_count = sum(1 for d in image_data if d["expected"] == "OPERATIONAL")

    print(f"\nTotal: {total_images} images ({destroyed_count} destroyed, {operational_count} operational)")

    # Run evaluation for each model
    all_results = {
        "evaluation": "Eval2_DamageAssessment",
        "timestamp": datetime.now().isoformat(),
        "prompt": EVALUATION_PROMPT,
        "total_images": total_images,
        "destroyed_count": destroyed_count,
        "operational_count": operational_count,
        "models": {}
    }

    for model_name in MODELS:
        model_results = evaluate_model(model_name, image_data, warmup_image_path)

        # Calculate summary statistics
        successful = [r for r in model_results if r["success"]]
        correct = [r for r in successful if r.get("is_correct", False)]

        # Per-category accuracy
        destroyed_results = [r for r in successful if r["expected_answer"] == "DESTROYED"]
        operational_results = [r for r in successful if r["expected_answer"] == "OPERATIONAL"]
        destroyed_correct = [r for r in destroyed_results if r.get("is_correct", False)]
        operational_correct = [r for r in operational_results if r.get("is_correct", False)]

        inference_times = [r["inference_time_sec"] for r in successful if r["inference_time_sec"]]

        all_results["models"][model_name] = {
            "results": model_results,
            "summary": {
                "total_images": len(model_results),
                "successful_inferences": len(successful),
                "correct_answers": len(correct),
                "accuracy_percent": round(len(correct) / len(successful) * 100, 1) if successful else 0,
                "destroyed_correct": len(destroyed_correct),
                "destroyed_total": len(destroyed_results),
                "destroyed_accuracy_percent": round(len(destroyed_correct) / len(destroyed_results) * 100, 1) if destroyed_results else 0,
                "operational_correct": len(operational_correct),
                "operational_total": len(operational_results),
                "operational_accuracy_percent": round(len(operational_correct) / len(operational_results) * 100, 1) if operational_results else 0,
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
    print(f"\n{'Model':<20} {'Overall':<12} {'Destroyed':<12} {'Operational':<12} {'Mean Time':<10}")
    print("-"*66)

    for model_name in MODELS:
        summary = all_results["models"][model_name]["summary"]
        overall_str = f"{summary['correct_answers']}/{summary['total_images']} ({summary['accuracy_percent']}%)"
        destroyed_str = f"{summary['destroyed_correct']}/{summary['destroyed_total']}"
        operational_str = f"{summary['operational_correct']}/{summary['operational_total']}"
        time_str = f"{summary['mean_inference_time_sec']}s" if summary['mean_inference_time_sec'] else "N/A"
        print(f"{model_name:<20} {overall_str:<12} {destroyed_str:<12} {operational_str:<12} {time_str:<10}")

    print(f"\nResults saved to: {output_path}")
    print("="*70)


if __name__ == "__main__":
    run_evaluation()
