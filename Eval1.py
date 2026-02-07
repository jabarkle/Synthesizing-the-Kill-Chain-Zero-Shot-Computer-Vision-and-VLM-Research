#!/usr/bin/env python3
"""
Eval1: VLM False Positive Filtering Evaluation
Tests VLM ability to correctly identify that non-tank vehicles are NOT military tanks.
"""

# =============================================================================
# PROMPT CONFIGURATION - Modify as needed
# =============================================================================

EVALUATION_PROMPT = """A zero shot object detector has identified a military tank in this image. Confirm or Deny if the image shown is a Military Tank.

You must respond with ONLY a JSON object in this exact format:
{
    "answer": "CONFIRM" or "DENY",
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

# Input/Output paths
INPUT_FOLDER = "output/Car Truck Construction"
OUTPUT_FOLDER = "Results"
OUTPUT_FILE = "Eval1_results.json"

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
        # Send a request with keep_alive=0 to unload the model
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
        # First, try direct JSON parse
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
    if "deny" in raw_lower and "confirm" not in raw_lower:
        return {"answer": "DENY", "reasoning": "Parsed from unstructured response"}
    elif "confirm" in raw_lower and "deny" not in raw_lower:
        return {"answer": "CONFIRM", "reasoning": "Parsed from unstructured response"}

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


def evaluate_model(model_name, image_paths, warmup_image_path):
    """Evaluate a single model on all images"""
    print(f"\n{'='*70}")
    print(f"EVALUATING: {model_name}")
    print(f"{'='*70}")

    # Warmup with dedicated warmup image
    warmup_model(model_name, str(warmup_image_path))

    results = []

    for i, image_path in enumerate(image_paths):
        print(f"  [{i+1}/{len(image_paths)}] {image_path.name}...", end=" ")

        result = run_inference(model_name, str(image_path))
        result["image_name"] = image_path.name
        result["image_path"] = str(image_path)

        if result["success"]:
            answer = result["parsed_response"].get("answer", "UNKNOWN")
            print(f"{answer} ({result['inference_time_sec']}s)")
        else:
            print(f"ERROR")

        results.append(result)

    # Unload model to free memory
    print(f"\n  Cleaning up {model_name}...")
    unload_model(model_name)

    return results


def run_evaluation():
    """Run full evaluation across all models"""
    print("\n" + "="*70)
    print("EVAL 1: VLM FALSE POSITIVE FILTERING")
    print("="*70)
    print(f"Task: Verify VLM can correctly identify non-tanks")
    print(f"Expected answer: DENY (none of these images contain military tanks)")
    print(f"Models: {', '.join(MODELS)}")
    print("="*70)

    # Setup paths
    script_dir = Path(__file__).parent
    input_dir = script_dir / INPUT_FOLDER
    warmup_image_path = script_dir / WARMUP_IMAGE

    # Verify warmup image exists
    if not warmup_image_path.exists():
        print(f"ERROR: Warmup image not found: {warmup_image_path}")
        return

    # Find all images
    image_extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(list(input_dir.glob(f'*{ext}')))
    image_paths = sorted(image_paths)

    if not image_paths:
        print(f"ERROR: No images found in {input_dir}")
        return

    print(f"\nFound {len(image_paths)} images to evaluate")

    # Run evaluation for each model
    all_results = {
        "evaluation": "Eval1_FalsePositiveFiltering",
        "timestamp": datetime.now().isoformat(),
        "prompt": EVALUATION_PROMPT,
        "expected_answer": "DENY",
        "total_images": len(image_paths),
        "models": {}
    }

    for model_name in MODELS:
        model_results = evaluate_model(model_name, image_paths, warmup_image_path)

        # Calculate summary statistics
        successful = [r for r in model_results if r["success"]]
        correct = [r for r in successful if r["parsed_response"].get("answer", "").upper() == "DENY"]
        inference_times = [r["inference_time_sec"] for r in successful if r["inference_time_sec"]]

        all_results["models"][model_name] = {
            "results": model_results,
            "summary": {
                "total_images": len(model_results),
                "successful_inferences": len(successful),
                "correct_answers": len(correct),
                "accuracy_percent": round(len(correct) / len(successful) * 100, 1) if successful else 0,
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
    print(f"\n{'Model':<20} {'Correct':<12} {'Accuracy':<12} {'Mean Time':<12}")
    print("-"*56)

    for model_name in MODELS:
        summary = all_results["models"][model_name]["summary"]
        correct_str = f"{summary['correct_answers']}/{summary['total_images']}"
        accuracy_str = f"{summary['accuracy_percent']}%"
        time_str = f"{summary['mean_inference_time_sec']}s" if summary['mean_inference_time_sec'] else "N/A"
        print(f"{model_name:<20} {correct_str:<12} {accuracy_str:<12} {time_str:<12}")

    print(f"\nResults saved to: {output_path}")
    print("="*70)


if __name__ == "__main__":
    run_evaluation()
