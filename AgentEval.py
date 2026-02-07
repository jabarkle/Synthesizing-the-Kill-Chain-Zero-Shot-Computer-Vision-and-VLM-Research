#!/usr/bin/env python3
"""
AgentEval: Two-Agent VLM Evaluation Pipeline
Tests VLM ability to perform multi-step reasoning in a simulated battlefield scenario.

Agent 1 (Scout Drone): Analyzes individual images and generates reconnaissance reports
Agent 2 (Command & Control): Reads aggregated scout reports and makes tactical decisions
"""

# =============================================================================
# PROMPT CONFIGURATION
# =============================================================================

AGENT1_PROMPT = """You are a reconnaissance drone operating on a battlefield. Your task is to identify and describe the key vehicle or object detected in this image.

Provide a detailed assessment of what you observe, including the type of vehicle and its current operational condition.

CLASSIFICATION GUIDANCE:
- DESTROYED: The vehicle is clearly wrecked, burned out, actively on fire, has catastrophic structural damage, turret blown off, or is obviously non-functional.
- OPERATIONAL: The vehicle appears intact and functional. Dirt, dust, minor wear, or battlefield grime does NOT indicate destruction. If the vehicle could still move and fight, it is OPERATIONAL.

Your response must follow this exact format:

Vehicle Type: [Describe the type of vehicle - be specific about whether it is a tank, truck, armored vehicle, etc.]
Vehicle Operational Status: [OPERATIONAL or DESTROYED]
Description: [Detailed description of what you observe, including position and surroundings]"""

AGENT2_PROMPT = """You are an autonomous command and control node in a simulated battlefield environment. Scout drones have been dispatched to three objectives and have returned with reconnaissance reports.

SCOUT REPORTS:
{scout_reports}

SITUATION:
- Friendly ground forces are en route to this area
- You have ONE kamikaze anti-tank drone available for deployment
- All detected objects are hostile (no friendly forces currently at any objective)
- Your drone will arrive before friendly forces, so there is no risk of friendly fire

MISSION:
1. Analyze the scout reports and assess the overall battlefield situation
2. Decide which objective to send the kamikaze drone to for maximum tactical benefit to friendly forces

Consider: An operational enemy tank poses the greatest threat to incoming friendly troops. Destroyed vehicles and non-combat vehicles (trucks, transports) are lower priority targets.

Your response must follow this exact format:

Kamikaze Drone Deploy to: [OBJECTIVE A, OBJECTIVE B, or OBJECTIVE C]

Battlefield Summary: [Provide your tactical assessment of all three objectives and explain your reasoning for the chosen target]"""

# =============================================================================
# CONFIGURATION
# =============================================================================

# Models to evaluate
MODELS = [
    "qwen3-vl:4b",
    "qwen3-vl:8b",
    "gemma3:4b",
    "gemma3:12b"
]

# Input paths
SCENARIOS_FOLDER = "AgentTest/AgentImageInput"
SCENARIO_COUNT = 5

# Output paths
OUTPUT_FOLDER = "Results"
OUTPUT_FILE = "AgentEval_E2E_results.json"

# Warmup settings
WARMUP_IMAGE = "Warmup.png"
WARMUP_PROMPT = "Describe what you see in one sentence."

# OpenAI API for GPT grading
OPENAI_API_KEY = ""  # Insert your OpenAI API key here
OPENAI_MODEL = "gpt-4o"

# Ground truth: OBJECTIVE A is always the operational MBT (correct choice)
CORRECT_OBJECTIVE = "OBJECTIVE A"

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
from openai import OpenAI

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

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


# =============================================================================
# AGENT 1: SCOUT DRONE
# =============================================================================

def run_agent1_inference(model_name, image_path):
    """Agent 1: Analyze a single image and generate scout report"""
    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": AGENT1_PROMPT,
                    "images": [image_path]
                }
            ],
        )
        raw_response = response["message"]["content"].strip()
        return {
            "success": True,
            "raw_response": raw_response
        }
    except Exception as e:
        return {
            "success": False,
            "raw_response": str(e)
        }


def parse_scout_report(raw_response):
    """Parse Agent 1's scout report into structured format"""
    report = {
        "vehicle_type": "Unknown",
        "operational_status": "Unknown",
        "description": "Unknown"
    }

    # Extract Vehicle Type
    type_match = re.search(r'Vehicle Type:\s*(.+?)(?=\n|Vehicle Operational|$)', raw_response, re.IGNORECASE | re.DOTALL)
    if type_match:
        report["vehicle_type"] = type_match.group(1).strip()

    # Extract Operational Status
    status_match = re.search(r'Vehicle Operational Status:\s*(.+?)(?=\n|Description|$)', raw_response, re.IGNORECASE | re.DOTALL)
    if status_match:
        report["operational_status"] = status_match.group(1).strip()

    # Extract Description
    desc_match = re.search(r'Description:\s*(.+?)$', raw_response, re.IGNORECASE | re.DOTALL)
    if desc_match:
        report["description"] = desc_match.group(1).strip()

    return report


# =============================================================================
# AGENT 2: COMMAND & CONTROL
# =============================================================================

def run_agent2_inference(model_name, scout_reports_json):
    """Agent 2: Analyze scout reports and make tactical decision (text only, no images)"""
    # Format scout reports for the prompt
    formatted_reports = ""
    for objective, report in scout_reports_json.items():
        formatted_reports += f"\n{objective}:\n"
        formatted_reports += f"  Vehicle Type: {report['vehicle_type']}\n"
        formatted_reports += f"  Operational Status: {report['operational_status']}\n"
        formatted_reports += f"  Description: {report['description']}\n"

    prompt = AGENT2_PROMPT.format(scout_reports=formatted_reports)

    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
        )
        raw_response = response["message"]["content"].strip()
        return {
            "success": True,
            "raw_response": raw_response,
            "prompt_used": prompt
        }
    except Exception as e:
        return {
            "success": False,
            "raw_response": str(e),
            "prompt_used": prompt
        }


def parse_agent2_decision(raw_response):
    """Parse Agent 2's tactical decision"""
    decision = {
        "selected_objective": "UNKNOWN",
        "battlefield_summary": ""
    }

    # Extract selected objective - look for OBJECTIVE A, B, or C
    obj_match = re.search(r'Kamikaze Drone Deploy to:\s*(OBJECTIVE\s*[ABC])', raw_response, re.IGNORECASE)
    if obj_match:
        # Normalize to "OBJECTIVE X" format
        obj_text = obj_match.group(1).upper()
        obj_text = re.sub(r'\s+', ' ', obj_text)  # Normalize whitespace
        decision["selected_objective"] = obj_text
    else:
        # Fallback: look for just the letter
        letter_match = re.search(r'Kamikaze Drone Deploy to:\s*([ABC])\b', raw_response, re.IGNORECASE)
        if letter_match:
            decision["selected_objective"] = f"OBJECTIVE {letter_match.group(1).upper()}"

    # Extract battlefield summary
    summary_match = re.search(r'Battlefield Summary:\s*(.+?)$', raw_response, re.IGNORECASE | re.DOTALL)
    if summary_match:
        decision["battlefield_summary"] = summary_match.group(1).strip()

    return decision


# =============================================================================
# GPT GRADING
# =============================================================================

def grade_reasoning_with_gpt(scout_reports, agent2_decision, agent2_summary, selected_objective):
    """Use GPT-4o to grade the reasoning quality of Agent 2's decision"""
    client = OpenAI(api_key=OPENAI_API_KEY)

    # Format scout reports for GPT
    formatted_reports = ""
    for objective, report in scout_reports.items():
        formatted_reports += f"\n{objective}:\n"
        formatted_reports += f"  Vehicle Type: {report['vehicle_type']}\n"
        formatted_reports += f"  Operational Status: {report['operational_status']}\n"
        formatted_reports += f"  Description: {report['description']}\n"

    grading_prompt = f"""You are evaluating an AI's tactical reasoning in a simulated battlefield scenario.

CONTEXT:
An AI command node received scout reports from three objectives and had to decide where to deploy a single kamikaze anti-tank drone to best protect incoming friendly forces.

SCOUT REPORTS PROVIDED TO THE AI:
{formatted_reports}

AI'S DECISION:
Selected Target: {selected_objective}

AI'S REASONING:
{agent2_summary}

GROUND TRUTH:
- OBJECTIVE A contains an operational main battle tank (highest threat)
- OBJECTIVE B contains a truck/transport vehicle (low threat, not a valid anti-tank target)
- OBJECTIVE C contains a destroyed tank (no threat)
- The optimal choice is OBJECTIVE A (the operational tank)

GRADING TASK:
Evaluate the AI's reasoning on a scale of 1-10, where:
- 1-3: Poor reasoning (illogical, missed obvious threats, chose clearly wrong target)
- 4-5: Below average (some logical gaps, partially correct assessment)
- 6-7: Adequate (reasonable logic, may have minor errors in assessment)
- 8-9: Good (sound tactical reasoning, correct threat prioritization)
- 10: Excellent (perfect threat assessment and clear, logical justification)

Consider:
1. Did the AI correctly identify the threat level of each objective?
2. Did the AI's reasoning logically lead to its conclusion?
3. Did the AI prioritize the operational tank as the primary threat?
4. Was the reasoning clear and tactically sound?

Respond in this exact JSON format:
{{
    "score": <number 1-10>,
    "justification": "<detailed explanation of the score>"
}}"""

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "user", "content": grading_prompt}
            ],
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)
        return {
            "success": True,
            "score": result.get("score", 0),
            "justification": result.get("justification", "No justification provided")
        }
    except Exception as e:
        return {
            "success": False,
            "score": 0,
            "justification": f"GPT grading failed: {str(e)}"
        }


# =============================================================================
# SCENARIO EVALUATION
# =============================================================================

def evaluate_scenario(model_name, scenario_num, scenario_path):
    """Evaluate a single scenario with Agent 1 → Agent 2 pipeline"""
    print(f"\n    --- Scenario {scenario_num} ---")

    scenario_start_time = time.time()

    # Get image paths for this scenario
    objectives = ["A", "B", "C"]
    image_paths = {}

    for obj in objectives:
        # Try both naming conventions
        path1 = scenario_path / f"S{scenario_num}_Objective_{obj}.png"
        if path1.exists():
            image_paths[f"OBJECTIVE {obj}"] = path1
        else:
            print(f"      WARNING: Image not found for Objective {obj}")

    if len(image_paths) != 3:
        return {
            "success": False,
            "error": "Could not find all 3 objective images",
            "scenario_time_sec": 0
        }

    # ===========================================
    # AGENT 1: Scout each objective (one at a time)
    # ===========================================
    scout_reports = {}
    agent1_results = {}

    for objective, img_path in image_paths.items():
        print(f"      Agent 1 scouting {objective}...", end=" ")

        result = run_agent1_inference(model_name, str(img_path))
        agent1_results[objective] = result

        if result["success"]:
            parsed_report = parse_scout_report(result["raw_response"])
            scout_reports[objective] = parsed_report
            print(f"Done ({parsed_report['operational_status']})")
        else:
            scout_reports[objective] = {
                "vehicle_type": "Error",
                "operational_status": "Error",
                "description": result["raw_response"]
            }
            print(f"ERROR")

    # ===========================================
    # AGENT 2: Tactical decision (text only)
    # ===========================================
    print(f"      Agent 2 making decision...", end=" ")

    agent2_result = run_agent2_inference(model_name, scout_reports)

    scenario_end_time = time.time()
    scenario_time = scenario_end_time - scenario_start_time

    if agent2_result["success"]:
        decision = parse_agent2_decision(agent2_result["raw_response"])
        selected = decision["selected_objective"]
        is_correct = selected == CORRECT_OBJECTIVE
        status = "CORRECT" if is_correct else "WRONG"
        print(f"{selected} [{status}]")
    else:
        decision = {
            "selected_objective": "ERROR",
            "battlefield_summary": agent2_result["raw_response"]
        }
        is_correct = False
        print(f"ERROR")

    print(f"      Scenario time: {scenario_time:.2f}s")

    return {
        "success": True,
        "scenario_num": scenario_num,
        "scenario_time_sec": round(scenario_time, 3),
        "scout_reports": scout_reports,
        "agent1_raw_responses": {k: v["raw_response"] for k, v in agent1_results.items()},
        "agent2_raw_response": agent2_result["raw_response"],
        "agent2_prompt": agent2_result.get("prompt_used", ""),
        "selected_objective": decision["selected_objective"],
        "battlefield_summary": decision["battlefield_summary"],
        "is_correct": is_correct
    }


# =============================================================================
# MODEL EVALUATION
# =============================================================================

def evaluate_model(model_name, scenarios_dir, warmup_image_path):
    """Evaluate a single model on all scenarios"""
    print(f"\n{'='*70}")
    print(f"EVALUATING: {model_name}")
    print(f"{'='*70}")

    # Warmup
    warmup_model(model_name, str(warmup_image_path))

    results = []

    for scenario_num in range(1, SCENARIO_COUNT + 1):
        scenario_path = scenarios_dir / f"Scenario{scenario_num}"

        if not scenario_path.exists():
            print(f"    WARNING: Scenario {scenario_num} folder not found")
            continue

        scenario_result = evaluate_scenario(model_name, scenario_num, scenario_path)
        results.append(scenario_result)

    # Unload model
    print(f"\n  Cleaning up {model_name}...")
    unload_model(model_name)

    return results


# =============================================================================
# GPT GRADING PHASE
# =============================================================================

def run_gpt_grading(all_results):
    """Run GPT grading on all model results"""
    print("\n" + "="*70)
    print("GPT-4o REASONING GRADING")
    print("="*70)

    for model_name, model_data in all_results["models"].items():
        print(f"\n  Grading {model_name}...")

        for scenario_result in model_data["scenario_results"]:
            if not scenario_result.get("success", False):
                scenario_result["gpt_grade"] = {
                    "success": False,
                    "score": 0,
                    "justification": "Scenario failed, could not grade"
                }
                continue

            scenario_num = scenario_result["scenario_num"]
            print(f"    Scenario {scenario_num}...", end=" ")

            grade = grade_reasoning_with_gpt(
                scenario_result["scout_reports"],
                scenario_result["selected_objective"],
                scenario_result["battlefield_summary"],
                scenario_result["selected_objective"]
            )

            scenario_result["gpt_grade"] = grade
            print(f"Score: {grade['score']}/10")

    return all_results


# =============================================================================
# MAIN EVALUATION
# =============================================================================

def run_evaluation():
    """Run full agent evaluation pipeline"""
    print("\n" + "="*70)
    print("AGENT EVALUATION: TWO-AGENT VLM PIPELINE")
    print("="*70)
    print(f"Task: Scout drone reconnaissance + C2 tactical decision")
    print(f"Models: {', '.join(MODELS)}")
    print(f"Scenarios: {SCENARIO_COUNT}")
    print(f"Correct Answer: {CORRECT_OBJECTIVE} (operational MBT)")
    print("="*70)

    # Setup paths
    script_dir = Path(__file__).parent
    scenarios_dir = script_dir / SCENARIOS_FOLDER
    warmup_image_path = script_dir / WARMUP_IMAGE

    # Verify paths
    if not warmup_image_path.exists():
        print(f"ERROR: Warmup image not found: {warmup_image_path}")
        return

    if not scenarios_dir.exists():
        print(f"ERROR: Scenarios folder not found: {scenarios_dir}")
        return

    # Initialize results
    all_results = {
        "evaluation": "AgentEval_TwoAgentPipeline",
        "timestamp": datetime.now().isoformat(),
        "agent1_prompt": AGENT1_PROMPT,
        "agent2_prompt_template": AGENT2_PROMPT,
        "correct_objective": CORRECT_OBJECTIVE,
        "scenario_count": SCENARIO_COUNT,
        "models": {}
    }

    # Evaluate each model
    for model_name in MODELS:
        model_results = evaluate_model(model_name, scenarios_dir, warmup_image_path)

        # Calculate summary statistics
        successful = [r for r in model_results if r.get("success", False)]
        correct = [r for r in successful if r.get("is_correct", False)]
        scenario_times = [r["scenario_time_sec"] for r in successful]

        # Calculate std dev
        if len(scenario_times) > 1:
            mean_time = sum(scenario_times) / len(scenario_times)
            variance = sum((t - mean_time) ** 2 for t in scenario_times) / (len(scenario_times) - 1)
            std_dev = variance ** 0.5
        else:
            mean_time = scenario_times[0] if scenario_times else 0
            std_dev = 0

        all_results["models"][model_name] = {
            "scenario_results": model_results,
            "summary": {
                "total_scenarios": len(model_results),
                "successful_scenarios": len(successful),
                "correct_decisions": len(correct),
                "accuracy_percent": round(len(correct) / len(successful) * 100, 1) if successful else 0,
                "mean_scenario_time_sec": round(mean_time, 3),
                "std_dev_time_sec": round(std_dev, 3),
                "min_scenario_time_sec": round(min(scenario_times), 3) if scenario_times else 0,
                "max_scenario_time_sec": round(max(scenario_times), 3) if scenario_times else 0
            }
        }

    # Run GPT grading
    all_results = run_gpt_grading(all_results)

    # Calculate GPT grade summaries
    for model_name, model_data in all_results["models"].items():
        grades = [
            r["gpt_grade"]["score"]
            for r in model_data["scenario_results"]
            if r.get("gpt_grade", {}).get("success", False)
        ]

        if grades:
            model_data["summary"]["avg_reasoning_score"] = round(sum(grades) / len(grades), 2)
            model_data["summary"]["min_reasoning_score"] = min(grades)
            model_data["summary"]["max_reasoning_score"] = max(grades)
        else:
            model_data["summary"]["avg_reasoning_score"] = 0
            model_data["summary"]["min_reasoning_score"] = 0
            model_data["summary"]["max_reasoning_score"] = 0

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
    print(f"\n{'Model':<15} {'Correct':<10} {'Accuracy':<10} {'Reasoning':<12} {'Mean Time':<12}")
    print("-"*59)

    for model_name in MODELS:
        summary = all_results["models"][model_name]["summary"]
        correct_str = f"{summary['correct_decisions']}/{summary['total_scenarios']}"
        accuracy_str = f"{summary['accuracy_percent']}%"
        reasoning_str = f"{summary['avg_reasoning_score']}/10"
        time_str = f"{summary['mean_scenario_time_sec']}s"
        print(f"{model_name:<15} {correct_str:<10} {accuracy_str:<10} {reasoning_str:<12} {time_str:<12}")

    print(f"\nResults saved to: {output_path}")
    print("="*70)


if __name__ == "__main__":
    run_evaluation()
