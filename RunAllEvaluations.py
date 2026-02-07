#!/usr/bin/env python3
"""
Run All Evaluations
Runs Grounding DINO detection, then executes all VLM evaluations and generates statistics.
"""

import time
from datetime import datetime

def run_all():
    print("\n" + "="*70)
    print("RUNNING FULL EVALUATION PIPELINE")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    total_start = time.time()

    # =========================================================================
    # STAGE 0: Grounding DINO Detection
    # =========================================================================
    print("\n\n" + "#"*70)
    print("# STAGE 0: GROUNDING DINO DETECTION")
    print("#"*70)

    gd_start = time.time()
    from GDDetections import process_all_videos
    process_all_videos()
    gd_time = time.time() - gd_start

    print(f"\n>>> Grounding DINO completed in {gd_time:.1f}s")

    # =========================================================================
    # EVAL 1: False Positive Filtering
    # =========================================================================
    print("\n\n" + "#"*70)
    print("# EVAL 1: FALSE POSITIVE FILTERING")
    print("#"*70)

    eval1_start = time.time()
    from Eval1 import run_evaluation as run_eval1
    run_eval1()
    eval1_time = time.time() - eval1_start

    print(f"\n>>> Eval1 completed in {eval1_time:.1f}s")

    # Generate Eval1 Statistics
    print("\n--- Generating Eval1 Statistics ---")
    from Eval1Statistics import run_statistics as run_eval1_stats
    run_eval1_stats()

    # =========================================================================
    # EVAL 2: Damage Assessment
    # =========================================================================
    print("\n\n" + "#"*70)
    print("# EVAL 2: DAMAGE ASSESSMENT")
    print("#"*70)

    eval2_start = time.time()
    from Eval2 import run_evaluation as run_eval2
    run_eval2()
    eval2_time = time.time() - eval2_start

    print(f"\n>>> Eval2 completed in {eval2_time:.1f}s")

    # Generate Eval2 Statistics
    print("\n--- Generating Eval2 Statistics ---")
    from Eval2Statistics import run_statistics as run_eval2_stats
    run_eval2_stats()

    # =========================================================================
    # EVAL 3: Vehicle Type Classification
    # =========================================================================
    print("\n\n" + "#"*70)
    print("# EVAL 3: VEHICLE TYPE CLASSIFICATION")
    print("#"*70)

    eval3_start = time.time()
    from Eval3 import run_evaluation as run_eval3
    run_eval3()
    eval3_time = time.time() - eval3_start

    print(f"\n>>> Eval3 completed in {eval3_time:.1f}s")

    # Generate Eval3 Statistics
    print("\n--- Generating Eval3 Statistics ---")
    from Eval3Statistics import run_statistics as run_eval3_stats
    run_eval3_stats()

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    total_time = time.time() - total_start

    print("\n\n" + "="*70)
    print("ALL EVALUATIONS COMPLETE")
    print("="*70)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nTiming Summary:")
    print(f"  Grounding DINO Detection:             {gd_time:.1f}s")
    print(f"  Eval1 (False Positive Filtering):     {eval1_time:.1f}s")
    print(f"  Eval2 (Damage Assessment):            {eval2_time:.1f}s")
    print(f"  Eval3 (Vehicle Classification):       {eval3_time:.1f}s")
    print(f"  ----------------------------------------")
    print(f"  Total Time:                           {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"\nOutput Files Generated:")
    print(f"  - output/ folder: Highest confidence frame PNGs")
    print(f"  - Results/ folder: Eval JSON, PNG tables, CSV exports")
    print("="*70)


if __name__ == "__main__":
    run_all()
