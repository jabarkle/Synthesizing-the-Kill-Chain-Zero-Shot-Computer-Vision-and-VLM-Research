#!/usr/bin/env python3
"""
Run Agent Evaluations
Executes both agent evaluation pipelines:
  1. End-to-End (E2E): Each model does its own scouting + decision-making
  2. Controlled Input: Qwen 8B scouts for all, each model makes decisions
"""

import time
from datetime import datetime


def run_all():
    print("\n" + "="*70)
    print("RUNNING AGENT EVALUATION PIPELINES")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    total_start = time.time()

    # =========================================================================
    # PIPELINE 1: END-TO-END EVALUATION
    # Each model does its own scouting (Agent 1) + decision-making (Agent 2)
    # =========================================================================
    print("\n\n" + "#"*70)
    print("# PIPELINE 1: END-TO-END (E2E) EVALUATION")
    print("# Each model performs both scouting and decision-making")
    print("#"*70)

    e2e_eval_start = time.time()
    from AgentEval import run_evaluation as run_e2e_eval
    run_e2e_eval()
    e2e_eval_time = time.time() - e2e_eval_start

    print(f"\n>>> E2E Evaluation completed in {e2e_eval_time:.1f}s")

    # Generate E2E Statistics
    print("\n\n" + "#"*70)
    print("# GENERATING E2E STATISTICS")
    print("#"*70)

    e2e_stats_start = time.time()
    from AgentEvalStatistics import run_statistics as run_e2e_stats
    run_e2e_stats()
    e2e_stats_time = time.time() - e2e_stats_start

    print(f"\n>>> E2E Statistics completed in {e2e_stats_time:.1f}s")

    # Generate E2E Scout Reports Markdown
    print("\n\n" + "#"*70)
    print("# GENERATING E2E SCOUT REPORTS MARKDOWN")
    print("#"*70)

    from GenerateScoutReports import run as run_scout_reports
    run_scout_reports()

    # =========================================================================
    # PIPELINE 2: CONTROLLED INPUT EVALUATION
    # Qwen 8B scouts for all, each model only does decision-making (Agent 2)
    # =========================================================================
    print("\n\n" + "#"*70)
    print("# PIPELINE 2: CONTROLLED INPUT EVALUATION")
    print("# Qwen 8B scouts for all, models only make decisions")
    print("#"*70)

    ctrl_eval_start = time.time()
    from AgentEvalControlled import run_evaluation as run_ctrl_eval
    run_ctrl_eval()
    ctrl_eval_time = time.time() - ctrl_eval_start

    print(f"\n>>> Controlled Input Evaluation completed in {ctrl_eval_time:.1f}s")

    # Generate Controlled Statistics
    print("\n\n" + "#"*70)
    print("# GENERATING CONTROLLED INPUT STATISTICS")
    print("#"*70)

    ctrl_stats_start = time.time()
    from AgentEvalControlledStatistics import run_statistics as run_ctrl_stats
    run_ctrl_stats()
    ctrl_stats_time = time.time() - ctrl_stats_start

    print(f"\n>>> Controlled Input Statistics completed in {ctrl_stats_time:.1f}s")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    total_time = time.time() - total_start

    print("\n\n" + "="*70)
    print("ALL AGENT EVALUATIONS COMPLETE")
    print("="*70)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nTiming Summary:")
    print(f"  E2E Evaluation:              {e2e_eval_time:.1f}s")
    print(f"  E2E Statistics:              {e2e_stats_time:.1f}s")
    print(f"  Controlled Evaluation:       {ctrl_eval_time:.1f}s")
    print(f"  Controlled Statistics:       {ctrl_stats_time:.1f}s")
    print(f"  ----------------------------------------")
    print(f"  Total Time:                  {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"\nOutput Files Generated:")
    print(f"\n  End-to-End (E2E) Pipeline:")
    print(f"    - Results/AgentEval_E2E_results.json")
    print(f"    - Results/AgentEval_E2E_results.csv")
    print(f"    - Results/AgentEval_E2E_table.png")
    print(f"    - Results/AgentEval_E2E_appendix.md")
    print(f"    - Results/AgentEval_E2E_scout_reports.md")
    print(f"\n  Controlled Input Pipeline:")
    print(f"    - Results/AgentEval_Controlled_results.json")
    print(f"    - Results/AgentEval_Controlled_results.csv")
    print(f"    - Results/AgentEval_Controlled_table.png")
    print(f"    - Results/AgentEval_Controlled_appendix.md")
    print("="*70)


if __name__ == "__main__":
    run_all()
