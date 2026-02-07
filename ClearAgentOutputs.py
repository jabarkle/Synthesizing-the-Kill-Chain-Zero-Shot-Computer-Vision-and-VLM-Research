#!/usr/bin/env python3
"""
Clear Agent Outputs
Deletes all agent evaluation output files from Results/ folder for a fresh run.
Does NOT affect the original evaluation outputs (Eval1, Eval2, Eval3) or input images.
"""

from pathlib import Path


def clear_agent_outputs():
    print("\n" + "="*70)
    print("CLEARING AGENT EVALUATION OUTPUTS")
    print("="*70)

    script_dir = Path(__file__).parent
    results_dir = script_dir / "Results"

    # Agent-specific output files (both E2E and Controlled pipelines)
    agent_files = [
        # End-to-End (E2E) pipeline outputs
        "AgentEval_E2E_results.json",
        "AgentEval_E2E_results.csv",
        "AgentEval_E2E_table.png",
        "AgentEval_E2E_appendix.md",
        "AgentEval_E2E_scout_reports.md",
        # Controlled Input pipeline outputs
        "AgentEval_Controlled_results.json",
        "AgentEval_Controlled_results.csv",
        "AgentEval_Controlled_table.png",
        "AgentEval_Controlled_appendix.md",
    ]

    files_deleted = 0

    if results_dir.exists():
        print("\nClearing agent evaluation files from Results/ folder...")

        for filename in agent_files:
            filepath = results_dir / filename
            if filepath.exists():
                filepath.unlink()
                files_deleted += 1
                print(f"  Deleted: {filename}")
            else:
                print(f"  Not found (skipping): {filename}")
    else:
        print("\nResults/ folder does not exist, nothing to clear.")

    # Summary
    print("\n" + "="*70)
    print("CLEANUP COMPLETE")
    print("="*70)
    print(f"Files deleted: {files_deleted}")
    print(f"\nNote: Original evaluation files (Eval1, Eval2, Eval3) were NOT affected.")
    print(f"Note: Input images in AgentTest/ were NOT affected.")
    print("\nReady for fresh run: python RunAgentEvaluations.py")
    print("="*70)


if __name__ == "__main__":
    clear_agent_outputs()
