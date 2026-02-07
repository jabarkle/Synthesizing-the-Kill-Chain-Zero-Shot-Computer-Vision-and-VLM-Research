#!/usr/bin/env python3
"""
Clear All Outputs
Deletes all generated files from output/ and Results/ folders for a fresh run.
"""

import shutil
from pathlib import Path


def clear_outputs():
    print("\n" + "="*70)
    print("CLEARING ALL OUTPUTS")
    print("="*70)

    script_dir = Path(__file__).parent
    output_dir = script_dir / "output"
    results_dir = script_dir / "Results"

    files_deleted = 0
    folders_cleared = 0

    # Clear output/ folder (Grounding DINO frames)
    if output_dir.exists():
        print("\nClearing output/ folder...")
        for item in output_dir.iterdir():
            if item.is_dir():
                # Clear contents of subdirectories but keep the folders
                for file in item.iterdir():
                    file.unlink()
                    files_deleted += 1
                print(f"  Cleared: {item.name}/")
                folders_cleared += 1
            elif item.is_file():
                item.unlink()
                files_deleted += 1
                print(f"  Deleted: {item.name}")
    else:
        print("\noutput/ folder does not exist, skipping...")

    # Clear Results/ folder (Eval outputs)
    if results_dir.exists():
        print("\nClearing Results/ folder...")
        for item in results_dir.iterdir():
            if item.is_file():
                item.unlink()
                files_deleted += 1
                print(f"  Deleted: {item.name}")
    else:
        print("\nResults/ folder does not exist, skipping...")

    # Summary
    print("\n" + "="*70)
    print("CLEANUP COMPLETE")
    print("="*70)
    print(f"Files deleted: {files_deleted}")
    print(f"Folders cleared: {folders_cleared}")
    print("\nReady for fresh run: python RunAllEvaluations.py")
    print("="*70)


if __name__ == "__main__":
    clear_outputs()
