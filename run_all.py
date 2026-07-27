# run_all.py
# ---------------------------------------------------------------
# DREAM Framework — Full Pipeline Orchestrator
# Executes all phases in sequence. Aborts on first failure.
# ---------------------------------------------------------------

import subprocess
import sys
import time

steps = [
    # (label, script_path)
    ("Phase 0 — Preprocessing",                  "phase0_dataset/preAnaliseDataset.py"),
    ("Phase 1 — Global Model Selection",          "phase1_global/runPhase1_-_Dream.py"),
    ("Phase 1 — Results Extraction",             "phase1_global/runPhase1_-_Results.py"),
    ("Phase 1 — Analysis",                        "phase1_global/runPhase1_Analysis.py"),
    ("Phase 2 — Local Competence Estimation",     "phase2_competence/runPhase2.py"),
    ("Phase 2 — Results Extraction",             "phase2_competence/runPhase2_results.py"),
    ("Phase 3 — Dynamic Combination",             "phase3_dynamic/runPhase3.py"),
    ("Phase 3 — Baselines",                       "phase3_dynamic/runPhase3_Baseline.py"),
    ("Phase 3 — Boxplots (overall)",              "phase3_statical/01_boxplots_phase3.py"),
    ("Phase 3 — Statistical Tests (overall)",     "phase3_statical/02_statistical_tests_phase3.py"),
    ("Phase 3 — Boxplots Comparison A",           "phase3_statical/03_boxplots_compA.py"),
    ("Phase 3 — Stats Comparison A",              "phase3_statical/04_stats_compA.py"),
    ("Phase 3 — Boxplots Comparison B",           "phase3_statical/05_boxplots_compB.py"),
    ("Phase 3 — Stats Comparison B",              "phase3_statical/06_stats_compB.py"),
    ("Phase 3 — Boxplots Comparison C",           "phase3_statical/07_boxplots_compC_.py"),
    ("Phase 3 — Stats Comparison C",              "phase3_statical/08_stats_compC_.py"),
    ("Phase 3 — Boxplots Comparison D (MINE 10)", "phase3_statical/09_boxplots_compD_mine10.py"),
    ("Phase 3 — Stats Comparison D (MINE 10)",    "phase3_statical/10_stats_compD_mine10.py"),
    ("Phase 3 — Boxplots Comparison D (MINE 100)","phase3_statical/11_boxplots_compD_mine100.py"),
    ("Phase 3 — Stats Comparison D (MINE 100)",   "phase3_statical/12_stats_compD_mine100.py"),
]

def run_pipeline():
    total_start = time.time()
    print("\n" + "=" * 60)
    print("  DREAM Framework — Full Pipeline")
    print("=" * 60)

    for i, (name, script) in enumerate(steps, start=1):
        print(f"\n[{i}/{len(steps)}] {name}")
        print("-" * 60)
        step_start = time.time()

        result = subprocess.run(
            [sys.executable, script],
            check=False
        )

        elapsed = time.time() - step_start

        if result.returncode != 0:
            print(f"\n[ERROR] Step '{name}' failed with return code {result.returncode}.")
            print("Pipeline aborted.")
            sys.exit(result.returncode)

        print(f"[OK] Completed in {elapsed:.1f}s")

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 60)
    print(f"  Pipeline completed successfully in {total_elapsed:.1f}s")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    run_pipeline()
