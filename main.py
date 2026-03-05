"""
Hash Code 2018 – Self-Driving Rides  |  Metaheuristics Optimizer
================================================================
Interactive CLI.  Run:  python main.py
"""
import os
import sys
import json
import glob

# Ensure project root is on path when running from any directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.parser import parse_input
from src.experiment import run_experiments, DEFAULT_PARAMS, ALGORITHM_FNS
from src.visualization import (
    plot_evolution,
    plot_comparison,
    plot_time_vs_score,
    plot_multi_instance,
)


DATA_DIR    = os.path.join(os.path.dirname(__file__), "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


# ── helpers ──────────────────────────────────────────────────────────────────

def list_instances():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.in")))
    if not files:
        print("  No .in files found in data/")
        return []
    for i, f in enumerate(files):
        print(f"  [{i}] {os.path.basename(f)}")
    return files


def pick_instance(files):
    idx = input("Select instance number: ").strip()
    try:
        return files[int(idx)]
    except (ValueError, IndexError):
        print("  Invalid selection.")
        return None


def edit_params(algo_name: str) -> dict:
    defaults = DEFAULT_PARAMS[algo_name]
    print(f"\n  Default params for {algo_name}:")
    for k, v in defaults.items():
        print(f"    {k} = {v}")

    custom = {}
    print("  Enter new values (press Enter to keep default):")
    for k, v in defaults.items():
        raw = input(f"    {k} [{v}]: ").strip()
        if raw:
            try:
                custom[k] = type(v)(raw)
            except ValueError:
                print(f"    Invalid, keeping {v}")
    return custom


def load_summary(instance_name: str) -> dict:
    tag = instance_name.replace(".in", "")
    results = {}
    for algo in ALGORITHM_FNS:
        path = os.path.join(RESULTS_DIR, f"{tag}_{algo}.json")
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            results[algo] = {"score": data["score"], "time": data["time_seconds"],
                             "history": data["history"]}
    return results


# ── menu screens ─────────────────────────────────────────────────────────────

def menu_run():
    print("\n─── Run Algorithm(s) ───────────────────────────")
    files = list_instances()
    if not files:
        return
    path = pick_instance(files)
    if not path:
        return

    instance = parse_input(path)
    print(f"\n  Loaded: {instance}")

    print("\n  Select algorithms to run (comma-separated numbers, or 'all'):")
    algo_list = list(ALGORITHM_FNS.keys())
    for i, a in enumerate(algo_list):
        print(f"  [{i}] {a}")

    choice = input("  Choice [all]: ").strip()
    if choice.lower() == "all" or choice == "":
        selected = algo_list
    else:
        try:
            selected = [algo_list[int(x)] for x in choice.split(",")]
        except (ValueError, IndexError):
            print("  Invalid. Running all.")
            selected = algo_list

    # Optional param customisation
    custom_params: dict = {}
    customise = input("\n  Customise parameters? (y/N): ").strip().lower()
    if customise == "y":
        for algo in selected:
            custom_params[algo] = edit_params(algo)

    seed_raw = input("\n  Random seed [42]: ").strip()
    seed = int(seed_raw) if seed_raw.isdigit() else 42

    print(f"\n  Running on: {instance.filename}")
    summary = run_experiments(
        instance,
        results_dir=RESULTS_DIR,
        params=custom_params,
        algorithms=selected,
        seed=seed,
        verbose=True,
    )

    print("\n  ── Results ──────────────────────────")
    for algo, data in summary.items():
        print(f"  {algo:<22}  score={data['score']:>8,}  time={data['time']:.2f}s")

    # Auto-save comparison plot
    tag = instance.filename.replace(".in", "")
    plot_comparison(summary, instance.filename,
                    save_path=os.path.join(RESULTS_DIR, f"{tag}_comparison.png"))
    plot_time_vs_score(summary, instance.filename,
                       save_path=os.path.join(RESULTS_DIR, f"{tag}_time_vs_score.png"))
    for algo, data in summary.items():
        plot_evolution(data["history"], algo, instance.filename,
                       save_path=os.path.join(RESULTS_DIR, f"{tag}_{algo}_evolution.png"))
    print("\n  Plots saved to results/")


def menu_view_results():
    print("\n─── View Saved Results ─────────────────────────")
    files = list_instances()
    if not files:
        return
    path = pick_instance(files)
    if not path:
        return

    instance_name = os.path.basename(path)
    summary = load_summary(instance_name)

    if not summary:
        print("  No saved results found. Run algorithms first.")
        return

    print("\n  ── Saved Results ──")
    for algo, data in summary.items():
        print(f"  {algo:<22}  score={data['score']:>8,}  time={data['time']:.2f}s")

    tag = instance_name.replace(".in", "")
    plot_comparison(summary, instance_name,
                    save_path=os.path.join(RESULTS_DIR, f"{tag}_comparison.png"))
    plot_time_vs_score(summary, instance_name,
                       save_path=os.path.join(RESULTS_DIR, f"{tag}_time_vs_score.png"))
    print("  Plots updated.")


def menu_multi_instance():
    print("\n─── Compare Across All Instances ───────────────")
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.in")))
    if not files:
        print("  No instances found.")
        return

    all_results = {}
    for f in files:
        name = os.path.basename(f)
        sr = load_summary(name)
        if sr:
            all_results[name.replace(".in", "")] = sr

    if not all_results:
        print("  No saved results found. Run algorithms first.")
        return

    save_path = os.path.join(RESULTS_DIR, "multi_instance_comparison.png")
    plot_multi_instance(all_results, save_path=save_path)
    print(f"  Saved → {save_path}")


def menu_test():
    """Quick sanity check using the example instance."""
    print("\n─── Running sanity test on a_example.in ────────")
    path = os.path.join(DATA_DIR, "a_example.in")
    if not os.path.exists(path):
        print("  data/a_example.in not found!")
        return

    from src.models import Solution
    from src.evaluation import score_solution

    instance = parse_input(path)
    sol = Solution(instance.F)
    sol.vehicle_rides[0] = [0]
    sol.vehicle_rides[1] = [2, 1]
    score = score_solution(instance, sol)
    assert score == 10, f"Expected 10, got {score}"
    print(f"  Example solution score = {score}  ✅ (expected 10)")

    # Quick run of all algorithms
    summary = run_experiments(instance, results_dir=RESULTS_DIR, verbose=True)
    print("\n  All algorithms finished.")
    for algo, d in summary.items():
        print(f"  {algo:<22} score={d['score']}")


# ── main loop ────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  Hash Code 2018 – Self-Driving Rides Optimizer")
    print("=" * 55)

    while True:
        print("""
  [1] Run algorithm(s) on an instance
  [2] View / plot saved results
  [3] Compare across all instances
  [4] Run sanity test (example instance)
  [0] Exit
""")
        choice = input("  Choice: ").strip()

        if choice == "1":
            menu_run()
        elif choice == "2":
            menu_view_results()
        elif choice == "3":
            menu_multi_instance()
        elif choice == "4":
            menu_test()
        elif choice == "0":
            print("  Bye!")
            break
        else:
            print("  Unknown choice.")


if __name__ == "__main__":
    main()
