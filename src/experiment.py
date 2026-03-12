"""
Experiment runner.

Runs all four algorithms on a given problem instance,
saves per-run results to results/ and returns summary data.
"""
import os
import time
import json
from typing import Dict, Any

from src.models import ProblemInstance
from src.parser import write_solution
from src.evaluation import score_solution
from src.algorithms.hill_climbing import hill_climbing
from src.algorithms.simulated_annealing import simulated_annealing
from src.algorithms.tabu_search import tabu_search
from src.algorithms.genetic_algorithm import genetic_algorithm
from src.algorithms.hybrid_ga_ts import hybrid_ga_ts


# ── default parameter sets ───────────────────────────────────────────────────

DEFAULT_PARAMS = {
    "hill_climbing": {
        "max_iterations": 2000,
        "neighbours_per_step": 5,
        "max_no_improve": 300,
        "restarts": 3,
    },
    "simulated_annealing": {
        "max_iterations": 10000,
        "initial_temp": 500.0,
        "cooling_rate": 0.995,
        "min_temp": 0.1,
    },
    "tabu_search": {
        "max_iterations": 4000,
        "tabu_tenure": 20,
        "neighbours_per_step": 10,
    },
    "genetic_algorithm": {
        "population_size": 20,
        "generations": 200,
        "mutation_rate": 0.3,
        "elite_size": 2,
        "tournament_k": 3,
    },
    "hybrid_ga_ts": {
        "population_size": 10,
        "generations": 50,
        "elite_size": 2,
        "tournament_k": 3,
        "ts_iterations": 15,
        "ts_tenure": 5,
        "ts_neighbors": 5,
    }
}

ALGORITHM_FNS = {
    "hill_climbing": hill_climbing,
    "simulated_annealing": simulated_annealing,
    "tabu_search": tabu_search,
    "genetic_algorithm": genetic_algorithm,
    "hybrid_ga_ts": hybrid_ga_ts,
}


def run_experiments(
    instance: ProblemInstance,
    results_dir: str = "results",
    params: Dict[str, Dict] = None,
    algorithms: list = None,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run selected algorithms on the instance.

    Parameters
    ----------
    instance     : ProblemInstance
    results_dir  : folder to write output files
    params       : dict of per-algorithm parameter overrides
    algorithms   : list of algorithm names to run (default: all four)
    seed         : random seed
    verbose      : print progress

    Returns
    -------
    summary dict  { algo_name: {score, time, history} }
    """
    if params is None:
        params = {}
    if algorithms is None:
        algorithms = list(ALGORITHM_FNS.keys())

    os.makedirs(results_dir, exist_ok=True)
    summary: Dict[str, Any] = {}

    for algo_name in algorithms:
        if algo_name not in ALGORITHM_FNS:
            print(f"  [WARN] Unknown algorithm: {algo_name}")
            continue

        # Merge default params with any overrides
        algo_params = {**DEFAULT_PARAMS[algo_name], **params.get(algo_name, {})}
        algo_params["seed"] = seed

        if verbose:
            print(f"\n  Running {algo_name}  params={algo_params}")

        fn = ALGORITHM_FNS[algo_name]
        t0 = time.perf_counter()
        best_sol, best_score, history = fn(instance, **algo_params)
        elapsed = time.perf_counter() - t0

        if verbose:
            print(f"    Score={best_score}  Time={elapsed:.2f}s  Iters={len(history)}")

        # Save solution file
        instance_tag = instance.filename.replace(".in", "")
        sol_path = os.path.join(results_dir, f"{instance_tag}_{algo_name}.out")
        write_solution(sol_path, best_sol)

        # Save metadata
        meta = {
            "algorithm": algo_name,
            "instance": instance.filename,
            "score": best_score,
            "time_seconds": round(elapsed, 4),
            "iterations": len(history),
            "params": algo_params,
            "history": history,
        }
        meta_path = os.path.join(results_dir, f"{instance_tag}_{algo_name}.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        summary[algo_name] = {
            "score": best_score,
            "time": elapsed,
            "history": history,
            "solution": best_sol,
        }

    # Save cross-algorithm summary
    summary_path = os.path.join(results_dir, f"{instance.filename.replace('.in','')}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            {k: {"score": v["score"], "time": round(v["time"], 4)} for k, v in summary.items()},
            f, indent=2
        )

    return summary
