"""
Tabu Search.

Maintains a tabu list of recently visited solution fingerprints to
prevent cycling. At each step the best non-tabu neighbour is accepted
even if it is worse than the current solution.

Fingerprint = tuple of vehicle ride lists (hashable, lightweight).
"""
import random
from collections import deque
from typing import List, Tuple

from src.models import ProblemInstance, Solution
from src.evaluation import score_solution
from src.utils.neighbors import random_neighbor
from src.utils.random_solution import greedy_solution


def _fingerprint(sol: Solution) -> tuple:
    return tuple(tuple(v) for v in sol.vehicle_rides)


def tabu_search(
    instance: ProblemInstance,
    max_iterations: int = 2000,
    tabu_tenure: int = 20,
    neighbours_per_step: int = 10,
    seed: int = 42,
) -> Tuple[Solution, int, List[int]]:
    """
    Run Tabu Search.

    Returns
    -------
    best_solution : Solution
    best_score    : int
    score_history : list[int]
    """
    random.seed(seed)

    current = greedy_solution(instance)
    current_score = score_solution(instance, current)

    best_solution = current.clone()
    best_score = current_score

    tabu: deque = deque(maxlen=tabu_tenure)
    tabu.append(_fingerprint(current))

    history: List[int] = [best_score]

    for _ in range(max_iterations):
        neighbours = [random_neighbor(current) for _ in range(neighbours_per_step)]

        best_nb = None
        best_nb_score = -1

        for nb in neighbours:
            fp = _fingerprint(nb)
            nb_score = score_solution(instance, nb)

            # Aspiration criterion: accept tabu move if it beats global best
            if fp in tabu and nb_score <= best_score:
                continue

            if nb_score > best_nb_score:
                best_nb_score = nb_score
                best_nb = nb
                best_nb_fp = fp

        if best_nb is None:
            # All neighbours are tabu – take the best anyway
            best_nb = max(neighbours, key=lambda n: score_solution(instance, n))
            best_nb_score = score_solution(instance, best_nb)
            best_nb_fp = _fingerprint(best_nb)

        current = best_nb
        current_score = best_nb_score
        tabu.append(best_nb_fp)

        if current_score > best_score:
            best_score = current_score
            best_solution = current.clone()

        history.append(best_score)

    return best_solution, best_score, history
