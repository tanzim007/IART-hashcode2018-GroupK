"""
Hill Climbing – steepest ascent with random restarts.

At each iteration we generate `neighbours_per_step` neighbours and
accept the best one if it improves the current score.
Stops when no improvement found for `max_no_improve` consecutive steps
or the time/iteration budget is exhausted.
"""
import time
import random
from typing import List, Tuple

from src.models import ProblemInstance, Solution
from src.evaluation import score_solution
from src.utils.neighbors import random_neighbor
from src.utils.random_solution import greedy_solution, random_solution


def hill_climbing(
    instance: ProblemInstance,
    max_iterations: int = 1000,
    neighbours_per_step: int = 5,
    max_no_improve: int = 200,
    restarts: int = 3,
    seed: int = 42,
) -> Tuple[Solution, int, List[int]]:
    """
    Run Hill Climbing on the given instance.

    Returns
    -------
    best_solution   : Solution
    best_score      : int
    score_history   : list of scores recorded each iteration (all restarts)
    """
    random.seed(seed)
    history: List[int] = []

    best_solution = greedy_solution(instance)
    best_score = score_solution(instance, best_solution)

    for restart in range(restarts):
        # first restart uses greedy; subsequent ones use random
        if restart == 0:
            current = best_solution.clone()
        else:
            current = random_solution(instance)
        current_score = score_solution(instance, current)

        no_improve = 0

        for _ in range(max_iterations // restarts):
            # generate neighbourhood
            neighbours = [random_neighbor(current) for _ in range(neighbours_per_step)]
            neighbour_scores = [score_solution(instance, n) for n in neighbours]

            best_nb_idx = max(range(len(neighbour_scores)), key=lambda i: neighbour_scores[i])
            best_nb_score = neighbour_scores[best_nb_idx]

            history.append(current_score)

            if best_nb_score > current_score:
                current = neighbours[best_nb_idx]
                current_score = best_nb_score
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= max_no_improve:
                break

        if current_score > best_score:
            best_score = current_score
            best_solution = current.clone()

    return best_solution, best_score, history
