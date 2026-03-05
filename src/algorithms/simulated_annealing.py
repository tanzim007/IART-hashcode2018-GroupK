"""
Simulated Annealing.

Accepts worse solutions with probability exp(-(delta)/T) where T is
the current temperature, which cools according to a geometric schedule.
"""
import math
import random
from typing import List, Tuple

from src.models import ProblemInstance, Solution
from src.evaluation import score_solution
from src.utils.neighbors import random_neighbor
from src.utils.random_solution import greedy_solution


def simulated_annealing(
    instance: ProblemInstance,
    max_iterations: int = 5000,
    initial_temp: float = 500.0,
    cooling_rate: float = 0.995,
    min_temp: float = 0.1,
    seed: int = 42,
) -> Tuple[Solution, int, List[int]]:
    """
    Run Simulated Annealing.

    Returns
    -------
    best_solution   : Solution
    best_score      : int
    score_history   : list[int]   – best score seen so far at each iteration
    """
    random.seed(seed)

    current = greedy_solution(instance)
    current_score = score_solution(instance, current)

    best_solution = current.clone()
    best_score = current_score

    temp = initial_temp
    history: List[int] = [best_score]

    for iteration in range(max_iterations):
        if temp < min_temp:
            break

        neighbour = random_neighbor(current)
        nb_score  = score_solution(instance, neighbour)
        delta     = nb_score - current_score

        # Accept if better, or probabilistically if worse
        if delta > 0 or random.random() < math.exp(delta / temp):
            current = neighbour
            current_score = nb_score

        if current_score > best_score:
            best_score = current_score
            best_solution = current.clone()

        temp *= cooling_rate
        history.append(best_score)

    return best_solution, best_score, history
