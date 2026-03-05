"""
Genetic Algorithm.

Each individual is a Solution.
Crossover: for each vehicle, randomly pick which parent to inherit from.
Mutation : apply a random neighbourhood operator.
Selection: tournament selection.
"""
import random
from typing import List, Tuple

from src.models import ProblemInstance, Solution
from src.evaluation import score_solution
from src.utils.neighbors import random_neighbor, move_ride
from src.utils.random_solution import greedy_solution, random_solution


# ── crossover ────────────────────────────────────────────────────────────────

def _crossover(parent1: Solution, parent2: Solution,
               instance: ProblemInstance) -> Solution:
    """
    Order-preserving crossover.
    For each vehicle, pick the ride list from parent1 or parent2.
    Then fix duplicate / missing rides.
    """
    n = len(parent1.vehicle_rides)
    child = Solution(n)

    assigned: set = set()

    for v in range(n):
        src = parent1 if random.random() < 0.5 else parent2
        new_rides = [r for r in src.vehicle_rides[v] if r not in assigned]
        child.vehicle_rides[v] = new_rides
        assigned.update(new_rides)

    # Collect unassigned rides and distribute them
    all_rides = set(range(instance.N))
    missing = list(all_rides - assigned)
    random.shuffle(missing)

    for rid in missing:
        v = random.randrange(n)
        child.vehicle_rides[v].append(rid)

    return child


# ── selection ────────────────────────────────────────────────────────────────

def _tournament(population: List[Solution], scores: List[int],
                k: int = 3) -> Solution:
    candidates = random.sample(range(len(population)), min(k, len(population)))
    winner = max(candidates, key=lambda i: scores[i])
    return population[winner].clone()


# ── main ─────────────────────────────────────────────────────────────────────

def genetic_algorithm(
    instance: ProblemInstance,
    population_size: int = 20,
    generations: int = 100,
    mutation_rate: float = 0.3,
    elite_size: int = 2,
    tournament_k: int = 3,
    seed: int = 42,
) -> Tuple[Solution, int, List[int]]:
    """
    Run a Genetic Algorithm.

    Returns
    -------
    best_solution : Solution
    best_score    : int
    score_history : list[int]  – best score per generation
    """
    random.seed(seed)

    # Initialise: one greedy + rest random
    population: List[Solution] = [greedy_solution(instance)]
    while len(population) < population_size:
        population.append(random_solution(instance))

    scores = [score_solution(instance, ind) for ind in population]
    best_idx = max(range(population_size), key=lambda i: scores[i])
    best_solution = population[best_idx].clone()
    best_score = scores[best_idx]

    history: List[int] = [best_score]

    for _ in range(generations):
        # Sort by score descending for elitism
        ranked = sorted(range(len(population)), key=lambda i: scores[i], reverse=True)
        elites = [population[i].clone() for i in ranked[:elite_size]]

        new_population: List[Solution] = elites[:]

        while len(new_population) < population_size:
            p1 = _tournament(population, scores, tournament_k)
            p2 = _tournament(population, scores, tournament_k)
            child = _crossover(p1, p2, instance)

            if random.random() < mutation_rate:
                child = random_neighbor(child)

            new_population.append(child)

        population = new_population
        scores = [score_solution(instance, ind) for ind in population]

        gen_best_idx = max(range(len(scores)), key=lambda i: scores[i])
        if scores[gen_best_idx] > best_score:
            best_score = scores[gen_best_idx]
            best_solution = population[gen_best_idx].clone()

        history.append(best_score)

    return best_solution, best_score, history
