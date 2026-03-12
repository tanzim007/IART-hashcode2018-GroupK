"""
Hybrid Genetic Algorithm + Tabu Search.

Motivated by Glover, Laguna & Martí (1993):
- GA provides global diversity through order-preserving crossover.
- Tabu Search replaces standard mutation, providing aggressive local refinement.
"""
import random
from collections import deque
from typing import List, Tuple

from src.models import ProblemInstance, Solution
from src.evaluation import score_solution
from src.utils.neighbors import random_neighbor
from src.utils.random_solution import greedy_solution, random_solution


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

    all_rides = set(range(instance.N))
    missing = list(all_rides - assigned)
    random.shuffle(missing)

    for rid in missing:
        v = random.randrange(n)
        child.vehicle_rides[v].append(rid)

    return child


def _tournament(population: List[Solution], scores: List[int],
                k: int = 3) -> Solution:
    """Tournament selection for parent picking."""
    candidates = random.sample(range(len(population)), min(k, len(population)))
    winner = max(candidates, key=lambda i: scores[i])
    return population[winner].clone()


# ── TS Components (Smart Mutation) ───────────────────────────────────────────

def _fingerprint(sol: Solution) -> tuple:
    """Hashable fingerprint for the Tabu memory."""
    return tuple(tuple(v) for v in sol.vehicle_rides)


def _local_tabu_search(
        instance: ProblemInstance,
        initial_solution: Solution,
        initial_score: int,
        iterations: int = 15,
        tabu_tenure: int = 5,
        neighbours_per_step: int = 5
) -> Tuple[Solution, int]:
    """
    A short, aggressive Tabu Search used to refine a child solution.
    """
    current = initial_solution.clone()
    current_score = initial_score

    best_solution = current.clone()
    best_score = current_score

    tabu: deque = deque(maxlen=tabu_tenure)
    tabu.append(_fingerprint(current))

    for _ in range(iterations):
        neighbours = [random_neighbor(current) for _ in range(neighbours_per_step)]

        best_nb = None
        best_nb_score = -1
        best_nb_fp = None

        for nb in neighbours:
            fp = _fingerprint(nb)
            nb_score = score_solution(instance, nb)

            # Aspiration criterion: accept if it beats the historical best
            if fp in tabu and nb_score <= best_score:
                continue

            if nb_score > best_nb_score:
                best_nb_score = nb_score
                best_nb = nb
                best_nb_fp = fp

        if best_nb is None:
            # Fallback if all are tabu
            best_nb = max(neighbours, key=lambda n: score_solution(instance, n))
            best_nb_score = score_solution(instance, best_nb)
            best_nb_fp = _fingerprint(best_nb)

        current = best_nb
        current_score = best_nb_score
        tabu.append(best_nb_fp)

        if current_score > best_score:
            best_score = current_score
            best_solution = current.clone()

    return best_solution, best_score


# ── Main Hybrid Algorithm ────────────────────────────────────────────────────

def hybrid_ga_ts(
        instance: ProblemInstance,
        population_size: int = 10,
        generations: int = 50,
        elite_size: int = 2,
        tournament_k: int = 3,
        ts_iterations: int = 15,
        ts_tenure: int = 5,
        ts_neighbors: int = 5,
        seed: int = 42,
) -> Tuple[Solution, int, List[int]]:
    """
    Run Hybrid GA + Tabu Search.
    """
    random.seed(seed)

    # Initialise population
    population: List[Solution] = [greedy_solution(instance)]
    while len(population) < population_size:
        population.append(random_solution(instance))

    scores = [score_solution(instance, ind) for ind in population]
    best_idx = max(range(population_size), key=lambda i: scores[i])
    best_solution = population[best_idx].clone()
    best_score = scores[best_idx]

    history: List[int] = [best_score]

    for gen in range(generations):
        # Elitism
        ranked = sorted(range(len(population)), key=lambda i: scores[i], reverse=True)
        new_population: List[Solution] = [population[i].clone() for i in ranked[:elite_size]]

        # Generate offspring
        while len(new_population) < population_size:
            p1 = _tournament(population, scores, tournament_k)
            p2 = _tournament(population, scores, tournament_k)

            # 1. Global Exploration: Crossover
            child = _crossover(p1, p2, instance)
            child_score = score_solution(instance, child)

            # 2. Local Exploitation: Tabu Search as smart mutation
            child, _ = _local_tabu_search(
                instance,
                child,
                child_score,
                iterations=ts_iterations,
                tabu_tenure=ts_tenure,
                neighbours_per_step=ts_neighbors
            )

            new_population.append(child)

        population = new_population
        scores = [score_solution(instance, ind) for ind in population]

        # Update global best
        gen_best_idx = max(range(len(scores)), key=lambda i: scores[i])
        if scores[gen_best_idx] > best_score:
            best_score = scores[gen_best_idx]
            best_solution = population[gen_best_idx].clone()

        history.append(best_score)

    return best_solution, best_score, history