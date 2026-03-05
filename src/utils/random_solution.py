"""
Random and greedy solution generators.
"""
import random
from src.models import ProblemInstance, Solution
from src.evaluation import dist


def random_solution(instance: ProblemInstance, seed: int = None) -> Solution:
    """
    Randomly assign rides to vehicles (shuffled, then round-robin).
    Not quality-focused – used as baseline / initial state for metaheuristics.
    """
    if seed is not None:
        random.seed(seed)

    sol = Solution(instance.F)
    ride_ids = list(range(instance.N))
    random.shuffle(ride_ids)

    for i, rid in enumerate(ride_ids):
        v = i % instance.F
        sol.vehicle_rides[v].append(rid)

    return sol


def greedy_solution(instance: ProblemInstance) -> Solution:
    """
    Greedy nearest-feasible-ride assignment.
    Each vehicle at every step picks the best available ride:
      score = ride_distance + bonus_if_on_time  (feasibility first)
    This produces a decent starting point.
    """
    sol = Solution(instance.F)
    unassigned = set(range(instance.N))

    # Vehicle state: (row, col, step)
    state = [(0, 0, 0)] * instance.F

    # We do a simple greedy: for each vehicle, greedily pick rides
    for v in range(instance.F):
        row, col, step = state[v]
        while unassigned:
            best_rid = None
            best_score = -1

            for rid in unassigned:
                ride = instance.rides[rid]
                travel = dist(row, col, ride.a, ride.b)
                arrive = step + travel
                start  = max(arrive, ride.earliest_start)
                finish = start + ride.distance

                if finish > ride.latest_finish:
                    continue   # infeasible

                score = ride.distance
                if start == ride.earliest_start:
                    score += instance.B

                # prefer rides we can start on time and that have good score
                if score > best_score:
                    best_score = score
                    best_rid = rid

            if best_rid is None:
                break  # no feasible ride left for this vehicle

            ride = instance.rides[best_rid]
            travel = dist(row, col, ride.a, ride.b)
            arrive = step + travel
            start  = max(arrive, ride.earliest_start)
            finish = start + ride.distance

            sol.vehicle_rides[v].append(best_rid)
            unassigned.remove(best_rid)
            row, col, step = ride.x, ride.y, finish

    return sol
