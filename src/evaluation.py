"""
Scoring / evaluation function for Hash Code 2018.
"""
from src.models import ProblemInstance, Solution


def dist(r1: int, c1: int, r2: int, c2: int) -> int:
    """Manhattan distance."""
    return abs(r2 - r1) + abs(c2 - c1)


def score_solution(instance: ProblemInstance, solution: Solution) -> int:
    """
    Simulate all vehicles and compute the total score.

    Points per ride:
      +ride.distance  if finished <= latest_finish
      +instance.B     if started exactly at earliest_start
    """
    total = 0

    for ride_ids in solution.vehicle_rides:
        row, col = 0, 0
        step = 0

        for rid in ride_ids:
            ride = instance.rides[rid]

            travel = dist(row, col, ride.a, ride.b)
            arrive = step + travel
            start  = max(arrive, ride.earliest_start)
            finish = start + ride.distance

            if finish <= ride.latest_finish:
                total += ride.distance
                if start == ride.earliest_start:
                    total += instance.B

            # advance vehicle (even for failed rides)
            row, col = ride.x, ride.y
            step = finish

    return total


def score_vehicle(instance: ProblemInstance, ride_ids: list,
                  start_row: int = 0, start_col: int = 0,
                  start_step: int = 0) -> int:
    """Score a single vehicle's ride sequence (useful for incremental evaluation)."""
    total = 0
    row, col, step = start_row, start_col, start_step

    for rid in ride_ids:
        ride = instance.rides[rid]
        arrive = step + dist(row, col, ride.a, ride.b)
        start  = max(arrive, ride.earliest_start)
        finish = start + ride.distance

        if finish <= ride.latest_finish:
            total += ride.distance
            if start == ride.earliest_start:
                total += instance.B

        row, col, step = ride.x, ride.y, finish

    return total
