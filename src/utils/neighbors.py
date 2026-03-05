"""
Neighborhood operators for solution mutation.
All operators return a new Solution without modifying the original.
"""
import random
from src.models import ProblemInstance, Solution


# ── helpers ─────────────────────────────────────────────────────────────────

def _pick_two_vehicles(n_vehicles: int):
    v1 = random.randrange(n_vehicles)
    v2 = random.randrange(n_vehicles)
    while v2 == v1 and n_vehicles > 1:
        v2 = random.randrange(n_vehicles)
    return v1, v2


# ── operators ────────────────────────────────────────────────────────────────

def move_ride(sol: Solution) -> Solution:
    """
    Move a random ride from one vehicle to a random position in another vehicle.
    """
    n = len(sol.vehicle_rides)
    non_empty = [v for v in range(n) if sol.vehicle_rides[v]]
    if not non_empty:
        return sol.clone()

    v1 = random.choice(non_empty)
    v2 = random.randrange(n)

    new_sol = sol.clone()
    if not new_sol.vehicle_rides[v1]:
        return new_sol

    idx = random.randrange(len(new_sol.vehicle_rides[v1]))
    ride = new_sol.vehicle_rides[v1].pop(idx)

    insert_pos = random.randint(0, len(new_sol.vehicle_rides[v2]))
    new_sol.vehicle_rides[v2].insert(insert_pos, ride)

    return new_sol


def swap_rides(sol: Solution) -> Solution:
    """
    Swap two rides between two different vehicles.
    """
    n = len(sol.vehicle_rides)
    non_empty = [v for v in range(n) if sol.vehicle_rides[v]]
    if len(non_empty) < 2:
        return sol.clone()

    v1, v2 = random.sample(non_empty, 2)
    new_sol = sol.clone()

    if not new_sol.vehicle_rides[v1] or not new_sol.vehicle_rides[v2]:
        return new_sol

    i1 = random.randrange(len(new_sol.vehicle_rides[v1]))
    i2 = random.randrange(len(new_sol.vehicle_rides[v2]))

    new_sol.vehicle_rides[v1][i1], new_sol.vehicle_rides[v2][i2] = (
        new_sol.vehicle_rides[v2][i2], new_sol.vehicle_rides[v1][i1]
    )
    return new_sol


def reverse_segment(sol: Solution) -> Solution:
    """
    Reverse a random sub-sequence of rides within one vehicle (2-opt style).
    """
    n = len(sol.vehicle_rides)
    non_empty = [v for v in range(n) if len(sol.vehicle_rides[v]) >= 2]
    if not non_empty:
        return sol.clone()

    v = random.choice(non_empty)
    new_sol = sol.clone()
    rides = new_sol.vehicle_rides[v]

    i = random.randrange(len(rides) - 1)
    j = random.randint(i + 1, len(rides) - 1)
    rides[i:j+1] = rides[i:j+1][::-1]

    return new_sol


def random_neighbor(sol: Solution) -> Solution:
    """Randomly pick one of the three operators."""
    op = random.choice([move_ride, swap_rides, reverse_segment])
    return op(sol)
