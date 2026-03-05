"""
Parser and writer for Hash Code 2018 input/output files.
"""
import os
from src.models import Ride, ProblemInstance, Solution


def parse_input(filepath: str) -> ProblemInstance:
    """
    Parse a .in file and return a ProblemInstance.

    Header:  R C F N B T
    N lines: a b x y s f
    """
    with open(filepath, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    R, C, F, N, B, T = map(int, lines[0].split())
    filename = os.path.basename(filepath)
    instance = ProblemInstance(R=R, C=C, F=F, N=N, B=B, T=T, filename=filename)

    for i in range(1, N + 1):
        a, b, x, y, s, fin = map(int, lines[i].split())
        instance.rides.append(Ride(
            ride_id=i - 1,
            a=a, b=b, x=x, y=y,
            earliest_start=s,
            latest_finish=fin
        ))

    return instance


def write_solution(filepath: str, solution: Solution) -> None:
    """
    Write solution in Hash Code submission format.
    Each line: M R0 R1 ... RM-1
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        for rides in solution.vehicle_rides:
            parts = [str(len(rides))] + list(map(str, rides))
            f.write(" ".join(parts) + "\n")
