"""
Data models for Hash Code 2018 – Self-Driving Rides.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class Ride:
    """One pre-booked ride."""
    ride_id: int
    a: int        # start row
    b: int        # start col
    x: int        # finish row
    y: int        # finish col
    earliest_start: int
    latest_finish: int

    @property
    def distance(self) -> int:
        return abs(self.x - self.a) + abs(self.y - self.b)

    def __repr__(self):
        return (f"Ride({self.ride_id}: [{self.a},{self.b}]->[{self.x},{self.y}] "
                f"dist={self.distance} es={self.earliest_start} lf={self.latest_finish})")


@dataclass
class ProblemInstance:
    """All data for one problem file."""
    R: int          # rows
    C: int          # columns
    F: int          # vehicles
    N: int          # rides
    B: int          # on-time bonus
    T: int          # total steps
    filename: str = ""
    rides: List[Ride] = field(default_factory=list)

    def __repr__(self):
        return (f"ProblemInstance({self.filename} grid={self.R}x{self.C} "
                f"vehicles={self.F} rides={self.N} bonus={self.B} steps={self.T})")


class Solution:
    """
    Assignment of rides to vehicles.
    vehicle_rides[v] = ordered list of ride IDs for vehicle v.
    """

    def __init__(self, num_vehicles: int):
        self.vehicle_rides: List[List[int]] = [[] for _ in range(num_vehicles)]

    def clone(self) -> "Solution":
        s = Solution(len(self.vehicle_rides))
        s.vehicle_rides = [list(r) for r in self.vehicle_rides]
        return s

    def all_assigned_rides(self) -> List[int]:
        return [r for vr in self.vehicle_rides for r in vr]

    def __repr__(self):
        n = sum(len(v) for v in self.vehicle_rides)
        return f"Solution(vehicles={len(self.vehicle_rides)}, assigned={n})"
