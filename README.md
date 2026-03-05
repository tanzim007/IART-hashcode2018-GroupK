# Hash Code 2018 – Self-Driving Rides — Metaheuristics Optimizer

Solves the Google Hash Code 2018 Qualification Round problem using four metaheuristic algorithms:

- **Hill Climbing** (steepest ascent with restarts)
- **Simulated Annealing**
- **Tabu Search**
- **Genetic Algorithm**

---

## Project Structure

```
hashcode2018/
├── data/                    ← problem input files (.in)
│   ├── a_example.in
│   ├── b_small.in
│   ├── c_medium.in
│   └── d_large.in
├── results/                 ← output solutions + JSON metadata + plots
├── src/
│   ├── models.py            ← Ride, ProblemInstance, Solution
│   ├── parser.py            ← file I/O
│   ├── evaluation.py        ← scoring function
│   ├── experiment.py        ← runner + result storage
│   ├── visualization.py     ← matplotlib plots
│   ├── algorithms/
│   │   ├── hill_climbing.py
│   │   ├── simulated_annealing.py
│   │   ├── tabu_search.py
│   │   └── genetic_algorithm.py
│   └── utils/
│       ├── neighbors.py     ← move/swap/reverse operators
│       └── random_solution.py
└── main.py                  ← interactive CLI entry point
```

---

## Requirements

```
Python >= 3.9
matplotlib
numpy
```

Install:
```bash
pip install matplotlib numpy
```

---

## Running

```bash
cd hashcode2018
python main.py
```

### Menu options

| Option | Action |
|--------|--------|
| 1 | Run algorithm(s) on a chosen instance |
| 2 | View / regenerate plots from saved results |
| 3 | Compare all algorithms across all instances |
| 4 | Quick sanity test (example input, expected score = 10) |
| 0 | Exit |

---

## Adding a custom problem instance

Place any `.in` file in `data/`.  Format:
```
R C F N B T
a b x y s f     ← one line per ride
...
```

---

## Algorithm parameters (configurable via menu option 1)

### Hill Climbing
| Parameter | Default | Meaning |
|-----------|---------|---------|
| max_iterations | 2000 | Total iterations across all restarts |
| neighbours_per_step | 5 | Neighbours evaluated per iteration |
| max_no_improve | 300 | Early stop if no improvement |
| restarts | 3 | Number of random restarts |

### Simulated Annealing
| Parameter | Default | Meaning |
|-----------|---------|---------|
| max_iterations | 10000 | Max iterations |
| initial_temp | 500.0 | Starting temperature |
| cooling_rate | 0.995 | Geometric cooling factor |
| min_temp | 0.1 | Stop when temp drops below this |

### Tabu Search
| Parameter | Default | Meaning |
|-----------|---------|---------|
| max_iterations | 4000 | Max iterations |
| tabu_tenure | 20 | How long moves stay tabu |
| neighbours_per_step | 10 | Candidates per iteration |

### Genetic Algorithm
| Parameter | Default | Meaning |
|-----------|---------|---------|
| population_size | 20 | Individuals per generation |
| generations | 200 | Number of generations |
| mutation_rate | 0.3 | Probability of mutation per child |
| elite_size | 2 | Number of elites preserved |
| tournament_k | 3 | Tournament selection size |

---

## Output files (in `results/`)

| File | Description |
|------|-------------|
| `<instance>_<algo>.out` | Solution in Hash Code submission format |
| `<instance>_<algo>.json` | Score, time, history, params |
| `<instance>_comparison.png` | Evolution curves + bar chart |
| `<instance>_time_vs_score.png` | Runtime vs score scatter |
| `<instance>_<algo>_evolution.png` | Per-algorithm score evolution |
| `multi_instance_comparison.png` | All algorithms × all instances |
