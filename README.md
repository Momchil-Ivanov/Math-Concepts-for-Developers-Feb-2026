# Iterated Prisoner's Dilemma: Game Theory Analysis & Tournament

A comprehensive mathematical analysis of cooperation and competition in the Iterated Prisoner's Dilemma through game theory, strategy tournaments, and evolutionary simulations.

## Overview

This project explores how cooperation can emerge in competitive environments through repeated interactions. Using mathematical game theory and computational simulations, we analyze 19 different strategies competing in tournaments and evolving over generations.

## Key Questions

1. Which strategies succeed in round-robin tournaments?
2. What is the optimal balance between cooperation and retaliation?
3. How does cooperation evolve in populations over time?
4. Why does iteration fundamentally change rational behavior?

## Project Structure

- `main_notebook.ipynb` - Primary analysis with theory, code, and results
- `ipd/` - Python package (Iterated Prisoner's Dilemma; import e.g. `from ipd import ...` or `from ipd.strategies import ...`)
  - `payoffs.py` - PD parameters (T, R, P, S) and validation
  - `strategies.py` - Strategy class implementations
  - `tournament.py` - Tournament engine and match system
  - `analysis.py` - Statistical analysis functions
  - `visualization.py` - Plotting and visualization functions
- `tests/` - Pytest suite (`test_payoffs.py`, `test_game_noise.py`)
- `pytest.ini` - Pytest config (project root; sets `pythonpath` and `testpaths`)
- `requirements.txt` - Python dependencies

## Requirements

- **Python:** 3.14.0 (tested with this version only)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
jupyter notebook main_notebook.ipynb
```

## Testing

From the project root (same folder as `ipd/` and `pytest.ini`):

```bash
python -m pytest tests/ -v
```

- `tests/test_payoffs.py` — PD ordering constraints, `PayoffParams` validation, payoff matrices
- `tests/test_game_noise.py` — Trembling-hand noise on `Game` (`noise=0` vs `noise=1.0`)

`pytest` is listed in `requirements.txt`.

## Extensions (notebook)

Beyond the baseline tournament and evolutionary analysis, `main_notebook.ipynb` also covers:

- **Section 5.5** — Round-robin tournaments with **trembling-hand noise** (execution error after each intended move). Implemented via `noise` on `Game`, `Tournament`, and `EvolutionarySimulation` in `ipd/tournament.py`, with rankings compared across noise levels.
- **Section 5.6** — **Payoff parameter sweep** (e.g. varying temptation `T` through `PayoffParams`) with plots.

Section 6 ties these extensions back to the main conclusions.

## Mathematical Concepts Covered

- Game theory fundamentals and Nash equilibria
- Prisoner's Dilemma formulation and proofs
- Iterated games and the Folk theorem
- Evolutionary game theory and replicator dynamics
- Evolutionarily stable strategies (ESS)

## Strategies Implemented

1. Always Cooperate
2. Always Defect
3. Random
4. Tit-for-Tat
5. Tit-for-Two-Tats
6. Grim Trigger
7. Pavlov
8. Generous Tit-for-Tat
9. Gradual
10. Suspicious Tit-for-Tat
11. Prober
12. Adaptive
13. Hard Tit-for-Tat
14. Soft Majority
15. Hard Majority
16. Reverse Tit-for-Tat
17. Omega Tit-for-Tat
18. Tester
19. Firm But Fair

## Key References

See Section 4 in `main_notebook.ipynb` for the full reference list with links.

## Author

Final exam project for Math Concepts for Developers course, March 2026.

## Note on Git History

For the **first project submission**, a **force push removed the commit history** on GitHub, so that version no longer showed an incremental record of the work—only the state of the files at that time.

This repo is the **retake**. Its branch has **10 meaningful commits**, each covering a clear unit of work (implementation, tests, documentation), so Git shows how the project was built—unlike the first submission, where that history was lost to the force push.

## License

Educational purposes only.
