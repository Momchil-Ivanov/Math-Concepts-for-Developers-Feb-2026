# Iterated Prisoner's Dilemma: Game Theory Analysis & Tournament

A comprehensive mathematical analysis of cooperation and competition in the Iterated Prisoner's Dilemma through game theory, strategy tournaments, and evolutionary simulations.

## Overview

This project explores how cooperation can emerge in competitive environments through repeated interactions. Using mathematical game theory and computational simulations, we analyze 15 different strategies competing in tournaments and evolving over generations.

## Key Questions

1. Which strategies succeed in round-robin tournaments?
2. What is the optimal balance between cooperation and retaliation?
3. How does cooperation evolve in populations over time?
4. Why does iteration fundamentally change rational behavior?

## Project Structure

- `main_notebook.ipynb` - Primary analysis with theory, code, and results
- `strategies.py` - Strategy class implementations
- `tournament.py` - Tournament engine and match system
- `analysis.py` - Statistical analysis functions
- `visualization.py` - Plotting and visualization functions
- `requirements.txt` - Python dependencies

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
jupyter notebook main_notebook.ipynb
```

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
4. Tit-for-Tat (Axelrod's winner)
5. Tit-for-Two-Tats
6. Grim Trigger
7. Pavlov (Win-Stay, Lose-Shift)
8. Generous Tit-for-Tat
9. Gradual
10. Suspicious Tit-for-Tat
11. Prober
12. Adaptive
13. Hard Tit-for-Tat
14. Soft Majority
15. Hard Majority

## Key References

See Section 4 in `main_notebook.ipynb` for the full reference list with links.

## Author

Final exam project for Math Concepts for Developers course, March 2026.

## License

Educational purposes only.
