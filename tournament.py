"""
Tournament and Evolutionary Simulation Engine for IPD

This module implements the core tournament infrastructure:
- Game: Single match between two strategies
- Tournament: Round-robin competition with multiple strategies
- EvolutionarySimulation: Population dynamics using replicator dynamics

Author: Momchil Ivanov
Course: Math Concepts for Developers
Date: March 2026
"""

import numpy as np
import pandas as pd
from itertools import combinations
from strategies import Strategy


class Game:
    """
    Manages a single IPD match between two strategies.
    
    Uses standard Prisoner's Dilemma payoffs:
    - T (Temptation): 5 - payoff for defecting when opponent cooperates
    - R (Reward): 3 - payoff for mutual cooperation
    - P (Pdefection
    - S (Sucker): 0 - payoff for cunishment): 1 - payoff for mutual ooperating when opponent defects
    
    These satisfy the PD ordering: T > R > P > S and 2R > T + S
    """
    
    def __init__(self, strategy1, strategy2, rounds=200):
        self.strategy1 = strategy1
        self.strategy2 = strategy2
        self.rounds = rounds
        
        self.payoffs = {
            ('C', 'C'): (3, 3),
            ('C', 'D'): (0, 5),
            ('D', 'C'): (5, 0),
            ('D', 'D'): (1, 1)
        }
        
        self.history1 = []
        self.history2 = []
        self.score1 = 0
        self.score2 = 0
    
    def play_round(self):
        action1 = self.strategy1.choose_action(self.history1, self.history2)
        action2 = self.strategy2.choose_action(self.history2, self.history1)
        
        self.history1.append(action1)
        self.history2.append(action2)
        
        payoff1, payoff2 = self.payoffs[(action1, action2)]
        self.score1 += payoff1
        self.score2 += payoff2
        
        return action1, action2, payoff1, payoff2
    
    def play_match(self):
        self.strategy1.reset()
        self.strategy2.reset()
        
        self.history1 = []
        self.history2 = []
        self.score1 = 0
        self.score2 = 0
        
        for _ in range(self.rounds):
            self.play_round()
        
        return {
            'strategy1': self.strategy1.name,
            'strategy2': self.strategy2.name,
            'score1': self.score1,
            'score2': self.score2,
            'avg_score1': self.score1 / self.rounds,
            'avg_score2': self.score2 / self.rounds,
            'cooperation_rate1': self.history1.count('C') / self.rounds,
            'cooperation_rate2': self.history2.count('C') / self.rounds,
            'history1': self.history1,
            'history2': self.history2
        }


class Tournament:
    """
    Orchestrates round-robin IPD tournament between multiple strategies.
    
    Each strategy plays against every other strategy (and itself) for a 
    specified number of rounds. Results are aggregated to determine rankings.
    """
    
    def __init__(self, strategies, rounds_per_match=200):
        self.strategies = strategies
        self.rounds_per_match = rounds_per_match
        self.results = []
        self.scores = {strategy.name: 0 for strategy in strategies}
        self.matches_played = {strategy.name: 0 for strategy in strategies}
    
    def run_tournament(self):
        """
        Run complete round-robin tournament.
        
        Each strategy plays against every other strategy exactly once,
        and also plays against itself.
        """
        print(f"Starting tournament with {len(self.strategies)} strategies...")
        print(f"Rounds per match: {self.rounds_per_match}")
        print(f"Total matches: {len(self.strategies) * (len(self.strategies) + 1) // 2}\n")
        
        match_count = 0
        
        for i, strategy1 in enumerate(self.strategies):
            for j, strategy2 in enumerate(self.strategies):
                if j < i:
                    continue
                
                match_count += 1
                game = Game(strategy1, strategy2, self.rounds_per_match)
                result = game.play_match()
                
                self.results.append(result)
                self.scores[strategy1.name] += result['score1']
                self.scores[strategy2.name] += result['score2']
                self.matches_played[strategy1.name] += 1
                self.matches_played[strategy2.name] += 1
                
                if match_count % 10 == 0:
                    print(f"Completed {match_count} matches...")
        
        print(f"\nTournament complete! {match_count} matches played.\n")
        return self.results
    
    def get_rankings(self):
        rankings = []
        
        for strategy in self.strategies:
            name = strategy.name
            total_score = self.scores[name]
            matches = self.matches_played[name]
            avg_score = total_score / matches if matches > 0 else 0
            
            rankings.append({
                'Strategy': name,
                'Total Score': total_score,
                'Matches Played': matches,
                'Average Score': avg_score
            })
        
        rankings_df = pd.DataFrame(rankings)
        rankings_df = rankings_df.sort_values('Average Score', ascending=False)
        rankings_df = rankings_df.reset_index(drop=True)
        rankings_df.index = rankings_df.index + 1
        
        return rankings_df
    
    def get_matchup_matrix(self):
        """
        Create matrix showing scores for each pairwise matchup.
        
        Returns a DataFrame where entry [i,j] is strategy i's score against strategy j.
        """
        n = len(self.strategies)
        strategy_names = [s.name for s in self.strategies]
        
        matrix = np.zeros((n, n))
        
        for result in self.results:
            idx1 = strategy_names.index(result['strategy1'])
            idx2 = strategy_names.index(result['strategy2'])
            
            matrix[idx1, idx2] = result['avg_score1']
            
            if idx1 != idx2:
                matrix[idx2, idx1] = result['avg_score2']
        
        return pd.DataFrame(matrix, index=strategy_names, columns=strategy_names)


class EvolutionarySimulation:
    """
    Simulates evolutionary dynamics of IPD strategies over time.
    
    Uses replicator dynamics: strategies with higher fitness (payoff) increase
    in frequency, while lower-fitness strategies decrease.
    """
    
    def __init__(self, strategies, rounds_per_encounter=50, initial_population=None):
        self.strategies = strategies
        self.rounds_per_encounter = rounds_per_encounter
        
        if initial_population is None:
            n = len(strategies)
            self.population = np.ones(n) / n
        else:
            self.population = np.array(initial_population)
            self.population = self.population / self.population.sum()
        
        self.history = [self.population.copy()]
        self.fitness_matrix = None
    
    def compute_fitness_matrix(self):
        """
        Compute average payoffs for all strategy pairings.
        
        Returns n×n matrix where entry [i,j] is average payoff for 
        strategy i when playing against strategy j.
        """
        n = len(self.strategies)
        matrix = np.zeros((n, n))
        
        for i, strategy1 in enumerate(self.strategies):
            for j, strategy2 in enumerate(self.strategies):
                game = Game(strategy1, strategy2, self.rounds_per_encounter)
                result = game.play_match()
                matrix[i, j] = result['avg_score1']
        
        self.fitness_matrix = matrix
        return matrix
    
    def compute_fitness(self, population=None):
        """
        Compute fitness of each strategy given current population distribution.
        
        Fitness = expected payoff against a random opponent from the population.
        """
        if population is None:
            population = self.population
        
        if self.fitness_matrix is None:
            self.compute_fitness_matrix()
        
        fitness = self.fitness_matrix @ population
        return fitness
    
    def evolve_generation(self):
        """
        Evolve population by one generation using replicator dynamics.
        
        The replicator equation:
        ṗᵢ = pᵢ(fᵢ - f̄)
        
        Where pᵢ is frequency of strategy i, fᵢ is its fitness, 
        and f̄ is average population fitness.
        """
        fitness = self.compute_fitness()
        avg_fitness = np.dot(self.population, fitness)
        
        new_population = self.population * fitness / avg_fitness
        new_population = new_population / new_population.sum()
        
        new_population[new_population < 1e-6] = 0
        new_population = new_population / new_population.sum()
        
        self.population = new_population
        self.history.append(self.population.copy())
        
        return new_population, fitness, avg_fitness
    
    def run_simulation(self, generations=100, verbose=True):
        if verbose:
            print(f"Running evolutionary simulation for {generations} generations...\n")
        
        self.compute_fitness_matrix()
        
        for gen in range(generations):
            population, fitness, avg_fitness = self.evolve_generation()
            
            if verbose and (gen + 1) % 20 == 0:
                print(f"Generation {gen + 1:3d}: Avg Fitness = {avg_fitness:.3f}")
                top_indices = np.argsort(population)[-3:][::-1]
                for idx in top_indices:
                    if population[idx] > 0.01:
                        print(f"  {self.strategies[idx].name:25s}: {population[idx]:.1%}")
                print()
        
        if verbose:
            print("Simulation complete!\n")
        
        return self.history
    
    def get_final_distribution(self):
        results = []
        for i, strategy in enumerate(self.strategies):
            results.append({
                'Strategy': strategy.name,
                'Final Proportion': self.population[i],
                'Final Percentage': self.population[i] * 100
            })
        
        df = pd.DataFrame(results)
        df = df.sort_values('Final Proportion', ascending=False)
        df = df.reset_index(drop=True)
        return df
