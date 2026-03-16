"""
Statistical Analysis Functions for IPD

This module provides comprehensive statistical analysis tools for examining
tournament results, strategy performance, and cooperation dynamics.

Author: Momchil Ivanov
Course: Math Concepts for Developers
Date: March 2026
"""

import numpy as np
import pandas as pd


def calculate_strategy_statistics(tournament):
    """
    Calculate comprehensive statistics for each strategy.
    
    Parameters:
    -----------
    tournament : Tournament
        Tournament object with completed results
        
    Returns:
    --------
    DataFrame with metrics including:
        - Win rate, losses, ties
        - Average scores and variance
        - Cooperation rates
        - Performance consistency
    """
    stats = []
    
    for strategy in tournament.strategies:
        name = strategy.name
        
        strategy_results = [r for r in tournament.results 
                           if r['strategy1'] == name or r['strategy2'] == name]
        
        wins = 0
        losses = 0
        ties = 0
        total_score = 0
        coop_rates = []
        scores = []
        
        for result in strategy_results:
            if result['strategy1'] == name:
                my_score = result['score1']
                opp_score = result['score2']
                coop_rates.append(result['cooperation_rate1'])
            else:
                my_score = result['score2']
                opp_score = result['score1']
                coop_rates.append(result['cooperation_rate2'])
            
            scores.append(my_score)
            total_score += my_score
            
            if my_score > opp_score:
                wins += 1
            elif my_score < opp_score:
                losses += 1
            else:
                ties += 1
        
        total_matches = len(strategy_results)
        
        stats.append({
            'Strategy': name,
            'Wins': wins,
            'Losses': losses,
            'Ties': ties,
            'Win Rate': wins / total_matches if total_matches > 0 else 0,
            'Avg Score': total_score / total_matches if total_matches > 0 else 0,
            'Score StdDev': np.std(scores),
            'Avg Cooperation': np.mean(coop_rates)
        })
    
    df = pd.DataFrame(stats)
    df = df.sort_values('Avg Score', ascending=False)
    df = df.reset_index(drop=True)
    return df


def compute_cooperation_correlation(stats_df):
    """
    Compute correlation between cooperation rate and tournament success.
    
    Parameters:
    -----------
    stats_df : DataFrame
        Statistics from calculate_strategy_statistics()
        
    Returns:
    --------
    dict with correlation coefficient and interpretation
    """
    correlation = stats_df['Avg Cooperation'].corr(stats_df['Avg Score'])
    
    if correlation > 0.3:
        interpretation = "Positive correlation: More cooperative strategies tend to score higher"
    elif correlation < -0.3:
        interpretation = "Negative correlation: More defecting strategies tend to score higher"
    else:
        interpretation = "Weak correlation: No clear relationship between cooperation and success"
    
    return {
        'correlation': correlation,
        'interpretation': interpretation
    }


def compare_tournament_evolution(tournament, evolution):
    """
    Compare tournament rankings with evolutionary success.
    
    Parameters:
    -----------
    tournament : Tournament
        Completed tournament object
    evolution : EvolutionarySimulation
        Completed evolutionary simulation
        
    Returns:
    --------
    dict with comparison metrics
    """
    rankings = tournament.get_rankings()
    final_dist = evolution.get_final_distribution()
    
    top_5_tournament = set(rankings.head(5)['Strategy'].values)
    top_5_evolution = set(final_dist.head(5)['Strategy'].values)
    
    common_winners = top_5_tournament & top_5_evolution
    tournament_only = top_5_tournament - top_5_evolution
    evolution_only = top_5_evolution - top_5_tournament
    
    return {
        'top_tournament': list(top_5_tournament),
        'top_evolution': list(top_5_evolution),
        'common_winners': list(common_winners),
        'tournament_only': list(tournament_only),
        'evolution_only': list(evolution_only),
        'overlap_count': len(common_winners)
    }


def analyze_strategy_robustness(tournament):
    """
    Analyze how consistently each strategy performs across different opponents.
    
    Lower variance = more robust performance
    Higher variance = performance depends heavily on opponent
    
    Parameters:
    -----------
    tournament : Tournament
        Completed tournament object
        
    Returns:
    --------
    DataFrame with robustness metrics
    """
    robustness = []
    
    for strategy in tournament.strategies:
        name = strategy.name
        
        scores = []
        for result in tournament.results:
            if result['strategy1'] == name:
                scores.append(result['avg_score1'])
            elif result['strategy2'] == name:
                scores.append(result['avg_score2'])
        
        robustness.append({
            'Strategy': name,
            'Mean Score': np.mean(scores),
            'Std Dev': np.std(scores),
            'Min Score': np.min(scores),
            'Max Score': np.max(scores),
            'Score Range': np.max(scores) - np.min(scores),
            'Coefficient of Variation': np.std(scores) / np.mean(scores) if np.mean(scores) > 0 else 0
        })
    
    df = pd.DataFrame(robustness)
    df = df.sort_values('Coefficient of Variation', ascending=True)
    df = df.reset_index(drop=True)
    return df


def identify_dominant_strategies(matchup_matrix):
    """
    Identify strategies that dominate others (always score higher).
    
    Parameters:
    -----------
    matchup_matrix : DataFrame
        Head-to-head matchup matrix
        
    Returns:
    --------
    dict with domination relationships
    """
    strategies = matchup_matrix.index.tolist()
    dominations = {}
    
    for i, strategy_i in enumerate(strategies):
        dominated_by_i = []
        
        for j, strategy_j in enumerate(strategies):
            if i != j:
                if matchup_matrix.iloc[i, j] > matchup_matrix.iloc[j, i]:
                    dominated_by_i.append(strategy_j)
        
        if dominated_by_i:
            dominations[strategy_i] = dominated_by_i
    
    return dominations


def calculate_evolutionary_stability_index(evolution):
    """
    Calculate stability index for evolutionary simulation.
    
    Measures how much the population distribution changes over final generations.
    Lower values = more stable equilibrium
    
    Parameters:
    -----------
    evolution : EvolutionarySimulation
        Completed evolutionary simulation
        
    Returns:
    --------
    float : Stability index (average change in last 20% of generations)
    """
    history = np.array(evolution.history)
    
    cutoff = int(len(history) * 0.8)
    final_history = history[cutoff:]
    
    changes = []
    for i in range(1, len(final_history)):
        change = np.sum(np.abs(final_history[i] - final_history[i-1]))
        changes.append(change)
    
    stability_index = np.mean(changes)
    
    return stability_index
