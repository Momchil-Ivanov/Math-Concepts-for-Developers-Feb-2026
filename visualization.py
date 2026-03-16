"""
Visualization Functions for IPD Analysis

This module provides plotting functions for visualizing tournament results,
evolutionary dynamics, and statistical analyses.

Author: Momchil Ivanov
Course: Math Concepts for Developers
Date: March 2026
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['font.size'] = 10


def plot_tournament_rankings(rankings_df, title="Tournament Rankings"):
    """
    Visualize tournament rankings as horizontal bar chart.
    
    Parameters:
    -----------
    rankings_df : DataFrame
        Rankings from tournament.get_rankings()
    title : str
        Plot title
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    strategies = rankings_df['Strategy'].values
    scores = rankings_df['Average Score'].values
    
    colors = sns.color_palette("viridis", len(strategies))
    bars = ax.barh(strategies, scores, color=colors)
    
    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax.text(score + 0.05, bar.get_y() + bar.get_height()/2, 
                f'{score:.2f}', va='center', fontsize=9)
    
    ax.set_xlabel('Average Score per Match', fontsize=12, fontweight='bold')
    ax.set_ylabel('Strategy', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.show()


def plot_matchup_heatmap(matchup_matrix, title="Strategy Matchup Heatmap"):
    """
    Visualize head-to-head matchup scores as heatmap.
    
    Parameters:
    -----------
    matchup_matrix : DataFrame
        Matrix from tournament.get_matchup_matrix()
    title : str
        Plot title
    """
    fig, ax = plt.subplots(figsize=(14, 12))
    
    sns.heatmap(matchup_matrix, annot=True, fmt='.2f', cmap='RdYlGn', 
                center=2.0, vmin=0, vmax=5, cbar_kws={'label': 'Average Score'},
                linewidths=0.5, ax=ax)
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Opponent Strategy', fontsize=12, fontweight='bold')
    ax.set_ylabel('Player Strategy', fontsize=12, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_cooperation_rates(tournament_results, title="Cooperation Rates by Strategy"):
    """
    Visualize how much each strategy cooperates.
    
    Parameters:
    -----------
    tournament_results : list
        Results from tournament.run_tournament()
    title : str
        Plot title
    """
    coop_rates = {}
    
    for result in tournament_results:
        s1, s2 = result['strategy1'], result['strategy2']
        
        if s1 not in coop_rates:
            coop_rates[s1] = []
        if s2 not in coop_rates:
            coop_rates[s2] = []
        
        coop_rates[s1].append(result['cooperation_rate1'])
        coop_rates[s2].append(result['cooperation_rate2'])
    
    avg_coop = {strategy: np.mean(rates) for strategy, rates in coop_rates.items()}
    
    sorted_strategies = sorted(avg_coop.items(), key=lambda x: x[1], reverse=True)
    strategies, rates = zip(*sorted_strategies)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = sns.color_palette("coolwarm", len(strategies))
    bars = ax.barh(strategies, rates, color=colors)
    
    for bar, rate in zip(bars, rates):
        ax.text(rate + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{rate*100:.1f}%', va='center', fontsize=9)
    
    ax.set_xlabel('Average Cooperation Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('Strategy', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim(0, 1.1)
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.show()


def plot_score_vs_cooperation(tournament, title="Score vs Cooperation Rate"):
    """
    Scatter plot showing relationship between cooperation and success.
    
    Parameters:
    -----------
    tournament : Tournament
        Tournament object with completed results
    title : str
        Plot title
    """
    coop_rates = {}
    for result in tournament.results:
        s1, s2 = result['strategy1'], result['strategy2']
        
        if s1 not in coop_rates:
            coop_rates[s1] = []
        if s2 not in coop_rates:
            coop_rates[s2] = []
        
        coop_rates[s1].append(result['cooperation_rate1'])
        coop_rates[s2].append(result['cooperation_rate2'])
    
    avg_coop = {strategy: np.mean(rates) for strategy, rates in coop_rates.items()}
    
    rankings = tournament.get_rankings()
    
    x_data = [avg_coop[row['Strategy']] for _, row in rankings.iterrows()]
    y_data = rankings['Average Score'].values
    labels = rankings['Strategy'].values
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    scatter = ax.scatter(x_data, y_data, s=200, alpha=0.6, 
                         c=range(len(x_data)), cmap='viridis', 
                         edgecolors='black', linewidth=1.5)
    
    for x, y, label in zip(x_data, y_data, labels):
        ax.annotate(label, (x, y), xytext=(5, 5), textcoords='offset points', 
                    fontsize=9, alpha=0.8)
    
    ax.set_xlabel('Average Cooperation Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Score per Match', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_evolutionary_dynamics(simulation, top_n=8, title="Evolutionary Dynamics Over Time", ylim=None):
    """
    Plot population proportions over generations.
    
    Parameters:
    -----------
    simulation : EvolutionarySimulation
        Simulation object with history
    top_n : int
        Number of top strategies to display (by final proportion)
    title : str
        Plot title
    """
    history = np.array(simulation.history)
    generations = len(history)
    
    final_proportions = history[-1]
    top_indices = np.argsort(final_proportions)[-top_n:][::-1]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for idx in top_indices:
        strategy_name = simulation.strategies[idx].name
        proportions = history[:, idx]
        ax.plot(range(generations), proportions, linewidth=2.5, 
                label=f"{strategy_name}", marker='o', markersize=3, 
                markevery=max(1, generations//20))
    
    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim(0, 1)
    
    ax.set_xlabel('Generation', fontsize=12, fontweight='bold')
    ax.set_ylabel('Population Proportion', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_tournament_vs_evolution(rankings_df, final_dist_df, title="Tournament Success vs Evolutionary Fitness"):
    """
    Compare tournament rankings with evolutionary success.
    
    Parameters:
    -----------
    rankings_df : DataFrame
        Tournament rankings
    final_dist_df : DataFrame
        Final evolutionary distribution
    title : str
        Plot title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    tournament_data = rankings_df.head(10)
    ax1.barh(tournament_data['Strategy'], tournament_data['Average Score'], 
             color=sns.color_palette("viridis", 10))
    ax1.set_xlabel('Average Score', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Strategy', fontsize=11, fontweight='bold')
    ax1.set_title('Tournament Rankings (Top 10)', fontsize=13, fontweight='bold')
    ax1.invert_yaxis()
    
    evolution_data = final_dist_df.head(10)
    ax2.barh(evolution_data['Strategy'], evolution_data['Final Percentage'], 
             color=sns.color_palette("plasma", 10))
    ax2.set_xlabel('Final Population %', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Strategy', fontsize=11, fontweight='bold')
    ax2.set_title('Evolutionary Winners (Top 10)', fontsize=13, fontweight='bold')
    ax2.invert_yaxis()
    
    plt.suptitle(title, fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()
