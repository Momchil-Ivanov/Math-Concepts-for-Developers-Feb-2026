"""
Iterated Prisoner's Dilemma Strategy Implementations

This module contains 14 strategy classes for playing the Iterated Prisoner's Dilemma.
Each strategy inherits from the abstract base Strategy class and implements a 
choose_action() method that returns 'C' (cooperate) or 'D' (defect).

Author: Momchil Ivanov
Course: Math Concepts for Developers
Date: March 2026
"""

from abc import ABC, abstractmethod
import random
import numpy as np


class Strategy(ABC):
    """
    Base class for all IPD strategies.
    
    Each strategy must implement choose_action() which returns 'C' (cooperate) 
    or 'D' (defect) based on the history of previous moves.
    """
    
    def __init__(self, name):
        self.name = name
    
    @abstractmethod
    def choose_action(self, my_history, opponent_history):
        """
        Choose an action based on game history.
        
        Parameters:
        -----------
        my_history : list of str
            History of this strategy's moves ['C', 'D', 'C', ...]
        opponent_history : list of str
            History of opponent's moves ['D', 'C', 'D', ...]
            
        Returns:
        --------
        str : 'C' for cooperate, 'D' for defect
        """
        pass
    
    def reset(self):
        """Reset any internal state (for strategies with memory)"""
        pass
    
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return f"{self.__class__.__name__}('{self.name}')"


class AlwaysCooperate(Strategy):
    """Always cooperates, regardless of opponent's actions."""
    
    def __init__(self):
        super().__init__("Always Cooperate")
    
    def choose_action(self, my_history, opponent_history):
        return 'C'


class AlwaysDefect(Strategy):
    """Always defects, regardless of opponent's actions."""
    
    def __init__(self):
        super().__init__("Always Defect")
    
    def choose_action(self, my_history, opponent_history):
        return 'D'


class Random(Strategy):
    """Randomly cooperates or defects with equal probability."""
    
    def __init__(self, cooperation_probability=0.5):
        super().__init__("Random")
        self.p_cooperate = cooperation_probability
    
    def choose_action(self, my_history, opponent_history):
        return 'C' if random.random() < self.p_cooperate else 'D'


class TitForTat(Strategy):
    """
    Cooperates on first move, then copies opponent's previous move.
    
    This is Anatol Rapoport's famous strategy that won Axelrod's tournaments.
    Simple, nice, retaliatory, and forgiving.
    """
    
    def __init__(self):
        super().__init__("Tit-for-Tat")
    
    def choose_action(self, my_history, opponent_history):
        if len(opponent_history) == 0:
            return 'C'
        return opponent_history[-1]


class TitForTwoTats(Strategy):
    """
    More forgiving variant: only retaliates after two consecutive defections.
    
    Cooperates on first move, then defects only if opponent defected 
    on both of the last two moves.
    """
    
    def __init__(self):
        super().__init__("Tit-for-Two-Tats")
    
    def choose_action(self, my_history, opponent_history):
        if len(opponent_history) < 2:
            return 'C'
        
        if opponent_history[-1] == 'D' and opponent_history[-2] == 'D':
            return 'D'
        return 'C'


class Grim(Strategy):
    """
    Grim Trigger: cooperates until opponent defects once, then defects forever.
    
    Also known as 'Trigger Strategy'. Extremely unforgiving - implements
    permanent retaliation for any betrayal.
    """
    
    def __init__(self):
        super().__init__("Grim Trigger")
        self.triggered = False
    
    def choose_action(self, my_history, opponent_history):
        if self.triggered:
            return 'D'
        
        if 'D' in opponent_history:
            self.triggered = True
            return 'D'
        
        return 'C'
    
    def reset(self):
        self.triggered = False


class Pavlov(Strategy):
    """
    Win-Stay, Lose-Shift strategy.
    
    Cooperates if both players chose the same action last round (mutual C or mutual D).
    Defects if players chose different actions (one cooperated, one defected).
    
    This implements a simple reinforcement learning rule: repeat what worked, 
    change what didn't.
    """
    
    def __init__(self):
        super().__init__("Pavlov")
    
    def choose_action(self, my_history, opponent_history):
        if len(my_history) == 0:
            return 'C'
        
        if my_history[-1] == opponent_history[-1]:
            return my_history[-1]
        else:
            return 'D' if my_history[-1] == 'C' else 'C'


class GenerousTitForTat(Strategy):
    """
    Tit-for-Tat with probabilistic forgiveness.
    
    Like TFT, but occasionally forgives defections and cooperates anyway.
    This helps escape mutual defection cycles caused by noise or mistakes.
    """
    
    def __init__(self, forgiveness_probability=0.1):
        super().__init__("Generous Tit-for-Tat")
        self.forgiveness = forgiveness_probability
    
    def choose_action(self, my_history, opponent_history):
        if len(opponent_history) == 0:
            return 'C'
        
        if opponent_history[-1] == 'D':
            if random.random() < self.forgiveness:
                return 'C'
        
        return opponent_history[-1]


class Gradual(Strategy):
    """
    Gradual retaliation strategy with increasing punishment.
    
    After opponent's nth defection:
    1. Defect n times in retaliation
    2. Then cooperate twice to signal willingness to restore cooperation
    
    This implements escalating punishment with clear reconciliation signals.
    """
    
    def __init__(self):
        super().__init__("Gradual")
        self.defection_count = 0
        self.punishment_left = 0
        self.calming_left = 0
    
    def choose_action(self, my_history, opponent_history):
        if len(opponent_history) == 0:
            return 'C'
        
        if opponent_history[-1] == 'D' and self.punishment_left == 0 and self.calming_left == 0:
            self.defection_count += 1
            self.punishment_left = self.defection_count
        
        if self.punishment_left > 0:
            self.punishment_left -= 1
            if self.punishment_left == 0:
                self.calming_left = 2
            return 'D'
        
        if self.calming_left > 0:
            self.calming_left -= 1
            return 'C'
        
        return 'C'
    
    def reset(self):
        self.defection_count = 0
        self.punishment_left = 0
        self.calming_left = 0


class SuspiciousTitForTat(Strategy):
    """
    Like Tit-for-Tat, but starts with defection instead of cooperation.
    
    More cautious variant that tests opponent before cooperating.
    """
    
    def __init__(self):
        super().__init__("Suspicious Tit-for-Tat")
    
    def choose_action(self, my_history, opponent_history):
        if len(opponent_history) == 0:
            return 'D'
        return opponent_history[-1]


class Prober(Strategy):
    """
    Tests opponent with initial defections, then adapts.
    
    Starts with D-C-C to probe opponent's response. If opponent cooperates
    after being defected on, switches to Always Defect (exploits naive opponent).
    Otherwise, plays Tit-for-Tat.
    """
    
    def __init__(self):
        super().__init__("Prober")
        self.mode = None
    
    def choose_action(self, my_history, opponent_history):
        if len(my_history) == 0:
            return 'D'
        elif len(my_history) == 1:
            return 'C'
        elif len(my_history) == 2:
            if opponent_history[1] == 'C':
                self.mode = 'exploit'
            else:
                self.mode = 'tft'
            return 'C'
        
        if self.mode == 'exploit':
            return 'D'
        else:
            return opponent_history[-1]
    
    def reset(self):
        self.mode = None


class Adaptive(Strategy):
    """
    Adapts based on which action (C or D) yields better average payoff.
    
    Tracks average scores when cooperating vs defecting, and increasingly
    favors the action that performed better. Uses standard PD payoffs.
    """
    
    def __init__(self):
        super().__init__("Adaptive")
        self.c_scores = []
        self.d_scores = []
        self.payoffs = {('C', 'C'): 3, ('C', 'D'): 0, ('D', 'C'): 5, ('D', 'D'): 1}
    
    def choose_action(self, my_history, opponent_history):
        if len(my_history) < 6:
            return 'C' if len(my_history) % 2 == 0 else 'D'
        
        my_last = my_history[-1]
        opp_last = opponent_history[-1]
        score = self.payoffs[(my_last, opp_last)]
        
        if my_last == 'C':
            self.c_scores.append(score)
        else:
            self.d_scores.append(score)
        
        avg_c = np.mean(self.c_scores) if self.c_scores else 0
        avg_d = np.mean(self.d_scores) if self.d_scores else 0
        
        return 'C' if avg_c >= avg_d else 'D'
    
    def reset(self):
        self.c_scores = []
        self.d_scores = []


class HardTitForTat(Strategy):
    """
    Aggressive variant: retaliates with extra defection.
    
    After opponent defects, defects twice before returning to cooperation.
    More punishing than standard TFT.
    """
    
    def __init__(self):
        super().__init__("Hard Tit-for-Tat")
        self.punishment_left = 0
    
    def choose_action(self, my_history, opponent_history):
        if len(opponent_history) == 0:
            return 'C'
        
        if self.punishment_left > 0:
            self.punishment_left -= 1
            return 'D'
        
        if opponent_history[-1] == 'D':
            self.punishment_left = 1
            return 'D'
        
        return 'C'
    
    def reset(self):
        self.punishment_left = 0


class SoftMajority(Strategy):
    """
    Cooperates if opponent has cooperated more than they've defected.
    
    Starts with cooperation. Then chooses based on opponent's cooperation rate:
    - If opponent cooperated >= 50% of the time, cooperate
    - Otherwise, defect
    """
    
    def __init__(self):
        super().__init__("Soft Majority")
    
    def choose_action(self, my_history, opponent_history):
        if len(opponent_history) == 0:
            return 'C'
        
        cooperation_rate = opponent_history.count('C') / len(opponent_history)
        return 'C' if cooperation_rate >= 0.5 else 'D'


class HardMajority(Strategy):
    """
    Like Soft Majority, but starts with defection and requires >50% cooperation.
    
    More skeptical variant that needs clear evidence of cooperation.
    """
    
    def __init__(self):
        super().__init__("Hard Majority")
    
    def choose_action(self, my_history, opponent_history):
        if len(opponent_history) == 0:
            return 'D'
        
        cooperation_rate = opponent_history.count('C') / len(opponent_history)
        return 'C' if cooperation_rate > 0.5 else 'D'


def get_all_strategies():
    """
    Convenience function to create instances of all implemented strategies.
    
    Returns:
    --------
    list : List of all strategy instances
    """
    return [
        AlwaysCooperate(),
        AlwaysDefect(),
        Random(),
        TitForTat(),
        TitForTwoTats(),
        Grim(),
        Pavlov(),
        GenerousTitForTat(),
        Gradual(),
        SuspiciousTitForTat(),
        Prober(),
        Adaptive(),
        HardTitForTat(),
        SoftMajority(),
        HardMajority()
    ]
