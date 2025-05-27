from enum import Enum
from Action import Action
import numpy as np
import random

class Agent:

    def __init__(self, epochs, epsilon, epsilon_min, epsilon_decay, learning_rate,learning_rate_decay, learning_rate_min, discount_factor):
        self.epochs = epochs
        
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        
        self.learning_rate = learning_rate
        self.learning_rate_min = learning_rate_min
        self.learning_rate_decay = learning_rate_decay

        self.training_error = []
        self.discount_factor = discount_factor

        n_states = 32*11*2 #player_hand, dealer first card, has atleast one soft_ace.
        n_actions = 2 #hit or stay.

        self.table = np.zeros((n_states, n_actions)) 

    def _print_odds(self, index):
        #asked chatGPT to generate.
        row = self.table[index]
        action_values = {
            "Hit": float(row[Action.Hit.value]),
            "Stand": float(row[Action.Stand.value])
        }

        # Determine the chosen action based on max Q-value
        chosen_action = max(action_values, key=action_values.get)

        #print(f"Q-values: {action_values} | Action:{chosen_action}")

    def get_q_table(self):
        return self.table
    
    def get_error(self):
        return self.training_error

    def get_action(self, state):
        index = self._state_to_index(state)
        if np.random.random() < self.epsilon:
            #print("action chosen at random.")
            return random.choice(list(Action))
        else:
            self._print_odds(index)
            return Action(np.argmax(self.table[index]))

    def _state_to_index(self, state):
    #GOT THIS SOLUTION FROM CHATGPT.

    # Adjust ranges so everything starts at 0
    # player_sum: 0 to 31 → 0–31
    # dealer_card: 1 to 11 → 0–10
    # has_soft_ace: True/False → 1/0
        player_sum, dealer_card, has_soft_ace = state
        return player_sum * 11 * 2 + (dealer_card - 1) * 2 + int(has_soft_ace)
        
    def update(self, state: tuple[int, int, bool], action: Action, reward, bust, next_state: tuple[int, int, bool]):
        next_index = self._state_to_index(next_state)
        index = self._state_to_index(state)

        next_q_value = (not bust) * np.max(self.table[next_index])
        diff = (reward + self.discount_factor * next_q_value - self.table[index][action.value])

        self.table[index][action.value] += self.learning_rate * diff
        
        self.training_error.append(diff)
        
        self._decay_epsilon() #update epsilon.
        self._decay_learning_rate()

    def _decay_epsilon(self):
        if self.epsilon*self.epsilon_decay > self.epsilon_min:
            self.epsilon = self.epsilon*self.epsilon_decay

    def _decay_learning_rate(self):
        if self.learning_rate*self.learning_rate_decay > self.learning_rate_min:
            self.learning_rate = self.learning_rate*self.learning_rate_decay


    

