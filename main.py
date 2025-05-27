from Deck import *
from Hand import Hand
from Environment import Environment
from Agent import Agent
from Plots import print_q_table, plot_tracking, plot_error

import time

EPOCHS = 10000000

EPSILON = [0.9]
EPSILON_DECAY = [0.95]
EPSILON_MIN = 0.005  # keep constant for now

LEARNING_RATE = [0.3]
LEARNING_RATE_DECAY = [0.85]
LEARNING_RATE_MIN = 0.005  # keep constant for now

DISCOUNT_FACTOR = [0.015]

NUMBER_OF_DECKS = 8

TUNING_SIZE = len(EPSILON) * len(EPSILON_DECAY) * len(LEARNING_RATE) * len(LEARNING_RATE_DECAY) * len(DISCOUNT_FACTOR)

def main():
    print("Simulating...")
    tuning_count = 0
    tuning_time = time.time()
    for learning_rate_decay in LEARNING_RATE_DECAY:
        for discount in DISCOUNT_FACTOR:
            for epsilon in EPSILON:
                for epsilon_decay in EPSILON_DECAY:
                    for learning_rate in LEARNING_RATE:
                            
                            start_time = time.time()
                            env = Environment(NUMBER_OF_DECKS, EPOCHS)
                            agent = Agent(EPOCHS, epsilon, EPSILON_MIN, epsilon_decay, learning_rate, learning_rate_decay,  LEARNING_RATE_MIN, discount)
                            simulate(env, agent)
                            print("Done with simulation, now fix plot.")
                            #total games played is == epochs.
                            win_track, lose_track, draw_track = env.get_track_lists()

                            #print_q_table(agent.get_q_table())
                            plot_error(agent.get_error())

                            tuning_count+=1

                        
                        
def simulate(env, agent):
    for epoch in range(EPOCHS):
        done = False
        current_state = env.reset()
        while not done:
            action = agent.get_action(current_state)

            next_state, reward, bust, terminate = env.step(action)

            agent.update(current_state, action, reward, bust, next_state)

            done = terminate or bust
            current_state = next_state
    

if __name__=="__main__":
    main()