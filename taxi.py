# import 
import gym
import random
import numpy as np


# Taxi Agent
def main():
    
    # create env
    env = gym.make('Taxi-v3')
    
    # define q table
    state_size = env.observation_space.n
    action_size = env.action_space.n
    
    q_table = np.zeros(shape=(state_size, action_size))
    
    # hyperparameter
    learning_rate = .9
    discount_rate = .9
    decay_rate = 5e-3
    epsilon = 1.
    
    # training variable
    episodes = 1000
    trajectory = 99
    
    # training 
    for eps in range(episodes):
        state = env.reset()
        done = False
        
        for s in range(trajectory):
            # explore
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            # exploit
            else:
                action = np.argmax(q_table[state, :])
                
            # take action, observe, reward
            next_state, reward, done, _ = env.step(action)
            
            # update q_table
            q_table[state, action] =  q_table[state, action] + learning_rate * (reward + discount_rate * np.max(q_table[next_state,:]) - q_table[state, action])
            
            # update state
            state = next_state
            
            if done:
                break
            
        # epsilon decay
        epsilon = np.exp(-decay_rate)
        
    print(f'Training Complete over {episodes} episodes.')
    input('Press Enter to watch trained agent...')
    
        
    # display trained agent
    state = env.reset()
    done = False
    reward = 0
    
    for s in range(trajectory):
        print('Trained Agent')
        print(f'Step {s+1}')
        
        action = np.argmax(q_table[state, :])
        next_state, reward, done, _ = env.step(action)
        reward += reward
        
        env.render()
        print(f'scores: {reward}')
        
        if done:
            break
        
    env.close()
        
if __name__ == "__main__":
    main()
    