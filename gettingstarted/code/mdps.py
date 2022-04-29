# reinforcement learning (mdps)
import gym 
import numpy as np

# create and environment
env = gym.make('CartPole-v1')
env.reset()

# environment action, observation space
# inspect action space by using `env.action_space.n`
# inspect observation space by using `env.observation_space.nvec`

# policy
def random_policy(state):
    return np.array([.5] * env.action_space.n) 

# reward 
# agent recieve a reward after interact(selecting action) with environment
'''
for action in range(1):
    _, reward, _, _ = env.step(env.action_space.sample())
'''

# return
# future expect cumulative reward (the sum of reward through out the episodes)
# return += gamma (discount rate) ** timestep * reward


if __name__ == "__main__":
    SHOW_EVERY = 50
    EPISODES = 500
    DISCOUNT = .9
    timestep = 0
    G = 0
    

    for episode in range(EPISODES):
        done = False
        env.reset()
        
        if episode % SHOW_EVERY == 0:
            render = True
            print(f'The return of episode {episode} is {np.round(G, 2)}.')
        else:
            render = False
        
        while not done:
            # action-environment interaction
            state_, reward, done, _ = env.step(env.action_space.sample())
            
            # render the environment
            if render: env.render()
            
            # calc the return
            G += DISCOUNT ** timestep * reward
            timestep += 1
        
        
    
    # close the environment
    env.close()
    
        
            
        