# introduction to stable-baselines3
from re import L
import gym
import numpy as np

from stable_baselines3 import A2C


# environment test using random action
def game_test():
    # create env
    env = gym.make('LunarLander-v2')
    
    # test game environment for 10 episode
    for _ in range(10):
        done = False
        env.reset()
        if done:
            env.reset()
        
        while not done:
            env.render()
            action = env.action_space.sample()
            _, _, done, _ = env.step(action)
                     
        
    env.close()
    
    
# moon landing using stable baselines3
def game_A2C():
    # create env
    env = gym.make('LunarLander-v2')
    
    # build a model
    model = A2C('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=100000)
  
    for i in range(10):
        done = False
        state = env.reset()
    
        if done:
            state = env.reset()
            
        while not done:
            action, _state = model.predict(state, deterministic=True)
            state, reward, done, info = env.step(action)
            env.render()
            
        
            
    env.close()
        
if __name__ == "__main__":
    game_A2C()