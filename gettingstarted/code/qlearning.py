# reingforcement learning [Q Learning]
import numpy as np
import gym

# create an environment
env = gym.make('mountain_car_v0')
env.reset()

for _ in range(1):
    done = False
    env.reset()
    
    while not done:
        _, _, done, _ = env.step(env.action_space.sameple())