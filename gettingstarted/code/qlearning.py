# reingforcement learning [Q Learning]
import numpy as np
import gym

# create an environment
env = gym.make('MountainCar-v0')
env.reset()

# state argreation 
# turn a continuos task into episodic task
GRID_SIZE = [20] * 2
win_size = (env.observation_space.high - env.observation_space.low) / GRID_SIZE

print(GRID_SIZE)

def test_env():
    for _ in range(1):
        done = False
        env.reset()
        
        while not done:
            _, _, done, _ = env.step(env.action_space.sample())
            env.render()
            
    env.close()
    
    

if __name__ == "__main__":
    pass