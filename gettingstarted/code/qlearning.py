# reingforcement learning [Q Learning]
import numpy as np
import gym


# create an environment
env = gym.make('MountainCar-v0')
env.reset()

# state argreation 
# turn a continuos task into episodic task
GRID_SIZE = [20] * len(env.observation_space)
win_size = (env.observation_space.high - env.observation_space.low) / GRID_SIZE

# create a q table (action-value) range from -2 to 0 with shape of [obs_state_size, obs_state_size, action_size]
Q = np.random.uniform(low= -2, high=0, size=GRID_SIZE + [env.action_space.n])


def test_env():
    for _ in range(1):
        done = False
        env.reset()
        
        while not done:
            _, _, done, _ = env.step(env.action_space.sample())
            env.render()
            
    env.close()
    

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / win_size
    return tuple(discrete_state.astype(np.int))


if __name__ == "__main__":
    learning_rate = .1
    discount_factor = .95
    episodes = 25000
    