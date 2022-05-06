# require package
# gym[box2d], stable-baselines3[extra], huggingface_sb3

# import package
import os
from token import OP
import gym

from huggingface_sb3 import load_from_hub, package_to_hub, push_to_hub
from huggingface_hub import notebook_login

from stable_baselines3 import PPO # learning algorithm
from stable_baselines3.common.evaluation import evaluate_policy # evaluate policy to eval the model
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, EvalCallback

import tensorboard
from torch import det # monitor for logging



LOG_DIRS = 'logs'
OPT_DIRS = 'opts'

# make log and optimize model direction
if not os.path.exists(LOG_DIRS): os.makedirs(LOG_DIRS)
if not os.path.exists(OPT_DIRS): os.makedirs(OPT_DIRS)


# test the environment
def test_agent(env):
    state = env.reset()
    
    done = False
    
    while not done:
    # take 20 random actions
        action = env.action_space.sample()
        _, _, done, _ = env.step(action)
        env.render()
        
        if done:
            print('Environment is reset')
            state = env.reset()
    
    env.close()
    

# vectorized environment
def vec_env():
    """
    Method for stacking multiple independent environment into a single environment
    """
    env = make_vec_env('LunarLander-v2', n_envs=16)
    

def train_model(timesteps=int(2e5)):
    env = gym.make('LunarLander-v2')
    
    # initialize the model
    model = PPO('MlpPolicy', env=env, n_steps= 1024, batch_size=64, n_epochs=4, gamma=.999, 
                gae_lambda= .98, ent_coef= .01, verbose=1, tensorboard_log=LOG_DIRS)
    
    # callbacks
    # save every n timesteps
    checkpoint_cb = CheckpointCallback(save_freq=500000, save_path=OPT_DIRS)
    # save best model
    BEST_DIR = os.path.join(OPT_DIRS, 'best_model')
    eval_cb = EvalCallback(env, best_model_save_path=BEST_DIR, log_path=LOG_DIRS, eval_freq=100000, deterministic=True, render=False)
    # call back chain list
    callback = CallbackList([checkpoint_cb, eval_cb])
    
    # train the agent
    model.learn(total_timesteps=timesteps, callback=callback, tb_log_name='PP0')
    
    return model
    
    
if __name__ == "__main__":
    # create an environment
    env = gym.make('LunarLander-v2')
    env = Monitor(env, LOG_DIRS)
    
    # A Quicker look at the Environment
    # observation space
    # obs_space = env.observation_space.shape
    # print(f'Observation Space shape: {obs_space}')
    
    # # actions space
    # action_space = env.action_space.n
    # print(f'Actions space shape: {action_space}')
    
    
    # test the agent
    TEST = False
    if TEST:
        test_agent(env)
    
    MODEL_PATH = os.path.join(OPT_DIRS, 'ppo_lunarlander_1e5')
    # build and train the model
    TRAIN = True
    if TRAIN:
        model = train_model(5000000)
        model.save(MODEL_PATH)
  
    
    model = PPO.load(MODEL_PATH, env=env)
    
    # evaluate the model
    mean_reward, std_reward= evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)
    print(f'Mean reward: {mean_reward:.2f}, Standard diviation reward per episode:{std_reward}')
    
    
    # Enjoy the trained agent
    for i in range(5):
        done = False
        state = env.reset()
        
        while not done:
            action, _ = model.predict(state, deterministic=True)
            state, reward, done, info = env.step(action)
            env.render()
    
    env.close()
   
    
