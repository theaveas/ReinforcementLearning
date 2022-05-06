# require package
# gym[box2d], stable-baselines3[extra]

# import package
import os
import gym

from stable_baselines3 import PPO # learning algorithm
from stable_baselines3.common.evaluation import evaluate_policy # evaluate policy to eval the model
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, EvalCallback

# make log and optimize model direction
LOG_DIRS = 'logs'
OPT_DIRS = 'opts'

if not os.path.exists(LOG_DIRS): os.makedirs(LOG_DIRS)
if not os.path.exists(OPT_DIRS): os.makedirs(OPT_DIRS)


# test the environment for one episode
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
    
    
# train agent
def train_agent(trial):
    env = gym.make("LunarLander-v2")
    env = Monitor(env, LOG_DIRS) 
    env = DummyVecEnv([lambda: env])

    # initialize the model
    model = PPO('MlpPolicy', env=env, verbose=1, tensorboard_log=LOG_DIRS)
    
    # callbacks
    # save every n timesteps
    checkpoint_cb = CheckpointCallback(save_freq=200000, save_path=OPT_DIRS)
    # save best model
    BEST_DIR = os.path.join(OPT_DIRS, 'best_model')
    eval_cb = EvalCallback(env, best_model_save_path=BEST_DIR, log_path=LOG_DIRS, eval_freq=100000, deterministic=True, render=False)
    # call back chain list
    callbacks = CallbackList([checkpoint_cb, eval_cb])
    
    # train the agent
    model.learn(total_timesteps=10000, callback=callbacks, tb_log_name='PP0')
    return model
    
    
if __name__ == "__main__":
    # create an environment
    # You can play with other environment By changing 'LunarLander-v2'
    env = gym.make('LunarLander-v2')
    env = Monitor(env, LOG_DIRS)
    env = DummyVecEnv([lambda: gym.make('LunarLander-v2')])
    
    # A Quicker look at the Environment
    # observation space
    # obs_space = env.observation_space.shape
    # print(f'Observation Space shape: {obs_space}')
    
    # # actions space
    # action_space = env.action_space.n
    # print(f'Actions space shape: {action_space}')
    
    MODEL_PATH = os.path.join(OPT_DIRS, 'ppo-lunarlander-v2')
    
    # test the agentc
    TEST = False
    if TEST:
        test_agent(env)
   
    # build and train the model
    TRAIN = True
    if TRAIN:
        model = train_agent()
        model.save(MODEL_PATH)
        
    # load a (best) model
    # best model path: 'opts/best_model/best_model.zip'
    model = PPO.load(MODEL_PATH, env=env)
    
    # evaluate the model
    mean_reward, std_reward= evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)
    print(f'Mean reward: {mean_reward:.2f}, Standard diviation reward per episode:{std_reward}')
    
    # Enjoy the trained agent 
    # Render out the environment
    for i in range(5):
        done = False
        state = env.reset()
        
        while not done:
            action, _ = model.predict(state, deterministic=True)
            state, reward, done, info = env.step(action)
            env.render()
    
    env.close()
   
    
