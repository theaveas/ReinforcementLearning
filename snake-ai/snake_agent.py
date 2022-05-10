from stable_baselines3 import PPO
from snake_env import SnakeEnv

if __name__ == "__main__":
    env = SnakeEnv()
    # define a model
    model = PPO('MlpPolicy', env, verbose=1)
    # train the model
    TRAIN = True
    if TRAIN:
        model.learn(total_timesteps=int(1e4))
        model.save('snake_agent_model')
    # load a saved model
    # model = PPO.load('snake_agent_model')
    for i in range(2):
        done = False
        state = env.reset()
        for _ in range(20):
            action, _ = model.predict(state, deterministic=True)
            state, reward, done, _ = env.step(action)
            env.render()
            if done:
                state = env.reset()
    env.close()
