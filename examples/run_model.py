import click
from stable_baselines3.common.base_class import BaseAlgorithm 
import gymnasium as gym

@click.command()
@click.argument('path')
def run_model(path):
    model = BaseAlgorithm.load(path)
    env = gym.make('Auto-ALS-v0', attach=False, render=False, autoplay=True)

    obs, info = env.reset()
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action = model.predict(obs, deterministic=True)
        _, reward, terminated, truncated, info = env.step(action)
        print(info['memos'])
        print(reward)

if __name__ == '__main__':
    run_model()