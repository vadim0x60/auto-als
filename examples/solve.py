from subprocess import call
import click
import os

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import torch

import wandb
from wandb.integration.sb3 import WandbCallback

import gymnasium as gym
import auto_als

TOTAL_TIMESTEPS=int(os.environ.get('TOTAL_TIMESTEPS', 1000))
N_EPISODE_STEPS = 1000

def no_history_obs(obs):
    obs = obs.copy()
    obs[:43][obs[:43] < 1] = 0
    return obs

@click.command()
@click.option('--attach/--launch', default= False)
@click.option('--render', is_flag=True)
@click.option('--baseline', is_flag=True)
@click.option('--device', default='auto')
def solve(attach, baseline, device, render):
    #evaluate_policy(model, env)
    try:
        env = gym.make('Auto-ALS-v0', attach=attach, autoplay=True, render=render)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=N_EPISODE_STEPS)

        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'

        config = {
            'total_timesteps': TOTAL_TIMESTEPS,
            'env': 'Auto-ALS-v0',
            'baseline': baseline
        }

        run = wandb.init(
            project="auto-als",
            config=config,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True
        )

        if baseline:
            env = gym.wrappers.TransformObservation(env, no_history_obs)

        print('First episode is a sanity check. Let us do something random')
        env.reset()
        #env.render("False")
        terminated = False
        truncated = False
        actions = []
        while not (terminated or truncated):
            if len(actions) == 20:
                action = 49 # end episode
            else:
                action = env.action_space.sample()
            actions.append(action)
            print(action)
            _, reward, terminated, truncated, info = env.step(action)
            print(info['memos'])
            print(reward)

        print('Sanity check OK, looking for optimal solution')

        callback = WandbCallback(model_save_freq=100, 
                                 gradient_save_freq=100, 
                                 model_save_path=f"models/{run.id}", 
                                 verbose=2)

        alg = PPO('MlpPolicy', env, verbose=1, tensorboard_log=f"runs/{run.id}", device=device)
        model = alg.learn(TOTAL_TIMESTEPS, callback=callback)
        evaluate_policy(model, env)
    finally:
        env.close()

    run.finish()

if __name__ == '__main__':
    solve()
