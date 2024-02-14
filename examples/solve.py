from subprocess import call
import click

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

import wandb
from wandb.integration.sb3 import WandbCallback

import gym
import auto_als

TOTAL_TIMESTEPS=10000

def no_history_obs(obs):
    obs = obs.copy()
    obs[:43][obs[:43] < 1] = 0
    return obs

@click.command()
@click.option('--attach/--launch', default= False)
@click.option('--baseline', is_flag=True)
def main(attach, baseline):
    #evaluate_policy(model, env)
    env = gym.make('Auto-ALS-v0', attach=attach, render = True)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=256)

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
    done = False
    actions = []
    while not done:
        if len(actions) == 20:
            action = 34 # end episode
        else:
            action = env.action_space.sample()
        actions.append(action)
        print(action)
        _, reward, done, _ = env.step(action)
        print(reward)

    env.reset()
    print('Sanity check OK, looking for optimal solution')

    alg = PPO('MlpPolicy', env, verbose=1, tensorboard_log=f"runs/{run.id}")
    model = alg.learn(TOTAL_TIMESTEPS, callback=WandbCallback(model_save_freq=100, gradient_save_freq=100, model_save_path=f"models/{run.id}", verbose=2))
    evaluate_policy(model, env)

    run.finish()

if __name__ == '__main__':
    main()
