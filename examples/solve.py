from subprocess import call
import click

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback

import gym
import auto_als

@click.command()
@click.option('--attach/--launch', default=False)
def main(attach):
    env = gym.make('Auto-ALS-v0', attach=attach)

    print('First episode is a sanity check. Let us do something random')
    env.reset()
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

    model = PPO('MlpPolicy', env, verbose=True).learn(10000, callback=CheckpointCallback(1000, 'PPO'))
    evaluate_policy(model, env)

if __name__ == '__main__':
    main()
