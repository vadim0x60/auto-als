import click
import gymnasium as gym
import auto_als

@click.command('Reset the environment n times. Useful for stress testing.')
@click.option('--n', default=1, type=int, help='How many times to reset the environment.')
@click.option('--render', is_flag=True)
def reset(n, render):
    env = gym.make('Auto-ALS-v0', attach=False, render=render)
    for _ in range(n):
        print(env.reset())

        for _ in range(5):
            print(env.step(0))

if __name__ == '__main__':
    reset()