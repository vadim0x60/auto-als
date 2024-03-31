import gymnasium as gym
import auto_als
import click

reset_hint = """
Number of times to reset the environment
Reset bombing has shown to be a useful stress test for the environment
"""

@click.command('Try all available actions')
@click.option('--render', is_flag=True)
@click.option('--reset-count', default=1, type=int, help=reset_hint)
def actions(render, reset_count):
    env = gym.make('Auto-ALS-v0', attach=False, render=render, autoplay=True)

    for _ in range(reset_count):
        print(env.reset())

    for idx, name in enumerate(auto_als.actions):
        print(name)
        ret = env.step(idx)
        print(ret)

        if (ret[2] or ret[3]):
            print(env.reset())

    env.close()

if __name__ == '__main__':
    actions()