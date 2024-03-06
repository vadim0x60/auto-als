import gymnasium as gym
import auto_als
import click

@click.command('Try all available actions')
@click.option('--render', is_flag=True)
def master(render):
    actions = {a: i for i, a in enumerate(auto_als.actions)}

    env = gym.make('Auto-ALS-v0', attach=False, render=render, autoplay=True)
    print(env.reset())
    for idx, name in enumerate(auto_als.actions):
        print(name)
        print(env.step(idx))

    env.close()

if __name__ == '__main__':
    master()