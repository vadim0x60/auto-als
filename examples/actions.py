import gymnasium as gym
import auto_als
import click

@click.command('Try all available actions')
@click.option('--render', is_flag=True)
def actions(render):
    actions = {a: i for i, a in enumerate(auto_als.actions)}

    env = gym.make('Auto-ALS-v0', attach=False, render=render, autoplay=True)
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