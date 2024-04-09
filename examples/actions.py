import gymnasium as gym
import auto_als
import click

@click.command('Try all available actions')
@click.option('--render', is_flag=True)
def actions(render):
    env = gym.make('Auto-ALS-v0', attach=False, render=render, autoplay=True)

    print(env.reset())

    plan = [
        a 
        for _ in range(10)
        for a in range(len(auto_als.actions) - 1)
    ] + [len(auto_als.actions) - 1]

    print(f'About to make {len(plan)} actions. Wish me luck')

    for a in plan:
        print(auto_als.actions[a])
        ret = env.step(a)
        print(ret)

        if (ret[2] or ret[3]):
            break

    env.close()

if __name__ == '__main__':
    actions()