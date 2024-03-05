import gymnasium as gym
import auto_als
import click

@click.command()
@click.option('--render', is_flag=True)
def master(render):
    actions = {a: i for i, a in enumerate(auto_als.actions)}

    env = gym.make('Auto-ALS-v0', attach=False, render=render, autoplay=True)
    for _ in range(1 if render else 3):
        print(env.reset())
        print(env.step(actions['AssessAirway']))
        print(env.step(actions['AssessBreathing']))
        print(env.step(actions['AssessCirculation']))
        print(env.step(actions['AssessDisability']))
        print(env.step(actions['AssessExposure']))
        if not render:
            print(env.step(actions['Finish']))

    env.close()

if __name__ == '__main__':
    master()