import gymnasium as gym
import auto_als
import click

@click.command()
@click.option('--render', is_flag=True)
def expert(render):
    actions = {a: i for i, a in enumerate(auto_als.actions)}

    try:
        env = gym.make('Auto-ALS-v0', attach=False, render=render, autoplay=True)
        for _ in range(1 if render else 3):
            print(env.reset())
            print(env.step(actions['ExamineAirway']))
            print(env.step(actions['ExamineBreathing']))
            print(env.step(actions['ExamineCirculation']))
            print(env.step(actions['ExamineDisability']))
            print(env.step(actions['ExamineExposure']))
            if not render:
                print(env.step(actions['Finish']))
    finally:
        env.close()

if __name__ == '__main__':
    expert()