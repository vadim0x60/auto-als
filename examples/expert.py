import gymnasium as gym
from auto_als import actions



actions = {a: i for i, a in enumerate(actions)}

env = gym.make('Auto-ALS-v0', attach=False, render=True, autoplay=False)
env.reset()
env.step(actions['AssessAirway'])
env.step(actions['AssessBreathing'])
env.step(actions['AssessCirculation'])
env.step(actions['AssessDisability'])
env.step(actions['AssessExposure'])