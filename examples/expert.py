import gymnasium as gym
from auto_als import actions



actions = {a: i for i, a in enumerate(actions)}

env = gym.make('Auto-ALS-v0', attach=False, render=True, autoplay=True)
env.reset()
print(env.step(actions['AssessAirway']))
print(env.step(actions['AssessBreathing']))
print(env.step(actions['AssessCirculation']))
print(env.step(actions['AssessDisability']))
print(env.step(actions['AssessExposure']))
print(env.step(actions['Finish']))