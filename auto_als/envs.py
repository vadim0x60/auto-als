import pkg_resources

from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper, UnityGymException

def AutoALS(attach=False, render=False):
    if attach:
        unity_env = UnityEnvironment()
    elif render:
        build = pkg_resources.resource_filename('auto_als.UnityBuilds', 'Autoplay')
        unity_env = UnityEnvironment(build)
    else:
        build = pkg_resources.resource_filename('auto_als.UnityBuilds', 'ServerAutoplay')
        unity_env = UnityEnvironment(build)
    return UnityToGymWrapper(unity_env)