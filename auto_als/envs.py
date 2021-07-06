import pkg_resources
import time

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.exception import UnityEnvironmentException, UnityWorkerInUseException

from mlagents_envs import logging_util
logger = logging_util.get_logger(__name__)

from gym_unity.envs import UnityToGymWrapper, UnityGymException

def proivision_unity_env(attach=False, render=False, sleep=0.1):
    logger.info(f"New unity environment will be provisioned in {sleep} seconds")
    time.sleep(sleep)

    try:
        if attach:
            unity_env = UnityEnvironment()
        elif render:
            build = pkg_resources.resource_filename('auto_als.UnityBuilds', 'Autoplay')
            unity_env = UnityEnvironment(build)
        else:
            build = pkg_resources.resource_filename('auto_als.UnityBuilds', 'ServerAutoplay')
            unity_env = UnityEnvironment(build)
        return unity_env
    except UnityWorkerInUseException:
        # Exponential backoff
        proivision_unity_env(attach, render, sleep * 2)

class AutoALS(UnityToGymWrapper):
    def __init__(self, attach=False, render=False):
        self.attach_ = attach
        self.render_ = render

        unity_env = proivision_unity_env(attach, render)
        super().__init__(unity_env)

    def reset(self):
        try:
            return super().reset()
        except (UnityEnvironmentException, UnityGymException):
            self._env.close()
            action_taken = 'reattach to' if self.attach_ else 'restart'
            logger.warn(f'Built-in reset functionality failed. Had to {action_taken} the environment')
            super().__init__(proivision_unity_env(self.attach_, self.render_))
            return super().reset()
