from distutils.command.build import build
from enum import auto
import time
from pathlib import Path
import sys

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.exception import UnityEnvironmentException, UnityWorkerInUseException

from mlagents_envs import logging_util
logger = logging_util.get_logger(__name__)

from gym_unity.envs import UnityToGymWrapper, UnityGymException

from worstpractices import remedy
from tenacity import retry
from tenacity.wait import wait_exponential
from tenacity.retry import retry_if_exception_type

BUILDS_PATH = Path(__name__).parent / 'UnityBuilds'

virtu_als_release = '1.0'

launcher_suffix = {
    'linux': '.x86_64',
    'win32': 'Win'
}

download_msg = """Downloading a copy of Virtu-ALS... 
                  This will take up to 0.5 GB of traffic"""

def download_build(build='full'):
    from urllib.request import urlopen
    from zipfile import ZipFile
    from io import BytesIO

    print(download_msg)

    loc = 'https://github.com/vadim0x60/virtu-als-plus/releases/download/'
    slug = f'{virtu_als_release}/{virtu_als_release}-{build}-{sys.platform}.zip'

    with urlopen(loc + slug) as zipresp:
        with ZipFile(BytesIO(zipresp.read())) as zfile:
            zfile.extractall(BUILDS_PATH)

@remedy(UnityEnvironmentException, download_build)
@retry(retry=retry_if_exception_type(UnityWorkerInUseException),
       wait=wait_exponential(multiplier=0.1, min=0.1))
def proivision_unity_env(build='full', attach=False, autoplay=True):
    launcher = BUILDS_PATH / f'{build.capitalize()}{launcher_suffix[sys.platform]}'
    launcher = str(launcher)

    additional_args = []
    if autoplay:
        additional_args.append('--autoplay')

    if attach:
        unity_env = UnityEnvironment()
    else:
        unity_env = UnityEnvironment(launcher, additional_args=additional_args)
    return unity_env

class AutoALS(UnityToGymWrapper):
    def __init__(self, attach=False, render='auto', autoplay=True):
        if render == 'auto':
            render = False if autoplay else True
        
        assert autoplay or render, 'Hybrid mode requires render to be set to True'

        self.attach_ = attach
        self.render_ = render

        build = 'full' if render else 'headless'

        unity_env = proivision_unity_env(build, attach, autoplay)
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