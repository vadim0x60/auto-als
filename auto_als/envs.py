from distutils.command.build import build
from enum import auto
import time
from pathlib import Path
import sys

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.exception import UnityEnvironmentException, UnityWorkerInUseException

from mlagents_envs import logging_util
logger = logging_util.get_logger(__name__)

from auto_als.unity_gym_env import UnityToGymWrapper, UnityGymException

from tenacity import retry
from tenacity.wait import wait_exponential
from tenacity.retry import retry_if_exception_type
from tenacity import stop_after_attempt

BUILDS_PATH = Path(__file__).parent.parent.resolve() / 'UnityBuilds'
ORIGIN = 'https://github.com/vadim0x60/virtu-als-plus/releases/download/1.1.2/'
DOWNLOAD_MSG = """Downloading a copy of Virtu-ALS... 
                  This will take up to 0.5 GB of traffic"""

launcher_suffix = {
    'linux': '.x86_64',
    'win32': 'Win'
}




def required_build(render=False):
    if sys.platform == 'linux':
        build = 'StandaloneLinux64'
    elif sys.platform == 'win32':
        # https://stackoverflow.com/questions/2208828/detect-64bit-os-windows-in-python
        if platform.machine().endswith('64'):
            build = 'StandaloneWindows64'
        else:
            build = 'StandaloneWindows32'
    elif sys.platform == 'darwin':
        build = 'virtu-als2018.app'
    else:
        raise UnityGymException(f'Unsupported platform: {sys.platform}')

    src = ORIGIN + build + ('' if render else '-standaloneBuildSubtargetServer') + '.zip'
    dest = BUILDS_PATH / ('graphic' if render else 'headless')
    launcher = BUILDS_PATH / ('graphic' if render else 'headless') / build

    return src, dest, launcher

def download_build(render):
    from urllib.request import urlopen
    from zipfile import ZipFile
    from io import BytesIO

    src, dest, _ = required_build(render)
    dest.mkdir(parents=True, exist_ok=True)

    with urlopen(src) as zipresp:
        with ZipFile(BytesIO(zipresp.read())) as zfile:
            zfile.extractall(dest)

@retry(retry=retry_if_exception_type(UnityEnvironmentException), 
       after=lambda rs: download_build(render=rs.args[0]),
       stop=stop_after_attempt(1))
@retry(retry=retry_if_exception_type(UnityWorkerInUseException),
       wait=wait_exponential(multiplier=0.1, min=0.1))
def proivision_unity_env(render=False, attach=False, autoplay=True):
    if attach:
        unity_env = UnityEnvironment()
    else:
        _, _, launcher = required_build(render)
        launcher = str(launcher)

        additional_args = []
        if autoplay:
            additional_args.append('--autoplay')

        unity_env = UnityEnvironment(launcher, additional_args=additional_args)
    return unity_env

class AutoALS(UnityToGymWrapper):
    def __init__(self, attach=False, render='auto', autoplay=True):
        if render == 'auto':
            render = False if autoplay else True
        
        assert autoplay or render, 'Hybrid mode requires render to be set to True'

        self.attach_ = attach
        self.render_ = render

        unity_env = proivision_unity_env(render, attach, autoplay)
        super().__init__(unity_env)

    def reset(self, seed=None):
        try:
            return super().reset()
        except (UnityEnvironmentException, UnityGymException):
            self._env.close()
            action_taken = 'reattach to' if self.attach_ else 'restart'
            logger.warn(f'Built-in reset functionality failed. Had to {action_taken} the environment')
            super().__init__(proivision_unity_env(self.attach_, self.render_))
            return super().reset()