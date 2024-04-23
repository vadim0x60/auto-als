from pathlib import Path
import sys
import time
import uuid
import gymnasium as gym
import os

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.exception import UnityEnvironmentException, UnityWorkerInUseException, UnityException
from mlagents_envs.side_channel.side_channel import (
    SideChannel,
    IncomingMessage,
    OutgoingMessage,
)

from mlagents_envs import logging_util
logger = logging_util.get_logger(__name__)

from auto_als.unity_gym_env import UnityToGymWrapper, UnityGymException
from auto_als.api import actions, observations

from tenacity import retry
from tenacity.wait import wait_exponential
from tenacity.retry import retry_if_exception_type
from tenacity import stop_after_attempt

class AutoALSException(Exception):
    pass

BUILDS_PATH = Path(__file__).parent.parent.resolve() / 'UnityBuilds'
ORIGIN = 'https://github.com/vadim0x60/virtu-als-plus/releases/download/1.5/'
DOWNLOAD_MSG = """Downloading a copy of Virtu-ALS... 
                  This will take up to 0.5 GB of traffic"""
SIDE_CHANNEL = uuid.UUID('bdb17919-c516-44da-b045-a2191e972dec')

def required_build():
    if sys.platform == 'linux':
        return 'StandaloneLinux64'
    elif sys.platform == 'win32':
        # https://stackoverflow.com/questions/2208828/detect-64bit-os-windows-in-python
        if platform.machine().endswith('64'):
            return 'StandaloneWindows64'
        else:
            return 'StandaloneWindows32'
    else:
        raise AutoALSException(f'Unsupported platform: {sys.platform}')

def download_build():
    from urllib.request import urlopen
    from zipfile import ZipFile
    from io import BytesIO

    build = required_build()

    with urlopen(ORIGIN + build + '.zip') as zipresp:
        with ZipFile(BytesIO(zipresp.read())) as zfile:
            zfile.extractall(BUILDS_PATH)

    (BUILDS_PATH / build).chmod(0o755)

@retry(retry=retry_if_exception_type(UnityEnvironmentException), 
       after=lambda rs: download_build(),
       stop=stop_after_attempt(2))
@retry(retry=retry_if_exception_type(UnityWorkerInUseException),
       wait=wait_exponential(multiplier=0.1, min=0.1))
def proivision_unity_env(render=False, attach=False, autoplay=True,
                         time_scale=None, side_channels=[], log_folder=None):
    if attach:
        unity_env = UnityEnvironment()
    else:
        build = required_build()
        launcher = str(BUILDS_PATH / build)

        additional_args = []
        if autoplay:
            additional_args.append('--autoplay')
        if time_scale:
            additional_args.append(f'--ts={time_scale}')

        unity_env = UnityEnvironment(launcher, no_graphics=not render, 
                                     additional_args=additional_args,
                                     log_folder=log_folder,
                                     side_channels=side_channels)
    return unity_env

class AutoALS(gym.Env, SideChannel):
    def __init__(self, attach=False, render=False, autoplay='auto', 
                 time_scale=None, log_folder='.'):
        gym.Env.__init__(self)
        SideChannel.__init__(self, SIDE_CHANNEL)

        if autoplay == 'auto':
            autoplay = False if render else True

        if time_scale is None:
            try:
                time_scale = float(os.environ['TIME_SCALE'])
                print('Setting timescale to', time_scale)
            except (KeyError, ValueError):
                pass
        
        assert autoplay or render, 'Hybrid mode requires render to be set to True'

        self.autoplay_ = autoplay
        self.time_scale_ = time_scale
        self.attach_ = attach
        self.render_ = render
        self.log_folder = log_folder
        self.memos = ''

        self.action_space = gym.spaces.Discrete(len(actions))
        self.observation_space = gym.spaces.Box(low=0, high=1, 
                                                shape=(len(observations),),
                                                dtype=float)

    def on_message_received(self, msg: IncomingMessage) -> None:
        self.memos += msg.read_string() + '\n'

    def close(self):
        try:
            self.rl_env.close()
            self.unity_env.close()

        except (AttributeError, UnityException, UnityGymException):
            pass

    def reset(self, seed=None):
        self.close()

        try:
            self.unity_env = proivision_unity_env(self.render_, self.attach_, 
                                                  self.autoplay_, self.time_scale_, [self], 
                                                  log_folder=self.log_folder)
            self.rl_env = UnityToGymWrapper(self.unity_env)
        except (UnityException, UnityGymException) as e:
            raise AutoALSException('Unity environment is not starting as expected') from e
        
        return self.rl_env.reset()
        
    def step(self, action):
        try:
            obs, reward, terminated, truncated, info = self.rl_env.step(action)
            info['memos'] = self.memos
            self.memos = ''
            return obs, reward, terminated, truncated, info
        except (UnityException, UnityGymException) as e:
            raise AutoALSException('Unity environment is not responding as expected') from e