from pathlib import Path
import sys
from typing import List
import uuid

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.exception import UnityEnvironmentException, UnityWorkerInUseException
from mlagents_envs.side_channel.side_channel import (
    SideChannel,
    IncomingMessage,
    OutgoingMessage,
)

from mlagents_envs import logging_util
logger = logging_util.get_logger(__name__)

from auto_als.unity_gym_env import UnityToGymWrapper, UnityGymException

from tenacity import retry
from tenacity.wait import wait_exponential
from tenacity.retry import retry_if_exception_type
from tenacity import stop_after_attempt

BUILDS_PATH = Path(__file__).parent.parent.resolve() / 'UnityBuilds'
ORIGIN = 'https://github.com/vadim0x60/virtu-als-plus/releases/download/1.2.1/'
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
    elif sys.platform == 'darwin':
        return 'virtu-als2018.app'
    else:
        raise UnityGymException(f'Unsupported platform: {sys.platform}')

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
def proivision_unity_env(render=False, attach=False, autoplay=True):
    if attach:
        unity_env = UnityEnvironment()
    else:
        build = required_build()
        launcher = str(BUILDS_PATH / build)

        additional_args = []
        if autoplay:
            additional_args.append('--autoplay')

        unity_env = UnityEnvironment(launcher, no_graphics=not render, 
                                     additional_args=additional_args)
    return unity_env

class MemoChannel(SideChannel):

    def __init__(self) -> None:
        super().__init__(SIDE_CHANNEL)

    def on_message_received(self, msg: IncomingMessage) -> None:
        """
        Note: We must implement this method of the SideChannel interface to
        receive messages from Unity
        """
        # We simply read a string from the message and print it.
        print(msg.read_string())

    def send_string(self, data: str) -> None:
        # Add the string to an OutgoingMessage
        msg = OutgoingMessage()
        msg.write_string(data)
        # We call this method to queue the data we want to send
        super().queue_message_to_send(msg)

class AutoALS(UnityToGymWrapper, SideChannel):
    def __init__(self, attach=False, render='auto', autoplay=True):
        if render == 'auto':
            render = False if autoplay else True
        
        assert autoplay or render, 'Hybrid mode requires render to be set to True'

        self.attach_ = attach
        self.render_ = render
        self.memos = ''

        unity_env = proivision_unity_env(render, attach, autoplay)
        UnityToGymWrapper.__init__(self, unity_env)
        SideChannel.__init__(self, SIDE_CHANNEL)

    def on_message_received(self, msg: IncomingMessage) -> None:
        self.memos += msg.read_string()

    def reset(self, seed=None):
        self.memos = ''

        try:
            return super().reset()
        except (UnityEnvironmentException, UnityGymException):
            self._env.close()
            action_taken = 'reattach to' if self.attach_ else 'restart'
            logger.warn(f'Built-in reset functionality failed. Had to {action_taken} the environment')
            super().__init__(proivision_unity_env(self.attach_, self.render_))
            return super().reset()
        
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info['memos'] = self.memos
        return obs, reward, terminated, truncated, info