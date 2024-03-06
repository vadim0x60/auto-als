# Linux only

from setuptools import setup
from pathlib import Path

HERE = Path(__name__).parent

setup(name='auto-als',
      version='1.4',
      description='OpenAI Gym Reinforcement Learning environment simulating hospital emergency ward',
      author='Vadim Liventsev',
      author_email='v.liventsev@tue.nl',
      url='https://github.com/vadim0x60/auto-als',
      packages=['auto_als'],
      install_requires=(HERE / 'auto_als' / 'requirements.txt').read_text().splitlines(),
     )