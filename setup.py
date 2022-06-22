# Linux only

from setuptools import setup
import sys

dependencies = [
    'gym-unity>=0.27'
]

# Being too lazy to build several versions of Virtu-ALS at this point
#assert sys.platform == 'linux', 'Only linux is currently supported. It would be pretty easy to support other platforms, contact the developers if you need that'

setup(name='auto-als',
      version='1.0',
      description='OpenAI Gym Reinforcement Learning environment simulating hospital emergency ward',
      author='Vadim Liventsev',
      author_email='v.liventsev@tue.nl',
      url='https://github.com/vadim0x60/auto-als',
      packages=['auto_als', 'auto_als.UnityBuilds'],
      install_requires = dependencies,
      include_package_data=True
     )