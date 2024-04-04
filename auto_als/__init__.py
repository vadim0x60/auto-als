from gymnasium.envs.registration import register

from auto_als.envs import AutoALSException
from auto_als.api import actions, observations

register(
    id='Auto-ALS-v0',
    entry_point='auto_als.envs:AutoALS',
    nondeterministic=True
)