from gym.envs.registration import register
from pathlib import Path
from gym_unity.envs import UnityGymException

UnityError = UnityGymException

register(
    id='Auto-ALS-v0',
    entry_point='auto_als.envs:AutoALS',
    nondeterministic=True
)