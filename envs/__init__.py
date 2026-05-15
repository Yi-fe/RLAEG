from gym.envs.registration import register
import gym
import numpy as np
np.random.seed(123)


def register(id, entry_point, force=True):
    env_specs = gym.envs.registry.env_specs
    if id in env_specs.keys():
        if not force:
            return
        del env_specs[id]
    gym.register(
        id=id,
        entry_point=entry_point,
    )


# Register modified versions of existing environments
register(
    id='malware-score-v0',
    entry_point='envs.env:Mal_Score_Env'
)

register(
    id='malware-label-v0',
    entry_point='envs.env:Mal_Label_Env'
)
