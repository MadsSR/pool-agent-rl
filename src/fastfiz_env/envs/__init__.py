from .BaseFastFiz import BaseFastFiz
from .BaseRLFastFiz import BaseRLFastFiz

from gymnasium import register

register(
    id='BaseFastFiz-v0',
    entry_point='fastfiz_env.envs:BaseFastFiz',
)

register(
    id='BaseRLFastFiz-v0',
    entry_point='fastfiz_env.envs:BaseRLFastFiz',
)
