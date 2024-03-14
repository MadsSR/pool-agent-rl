"""
Gymnasium environments for pool, using FastFiz to simulate the physics of the game.

Avaliable environments:
    - `BaseFastFiz-v0`: Base class for FastFiz.
    - `BaseRLFastFiz-v0`: Base class for FastFiz with reinforcement learning, using initial random table state.
    - `PocketFastFiz-v0`: Subclass of BaseRLFastFiz. Observes if a ball is pocketed.


### Example

Use the environment for training a reinforcement learning agent:

```python
import gymnasium as gym
from stable_baselines3 import PPO
import fastfiz_env # Register the environments


env = gym.make("BaseRLFastFiz-v0", num_balls=2, max_episode_steps=100)

model = PPO("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=10000)

```

"""

from . import utils, envs

from gymnasium.envs.registration import register

register(
    id='BaseFastFiz-v0',
    entry_point="fastfiz_env.envs:BaseFastFiz",
    disable_env_checker=True
)

register(
    id='BaseRLFastFiz-v0',
    entry_point="fastfiz_env.envs:BaseRLFastFiz",
    disable_env_checker=True
)

register(
    id='PocketFastFiz-v0',
    entry_point="fastfiz_env.envs:PocketFastFiz",
    disable_env_checker=True
)
