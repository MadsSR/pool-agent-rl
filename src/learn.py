from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from envs.BaseRLFastFiz import BaseRLFastFiz
import gymnasium as gym
import fastfiz as ff


gym.register(id="BaseRLFastFiz-v0", entry_point=BaseRLFastFiz,
             kwargs={"num_balls": 2}, max_episode_steps=100)


def make_env():
    return gym.make("BaseRLFastFiz-v0")


LOGS_DIR = "logs/"
TB_LOGS_DIR = "logs/tb_logs/"
MODEL_DIR = "models/"
BEST_MODEL_DIR = "models/best_models/"
MODEL_NAME = "ppo-v1"


env = VecNormalize(make_vec_env(make_env, n_envs=4),
                   training=True, norm_obs=True, norm_reward=True)
model = PPO(MlpPolicy, env, verbose=1, tensorboard_log=TB_LOGS_DIR)


checkpoint_callback = CheckpointCallback(
    save_freq=50_000, save_path=MODEL_DIR, save_replay_buffer=True, save_vecnormalize=True)


eval_callback = EvalCallback(eval_env=env, best_model_save_path=BEST_MODEL_DIR,
                             log_path=LOGS_DIR, eval_freq=5_000, deterministic=True, render=False)

try:
    model.learn(total_timesteps=50_000_000, callback=[
        checkpoint_callback, eval_callback], tb_log_name=MODEL_NAME)
except KeyboardInterrupt:
    print("Training interrupted")
    model.save(MODEL_DIR + MODEL_NAME)
else:
    model.save(MODEL_DIR + MODEL_NAME)
    print("Training finished")
