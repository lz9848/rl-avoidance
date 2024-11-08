from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
import numpy as np
import gymnasium as gym
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from env import AvdEnvVecObs
from stable_baselines3.common.callbacks import BaseCallback
from model import CustomActorCriticPolicy
import os

gamma = 0.95
gae_lambda = 0.95
rollout_len = 512
parallel_num = 8
iterations = 5000
features_dim = 512
exp_name = f"PPO-{features_dim}-{gamma}-{gae_lambda}"


def make_env():
    env = AvdEnvVecObs()
    env = Monitor(env)  # 包裹在 Monitor 中，用于记录奖励和回合长度
    return env


class Callback_PPO(BaseCallback):
    def __init__(self, verbose=0):
        super(Callback_PPO, self).__init__(verbose)
        self.reboot_interval = 2457600

    def _on_step(self) -> bool:
        return True

    def _init_callback(self) -> None:
        self.num_timesteps = self.model.num_timesteps
        self.reboot_count = self.model.num_timesteps // self.reboot_interval

    def _on_rollout_start(self):
        print("###start new rollout, reset env###")
        self.model._last_obs = self.training_env.reset()
        self.model._last_episode_starts = np.ones((self.model.env.num_envs,), dtype=bool)

    def _on_rollout_end(self):
        if self.model.num_timesteps // self.reboot_interval > self.reboot_count:
            print("###save model###")
            self.model.save(f"{exp_name}-{int(self.model.num_timesteps//1e6)}M")
            print("###reboot env###")
            self.reboot_count += 1
            self.training_env.close()
            self.model.set_env(SubprocVecEnv([make_env for _ in range(parallel_num)]), force_reset=False)


def train_avd_PPO():
    policy_kwargs = dict(
        net_arch=[512, 256, 256],
        features_dim=features_dim,
        model_dim=128,
    )

    custom_hyperparams = {
        "policy_kwargs": policy_kwargs,
        "learning_rate": 0.0001,
        "n_steps": rollout_len,
        "batch_size": 512,
        "n_epochs": 3,
        "gae_lambda": gae_lambda,
        "gamma": gamma,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 1.0,
        "max_grad_norm": 5.0,
        "tensorboard_log": f"./log/avoidance/{exp_name}",
        "verbose": 1,
    }

    vec_env = SubprocVecEnv([make_env for _ in range(parallel_num)])
    if os.path.exists("model.zip"):
        model = PPO.load("eval", env=vec_env, print_system_info=True)
    else:
        model = PPO(CustomActorCriticPolicy, vec_env, **custom_hyperparams)

    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"模型参数总数: {total_params}")

    callback = Callback_PPO(verbose=1)
    callback.init_callback(model)

    model.learn(total_timesteps=(rollout_len * parallel_num * iterations), callback=callback, reset_num_timesteps=False)
    model.save(f"{exp_name}-final")


def train_atari():
    train_game = "Breakout-v4"
    rollout_len = 2048
    parallel_num = 4

    def make_atari_env():
        env = gym.make(train_game)
        env = Monitor(env)  # 包裹在 Monitor 中，用于记录奖励和回合长度
        return env

    vec_env = SubprocVecEnv([make_atari_env for _ in range(parallel_num)])

    custom_hyperparams = {
        "learning_rate": 0.0001,
        "n_steps": rollout_len//parallel_num,
        "batch_size": 128,
        "n_epochs": 5,
        "gamma": 0.99,
        "clip_range": 0.2,
        "ent_coef": 0.001,
        "tensorboard_log": f"./log/{train_game}",
        "verbose": 2,
    }

    model = RecurrentPPO("CnnLstmPolicy", vec_env, **custom_hyperparams)
    model.learn(total_timesteps=(rollout_len*1000))


if __name__ == '__main__':
    train_avd_PPO()
