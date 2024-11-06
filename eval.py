from stable_baselines3 import PPO
from env import AvdEnvVecObs


def evaluate(num_episodes=10):
    env = AvdEnvVecObs(eval=True)
    model = PPO.load("eval", env=env, print_system_info=True)
    vec_env = model.get_env()
    success_cnt = 0
    eval_cnt = 0

    for _ in range(num_episodes):
        eval_cnt += 1
        obs = vec_env.reset()
        done = False
        while not done:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            if info[0]["episode_state"] == 2:
                success_cnt += 1
            if done:
                print(f"{success_cnt}/{eval_cnt}")


if __name__ == '__main__':
    evaluate(100)
