from model import AvoidanceExtractor
from env import AvdEnvVecObs
import torch
import time
from collections import deque
import matplotlib.pyplot as plt


class Tester:
    def __init__(self, policy):
        self.policy = policy
        self.env = AvdEnvVecObs(debug=False)
        self.rewards = deque([], maxlen=100)
        self.smooth_reward = []
        self.model = AvoidanceExtractor(self.env.observation_space)

    def eval(self, eval_num=100):
        obs_list = []
        for _ in range(eval_num):
            done = False
            episode_reward = 0
            obs, _ = self.env.reset()
            timestamp_pre = time.time()
            step_cost_list = []
            obs_list.append(obs)
            num_bullet = [0]
            step = 0
            while not done:
                step += 1
                if self.policy == 'random':
                    action = self.env.action_space.sample()
                else:
                    action = 0

                output = self.model(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
                next_obs, reward, done, _, info = self.env.step(action)
                episode_reward += reward
                obs_list.append(next_obs)
                num_bullet.append(info["num_bullet"])
                obs = next_obs

                timestamp_cur = time.time()
                step_cost = (timestamp_cur - timestamp_pre) * 1000
                step_cost_list.append(step_cost)
                timestamp_pre = timestamp_cur

            obs_list.append(0)
            self.rewards.append(episode_reward)
            self.smooth_reward.append(sum(self.rewards)/len(self.rewards))
            print(f"episode reward = {episode_reward}, avg step cost = {sum(step_cost_list)/len(step_cost_list)}")

        plt.clf()
        x = [i for i in range(len(self.smooth_reward))]
        y = self.smooth_reward
        plt.plot(x, y)
        plt.savefig(f'{self.policy}-rewards')
        self.env.close()


if __name__ == '__main__':
    tester = Tester('random')
    tester.eval()
