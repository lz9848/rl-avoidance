import struct
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os
import socket
import subprocess


class AvdEnvVecObs(gym.Env):

    def __init__(self, eval=False, debug=False):
        super(AvdEnvVecObs, self).__init__()
        self.eval = eval
        self.server_addr = "127.0.0.1"

        self.game_path = "./avdgym"
        if self.eval:
            self.game_exe = "rl-eval-15.exe"
        else:
            self.game_exe = "rl-hitmode-8.exe"

        self.debug = debug
        self.normalize = not self.debug

        if not self.debug:
            self._lanch_game()
        else:
            self.udpsocket, self.port = self._create_udp_socket(self.server_addr)
            self.client_addr = ("127.0.0.1", 11450)

        self.player_state_num = 9
        self.bullet_state_num = 6
        self.bullet_to_idx = dict()
        self.max_bullet_info = 50
        self.available_id = [i for i in range(self.max_bullet_info, -1, -1)]

        self.stack_frame = 1
        self._last_obs = np.zeros((self.stack_frame, self.player_state_num + self.bullet_state_num * self.max_bullet_info)).astype(np.float32)

        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(low=-2.0, high=2.0, shape=(self.stack_frame, self.player_state_num + self.bullet_state_num * self.max_bullet_info,), dtype=np.float32)
        self.action_repeat = 1
        self.fix_obs = True

        self.max_episode_step = 512 if not self.eval else 1300
        self.episode_step = 0

        self.hit_count = 0

    def step(self, act: int):
        obs = None
        done = False
        total_reward = 0
        truncate = False
        info = {}
        self.episode_step += 1
        for _ in range(self.action_repeat):
            self._execute_action(act)
            obs, info = self._get_game_state_fix()
            hit = info["hit"]
            episode_state = info["episode_state"]
            reward = self.get_reward(hit)
            total_reward += reward
            if episode_state == 0 or self.episode_step >= self.max_episode_step:
                if episode_state != 0:
                    info["episode_state"] = 2
                self._execute_action(0)
                done = True
                break
            else:
                done = False
        self._last_obs[1:, :] = self._last_obs[:-1, :]
        self._last_obs[0] = obs
        return self._last_obs.copy(), total_reward, done, truncate, info

    def reset(self, **kwargs):
        self._execute_action(6)
        while True:
            obs, info = self._get_game_state_fix()
            episode_state = info["episode_state"]
            if episode_state == 1:
                break
        self._last_obs = np.stack([obs] * self.stack_frame)
        self.episode_step = 0
        return self._last_obs.copy(), info

    def close(self):
        self.game_proc.kill()

    def render(self):
        pass

    def _lanch_game(self):
        self.udpsocket, self.port = self._create_udp_socket(self.server_addr)
        os.chdir(self.game_path)
        self.game_proc = subprocess.Popen([self.game_exe, str(self.port)])
        msg, self.client_addr = self.udpsocket.recvfrom(1024)
        print(f"{self.client_addr}: {msg.decode()}")
        os.chdir('../')

    def get_reward(self, hit):
        if hit == 0:
            reward = 0.1
            self.hit_count = 0
        else:
            self.hit_count += 1
            reward = -(self.hit_count ** 2)
        return reward

    def _create_udp_socket(self, addr: str):
        udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if self.debug:
            udp_socket.bind((addr, 11451))
        else:
            udp_socket.bind((addr, 0))
        port = udp_socket.getsockname()[1]
        return udp_socket, port

    def _execute_action(self, code):
        """
        ↖↑↗   213
        ←-→   405
        6 for release all and restart
        """
        data = struct.pack("!b", code + 1)
        self.udpsocket.sendto(data, self.client_addr)

    # def _get_game_state(self, normalize=True):
    #     info = {}
    #     bullet_num = 0
    #     hit = 1
    #     player_info_num = 7
    #     player_info_len = 26
    #     bullet_info_len = 20
    #     bullet_obs_num = 4
    #     obs = np.zeros((player_info_num + bullet_obs_num,)).astype(np.float32)
    #     data, address = self.udpsocket.recvfrom(1024)
    #     episode_state = struct.unpack('b', data[0:1])[0]
    #     if episode_state == 1:
    #         ptr = 1
    #         player_x1, player_y1, player_x2, player_y2, hspeed, vspeed, djump, predict = struct.unpack('ffffffbb', data[ptr:ptr + player_info_len])
    #         ptr += player_info_len
    #         # print(f"从 {address} 接收到消息: 玩家坐标=({player_x1, player_y1})({player_x2, player_y2}), djump={djump}, predict={predict}")
    #         player_state = np.array([player_x1, player_y1, player_x2, player_y2, hspeed, vspeed, djump])
    #
    #         bullet_num = min(self.max_bullet_info, (len(data) - ptr) // bullet_info_len)
    #         bullets = np.frombuffer(data[ptr:], dtype=np.float32).reshape((-1, 5))[:bullet_num]
    #         bullet_state = bullets[:, 1:].flatten()
    #         obs = np.concatenate((player_state, bullet_state))
    #
    #         if normalize:
    #             obs[:4] /= 500
    #             obs[4] /= 500
    #             obs[5] /= 500
    #             obs[6] /= 2
    #             obs[7::5] /= 500  # x 坐标归一化
    #             obs[8::5] /= 500  # y 坐标归一化
    #             obs[9::5] /= 500  # hspeed 归一化
    #             obs[10::5] /= 500  # vspeed 归一化
    #             obs[11::5] /= 1000  # spr_id 归一化
    #
    #     info["predict"] = predict
    #     info["num_bullet"] = bullet_num
    #     return obs, episode_state, info

    def _get_game_state_fix(self):
        info = {}
        bullet_num = 0
        hit = 0
        player_info_num = 9
        player_info_len = 27
        bullet_info_num = 6
        bullet_info_len = 28
        bullet_obs_num = 6
        bullet_obs_len = 20
        obs_len = player_info_num + self.max_bullet_info * bullet_obs_num
        obs = np.zeros((obs_len,)).astype(np.float32)
        self._clear_buffer()
        data, address = self.udpsocket.recvfrom(2048)
        episode_state = struct.unpack('b', data[0:1])[0]

        if episode_state == 1:
            ptr = 1
            player_x1, player_y1, player_x2, player_y2, hspeed, vspeed, pjump, djump, hit = struct.unpack('ffffffbbb', data[ptr:ptr + player_info_len])
            ptr += player_info_len
            # print(f"从 {address} 接收到消息: 玩家坐标=({player_x1, player_y1})({player_x2, player_y2}), djump={djump}, predict={predict}")
            obs[:player_info_num] = np.array([player_x1, player_y1, player_x2, player_y2, hspeed, vspeed, pjump, djump, hit])

            bullet_num = min(self.max_bullet_info, (len(data) - ptr) // bullet_info_len)
            bullets = np.frombuffer(data[ptr:], dtype=np.float32).reshape((-1, 7))[:bullet_num]
            # 检测销毁弹幕并空出索引位置
            cur_bullet_set = {bullet[0] for bullet in bullets}
            destoryed_bullet = [bullet_id for bullet_id in self.bullet_to_idx if bullet_id not in cur_bullet_set]
            for bullet_id in destoryed_bullet:
                self.available_id.append(self.bullet_to_idx[bullet_id])
                self.bullet_to_idx.pop(bullet_id)

            # 添加新弹幕状态
            for bullet in bullets:
                bullet_id = bullet[0]
                if bullet_id not in self.bullet_to_idx:
                    if self.available_id:
                        idx = self.available_id.pop()  # 从可用列表中取出索引
                        self.bullet_to_idx[bullet_id] = idx
                    else:
                        # 处理索引已满的情况，可能需要重新分配或扩展
                        continue

                idx = self.bullet_to_idx[bullet_id]
                obs[player_info_num + idx * bullet_obs_num: player_info_num + (idx + 1) * bullet_obs_num] = bullet[1:]

            if self.normalize:
                obs[:4] /= 500
                obs[4] /= 500
                obs[5] /= 500
                obs[7] /= 2
                obs[player_info_num + 1::6] /= 500  # x 坐标归一化
                obs[player_info_num + 2::6] /= 500  # y 坐标归一化
                obs[player_info_num + 3::6] /= 500  # hspeed 归一化
                obs[player_info_num + 4::6] /= 500  # vspeed 归一化
                obs[player_info_num + 5::6] /= 500  # 相对距离 归一化
                obs[player_info_num + 6::6] /= 360  # 相对方向 归一化

            assert len(obs) == obs_len, print(obs)
        info["hit"] = hit
        info["num_bullet"] = bullet_num
        info["episode_state"] = episode_state
        return obs, info

    def _clear_buffer(self, ):
        # 设置为非阻塞模式
        self.udpsocket.setblocking(False)
        while True:
            try:
                # 尝试读取数据，丢弃所有旧的数据包
                _, _ = self.udpsocket.recvfrom(2048)
            except BlockingIOError:
                # 没有更多数据时退出循环，缓冲区已清空
                break
        # 清空缓冲区后，重新设置为阻塞模式
        self.udpsocket.setblocking(True)


# if __name__ == '__main__':
#     udp_socket, _ = AvdEnvVecObs._create_udp_socket('127.0.0.1')
#     code = 1
#     data = struct.pack("!b", code + 1)
#     print(f"send{data}")
#     udp_socket.sendto(data, ("127.0.0.1", 11450))
