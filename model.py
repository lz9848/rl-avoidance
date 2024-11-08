import torch
import torch.nn as nn
import gymnasium as gym
import math
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class AttenionBlock(nn.Module):
    def __init__(self, encode_dim, model_dim):
        super(AttenionBlock, self).__init__()
        self.encode_dim = encode_dim
        self.Query = nn.Linear(encode_dim, model_dim)
        self.Key = nn.Linear(encode_dim, model_dim)
        self.Value = nn.Linear(encode_dim, encode_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, player_encode, bullet_encode):
        q = self.Query(player_encode)
        k = self.Key(bullet_encode)
        v = self.Value(bullet_encode)
        score = q @ k.transpose(-2, -1) / math.sqrt(self.encode_dim)
        score = self.softmax(score) @ v
        return score


class MultiHeadAttention(nn.Module):
    def __init__(self, encode_dim, model_dim, n_head):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_model = model_dim
        self.w_q = nn.Linear(encode_dim, self.d_model * self.n_head)
        self.w_k = nn.Linear(encode_dim, self.d_model * self.n_head)
        self.w_v = nn.Linear(encode_dim, encode_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, seq1, seq2, mask=None):
        batch, seq_num, dimension = seq2.shape
        q, k, v = self.w_q(seq1), self.w_k(seq2), self.w_v(seq2)
        n_v = dimension // self.n_head

        q = q.view(batch, 1, self.n_head, self.d_model).permute(0, 2, 1, 3)
        k = k.view(batch, seq_num, self.n_head, self.d_model).permute(0, 2, 1, 3)
        v = v.view(batch, seq_num, self.n_head, n_v).permute(0, 2, 1, 3)

        score = q @ k.transpose(2, 3) / math.sqrt(self.d_model)
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)
        score = self.softmax(score) @ v
        score = score.permute(0, 2, 1, 3).contiguous().view(batch, 1, dimension)

        return score


class AvoidanceExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces, features_dim: int = 256, model_dim: int = 128, n_head: int = 4):
        super(AvoidanceExtractor, self).__init__(observation_space, features_dim)
        self.player_size = 9
        self.bullet_size = 6
        self.seq_num = observation_space.shape[0]
        self.n_head = n_head
        self.player_encoder = nn.Sequential(
            nn.Linear(self.player_size * self.seq_num, features_dim),
            nn.ReLU(True),
            nn.Linear(features_dim, features_dim),
            nn.ReLU(True),
        )
        self.bullet_encoder = nn.Sequential(
            nn.Linear(self.bullet_size * self.seq_num, features_dim),
            nn.ReLU(True),
            nn.Linear(features_dim, features_dim),
            nn.ReLU(True),
        )
        # self.attention = AttenionBlock(encode_dim=features_dim, model_dim=model_dim)
        self.attention = MultiHeadAttention(encode_dim=features_dim, model_dim=model_dim, n_head=n_head)
        self.relu = nn.ReLU()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        seq_len = observations.shape[1]
        player_obs = observations[:, :, :self.player_size].reshape(batch_size, 1, -1)
        bullet_obs = observations[:, :, self.player_size:].reshape((batch_size, seq_len, -1, self.bullet_size)).permute(0, 2, 1, 3).flatten(2, 3)
        mask = bullet_obs.any(-1).unsqueeze(1).unsqueeze(1).repeat(1, self.n_head, 1, 1).float()
        player_encode = self.player_encoder(player_obs)
        bullet_encode = self.bullet_encoder(bullet_obs)
        attention = self.attention(player_encode, bullet_encode, mask)
        output = player_encode + attention

        empty_bullet_obs = mask.sum(-1).sum(1).unsqueeze(1) == 0.0
        output = torch.where(empty_bullet_obs, player_encode, output)
        output = output.squeeze(1)

        assert not torch.isnan(output).any(), print(output)
        return output


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, features_dim, model_dim, **kwargs):
        super(CustomActorCriticPolicy, self).__init__(
            observation_space, action_space, lr_schedule,
            features_extractor_class=AvoidanceExtractor,
            features_extractor_kwargs=dict(features_dim=features_dim, model_dim=model_dim),
            **kwargs
        )
