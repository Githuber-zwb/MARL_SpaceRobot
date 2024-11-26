import torch.nn as nn
import torch
import numpy as np
from .util import init, get_clones

"""MLP modules."""

class MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_size, layer_N, use_orthogonal, use_ReLU):
        super(MLPLayer, self).__init__()
        self._layer_N = layer_N

        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.fc1 = nn.Sequential(
            init_(nn.Linear(input_dim, hidden_size)), active_func, nn.LayerNorm(hidden_size))
        # self.fc1 = nn.Sequential(
            # init_(nn.Linear(input_dim, hidden_size)), active_func)
        # self.fc_h = nn.Sequential(init_(
        #     nn.Linear(hidden_size, hidden_size)), active_func, nn.LayerNorm(hidden_size))
        # self.fc2 = get_clones(self.fc_h, self._layer_N)
        self.fc2 = nn.ModuleList([nn.Sequential(init_(
            nn.Linear(hidden_size, hidden_size)), active_func, nn.LayerNorm(hidden_size)) for i in range(self._layer_N)])
        # self.fc2 = nn.ModuleList([nn.Sequential(init_(
        #     nn.Linear(hidden_size, hidden_size)), active_func) for i in range(self._layer_N)])


    def forward(self, x):
        x = self.fc1(x)
        for i in range(self._layer_N):
            x = self.fc2[i](x)
        return x


class MLPBase(nn.Module):
    def __init__(self, args, obs_shape, cat_self=True, attn_internal=False):
        super(MLPBase, self).__init__()

        self._use_feature_normalization = args.use_feature_normalization
        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU
        self._stacked_frames = args.stacked_frames
        self._layer_N = args.layer_N
        self.hidden_size = args.hidden_size

        obs_dim = obs_shape[0]

        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_dim)

        self.mlp = MLPLayer(obs_dim, self.hidden_size,
                              self._layer_N, self._use_orthogonal, self._use_ReLU)

    def forward(self, x):
        if self._use_feature_normalization:
            x = self.feature_norm(x)

        x = self.mlp(x)

        return x
    
class TransBase(nn.Module):
    def __init__(self, args, obs_shape, cat_self=True, attn_internal=False):
        super(TransBase, self).__init__()

        self._use_feature_normalization = args.use_feature_normalization
        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU
        self._stacked_frames = args.stacked_frames
        self._layer_N = args.layer_N
        self._num_agents = args.num_agents
        self.hidden_size = args.hidden_size
        self._num_envs = args.n_rollout_threads

        obs_dim = obs_shape[0]

        assert obs_dim % self._num_agents == 0
        self._single_obs_dim = obs_dim // self._num_agents

        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_dim)

        self.mlp1 = nn.Linear(self._single_obs_dim, self.hidden_size)

        self.enc = nn.TransformerEncoderLayer(self.hidden_size, 4, self.hidden_size // 4)

        self.mlp = MLPLayer(obs_dim, self.hidden_size,
                              self._layer_N, self._use_orthogonal, self._use_ReLU)

    def forward(self, x):
        if self._use_feature_normalization:
            x = self.feature_norm(x)
        
        x = x.reshape(-1, self._num_agents, self._single_obs_dim)
        x = self.enc(self.mlp1(x))
        x = torch.mean(x, dim=1)

        # x = self.mlp(x)

        return x