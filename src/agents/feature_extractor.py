"""Custom feature extractor for ToolpathEnv2D's Dict observation.

SB3's default ``MultiInputPolicy`` flattens the 2D maps into a long vector and
feeds them to a small MLP, which throws away all spatial structure. This
extractor instead stacks the material/visited/collision maps into image
channels and runs a small CNN, concatenating the scalar/vector observations
afterwards.
"""

import torch as th
import torch.nn as nn
from gymnasium import spaces
from gymnasium.spaces import flatdim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

MAP_KEYS = ("material_map", "visited_map", "collision_map")


class ToolpathCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, cnn_features: int = 128):
        vec_keys = [k for k in observation_space.spaces if k not in MAP_KEYS]
        vec_dim = sum(flatdim(observation_space.spaces[k]) for k in vec_keys)
        super().__init__(observation_space, features_dim=cnn_features + vec_dim)

        self._vec_keys = vec_keys
        n_channels = len(MAP_KEYS)
        map_shape = observation_space.spaces[MAP_KEYS[0]].shape
        assert map_shape is not None
        h, w = map_shape[0], map_shape[1]

        self._cnn = nn.Sequential(
            nn.Conv2d(n_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with th.no_grad():
            n_flatten = self._cnn(th.zeros(1, n_channels, h, w)).shape[1]
        self._linear = nn.Sequential(nn.Linear(n_flatten, cnn_features), nn.ReLU())

    def forward(self, observations: dict) -> th.Tensor:
        maps = th.stack([observations[k] for k in MAP_KEYS], dim=1)  # (B, C, H, W)
        cnn_out = self._linear(self._cnn(maps))
        vecs = [
            observations[k].reshape(observations[k].shape[0], -1)
            for k in self._vec_keys
        ]
        return th.cat([cnn_out, *vecs], dim=1)
