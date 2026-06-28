import numpy as np
import torch as th
from gymnasium import spaces

from agents.feature_extractor import ToolpathCombinedExtractor
from env.toolpath_env import ToolpathEnv2D


def test_extractor_forward():
    env = ToolpathEnv2D(config={"grid_size": 64, "max_steps": 50})
    obs, _ = env.reset(seed=0)

    obs_space = env.observation_space
    assert isinstance(obs_space, spaces.Dict)
    extractor = ToolpathCombinedExtractor(obs_space, cnn_features=64)
    assert extractor.features_dim == 64 + 4  # cnn_features + 4 scalar/vector dims

    # Batch of 2 identical observations
    batch = {
        k: th.as_tensor(np.stack([obs[k], obs[k]])).float() for k in obs
    }
    out = extractor(batch)
    assert out.shape == (2, extractor.features_dim)
    env.close()


if __name__ == "__main__":
    test_extractor_forward()
    print("Policy extractor test passed!")
