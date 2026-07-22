"""Frame stacking wrapper for temporal information.

Stacks multiple screenshots together to give the agent
information about recent visual changes.
"""

from __future__ import annotations

from collections import deque
from typing import Any, SupportsFloat, TypeAlias

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium import spaces

# Qontinui environments always expose dict observations and either a plain
# workflow index or a hybrid action dict (see QontinuiEnv in env.py).
_ObsType: TypeAlias = dict[str, Any]
_ActType: TypeAlias = int | dict[str, Any]


class FrameStackWrapper(gym.Wrapper[_ObsType, _ActType, _ObsType, _ActType]):
    """Stack multiple screenshot frames for temporal information.

    Useful for detecting animations, transitions, and temporal patterns.
    The stacked frames are added as a new axis at the beginning of the
    screenshot array.

    Example:
        env = QontinuiEnv(config_path="app.json")
        env = FrameStackWrapper(env, num_frames=4)
        # observation["screenshot"] shape: (4, H, W, C)
    """

    def __init__(self, env: gym.Env[_ObsType, _ActType], num_frames: int = 4):
        """Initialize frame stack wrapper.

        Args:
            env: Base environment
            num_frames: Number of frames to stack
        """
        super().__init__(env)
        self.num_frames = num_frames
        self._frames: deque[npt.NDArray[np.uint8]] = deque(maxlen=num_frames)

        # Update observation space
        old_space = env.observation_space
        if isinstance(old_space, spaces.Dict) and "screenshot" in old_space.spaces:
            screenshot_space = old_space["screenshot"]
            assert screenshot_space.shape is not None
            new_shape = (num_frames,) + screenshot_space.shape

            new_spaces = dict(old_space.spaces)
            new_spaces["screenshot"] = spaces.Box(
                low=0,
                high=255,
                shape=new_shape,
                dtype=np.uint8,
            )
            self.observation_space = spaces.Dict(new_spaces)
        else:
            self.observation_space = old_space

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset and initialize frame stack.

        Args:
            seed: Random seed
            options: Reset options

        Returns:
            Tuple of (observation, info)
        """
        obs, info = self.env.reset(seed=seed, options=options)

        # Initialize frame stack with copies of first frame
        if "screenshot" in obs:
            self._frames.clear()
            for _ in range(self.num_frames):
                self._frames.append(obs["screenshot"].copy())
            obs["screenshot"] = np.stack(list(self._frames), axis=0)

        return obs, info

    def step(
        self, action: Any
    ) -> tuple[dict[str, Any], SupportsFloat, bool, bool, dict[str, Any]]:
        """Step and update frame stack.

        Args:
            action: Action to take

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        obs, reward, terminated, truncated, info = self.env.step(action)

        if "screenshot" in obs:
            self._frames.append(obs["screenshot"].copy())
            obs["screenshot"] = np.stack(list(self._frames), axis=0)

        return obs, reward, terminated, truncated, info
