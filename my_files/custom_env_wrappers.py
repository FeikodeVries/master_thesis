"""Wrapper for augmenting observations by pixel values."""
import collections
import copy
from collections.abc import MutableMapping
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import cv2
from collections import deque
from gymnasium.error import DependencyNotInstalled
import torch


class ResizeObservationandFrameSkip(gym.ObservationWrapper, gym.utils.RecordConstructorArgs):
    """Resize the image observation.

    This wrapper works on environments with image observations. More generally,
    the input can either be two-dimensional (AxB, e.g. grayscale images) or
    three-dimensional (AxBxC, e.g. color images). This resizes the observation
    to the shape given by the 2-tuple :attr:`shape`.
    The argument :attr:`shape` may also be an integer, in which case, the
    observation is scaled to a square of side-length :attr:`shape`.

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.wrappers import ResizeObservation
        >>> env = gym.make("CarRacing-v2")
        >>> env.observation_space.shape
        (96, 96, 3)
        >>> env = ResizeObservation(env, 64)
        >>> env.observation_space.shape
        (64, 64, 3)
    """

    def __init__(self, env: gym.Env, shape: tuple[int, int] | int, frame_skip) -> None:
        """Resizes image observations to shape given by :attr:`shape`.

        Args:
            env: The environment to apply the wrapper
            shape: The shape of the resized observations
        """
        gym.utils.RecordConstructorArgs.__init__(self, shape=shape)
        gym.ObservationWrapper.__init__(self, env)

        if isinstance(shape, int):
            shape = (shape, shape)
        assert len(shape) == 2 and all(x > 0 for x in shape), \
            f"Expected shape to be a 2-tuple of positive integers, got: {shape}"

        self.shape = tuple(shape)

        assert frame_skip > 0, f"Expected the total action frames to be at least 1, got: {frame_skip}"
        self._frame_skip = frame_skip

        assert isinstance(env.observation_space['pixels'], spaces.Box), f"Expected the observation space to be Box, " \
                                                                  f"actual type: {type(env.observation_space['pixels'])}"
        dims = len(env.observation_space['pixels'].shape)
        assert (dims == 2 or dims == 3), f"Expected the observation space to have 2 or 3 dimensions, got: {dims}"

        obs_shape = env.observation_space['pixels'].shape[2:] + self.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        """Updates the observations by resizing the observation to shape given by :attr:`shape`.

        Args:
            observation: The observation to reshape

        Returns:
            The reshaped observations

        Raises:
            DependencyNotInstalled: opencv-python is not installed
        """
        try:
            import cv2
        except ImportError as e:
            raise DependencyNotInstalled(
                "opencv (cv2) is not installed, run `pip install gymnasium[other]`"
            ) from e

        observation = observation['pixels']
        observation = cv2.resize(observation, self.shape[::-1], interpolation=cv2.INTER_AREA)
        # Resize with pytorch to keep correct dimensions
        observation = torch.from_numpy(observation).permute(2, 0, 1).numpy()

        return observation

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = self.env.reset(seed=seed, options=options)

        return self.observation(obs), info

    def step(self, action):
        reward = 0
        for _ in range(self._frame_skip):
            obs, r, done, truncated, info = self.env.step(action)
            reward += r
            if done or truncated:
                break
        return self.observation(obs), reward, done, truncated, info


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=float
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        obs = obs / 255.
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        obs = obs / 255.
        self._frames.append(obs)
        return self._get_obs(), reward, done, truncated, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(np.array(self._frames), axis=0)


class FlattenFirst(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        shape = self.observation_space.shape
        self.observation_space = spaces.Box(shape=[shape[0] * shape[1], shape[2], shape[3]],
                                            low=0, high=255, dtype=np.uint8)

    def observation(self, observation):
        return observation
