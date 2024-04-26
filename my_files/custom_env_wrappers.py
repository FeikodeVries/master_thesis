"""Wrapper for augmenting observations by pixel values."""
import collections
import copy
from collections.abc import MutableMapping
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import cv2
import torch
import pathlib
import os
import matplotlib.pyplot as plt
import cleanrl.my_files.datahandling as dh
from cleanrl.my_files.active_icitris import active_iCITRISVAE

STATE_KEY = "state"


class CausalWrapper(gym.ObservationWrapper):
    """Augment observations by pixel values.

    Observations of this wrapper will be dictionaries of images.
    You can also choose to add the observation of the base environment to this dictionary.
    In that case, if the base environment has an observation space of type :class:`Dict`, the dictionary
    of rendered images will be updated with the base environment's observation. If, however, the observation
    space is of type :class:`Box`, the base environment's observation (which will be an element of the :class:`Box`
    space) will be added to the dictionary under the key "state".

    You can also grayscale the obtained images and resize them

    """

    def __init__(
        self,
        env: gym.Env,
        shape: Union[tuple, int],
        pixels_only: bool = True,
        render_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
        pixel_keys: Tuple[str, ...] = ("pixels",),
        rgb: bool = False,
        keep_dim: bool = False,
        batch_size: int = 200,
        causal: bool = True,
        latents: int = 32,
        causal_vars: int = 6
    ):
        """Initializes a new pixel Wrapper.

        Args:
            env: The environment to wrap.
            pixels_only (bool): If ``True`` (default), the original observation returned
                by the wrapped environment will be discarded, and a dictionary
                observation will only include pixels. If ``False``, the
                observation dictionary will contain both the original
                observations and the pixel observations.
            render_kwargs (dict): Optional dictionary containing that maps elements of ``pixel_keys``to
                keyword arguments passed to the :meth:`self.render` method.
            pixel_keys: Optional custom string specifying the pixel
                observation's key in the ``OrderedDict`` of observations.
                Defaults to ``(pixels,)``.

        Raises:
            AssertionError: If any of the keys in ``render_kwargs``do not show up in ``pixel_keys``.
            ValueError: If ``env``'s observation space is not compatible with the
                wrapper. Supported formats are a single array, or a dict of
                arrays.
            ValueError: If ``env``'s observation already contains any of the
                specified ``pixel_keys``.
            TypeError: When an unexpected pixel type is used
        """
        super().__init__(env)
        # START MYCODE
        # Init for grayscaling and resizing
        if isinstance(shape, int):
            shape = (shape, shape)

        self.keep_dim = keep_dim
        self.shape = tuple(shape)
        self.rgb = rgb

        # Updating the VAE
        self.batch_size = batch_size
        self.t = 0
        self.causal = causal
        # Data handling
        self.datahandling = dh.DataHandling()
        self.observations = []
        self.causal_vars = causal_vars
        self.num_latents = latents
        # END MYCODE

        # Avoid side-effects that occur when render_kwargs is manipulated
        render_kwargs = copy.deepcopy(render_kwargs)
        self.render_history = []

        if render_kwargs is None:
            render_kwargs = {}

        for key in render_kwargs:
            assert key in pixel_keys, (
                "The argument render_kwargs should map elements of "
                "pixel_keys to dictionaries of keyword arguments. "
                f"Found key '{key}' in render_kwargs but not in pixel_keys."
            )

        default_render_kwargs = {}
        if not env.render_mode:
            raise AttributeError(
                "env.render_mode must be specified to use PixelObservationWrapper:"
                "`gym.make(env_name, render_mode='rgb_array')`."
            )

        for key in pixel_keys:
            render_kwargs.setdefault(key, default_render_kwargs)

        wrapped_observation_space = env.observation_space

        if isinstance(wrapped_observation_space, spaces.Box):
            self._observation_is_dict = False
            invalid_keys = {STATE_KEY}
        elif isinstance(wrapped_observation_space, (spaces.Dict, MutableMapping)):
            self._observation_is_dict = True
            invalid_keys = set(wrapped_observation_space.spaces.keys())
        else:
            raise ValueError("Unsupported observation space structure.")

        if not pixels_only:
            # Make sure that now keys in the `pixel_keys` overlap with
            # `observation_keys`
            overlapping_keys = set(pixel_keys) & set(invalid_keys)
            if overlapping_keys:
                raise ValueError(
                    f"Duplicate or reserved pixel keys {overlapping_keys!r}."
                )

        if pixels_only:
            self.observation_space = spaces.Dict()
        elif self._observation_is_dict:
            self.observation_space = copy.deepcopy(wrapped_observation_space)
        else:
            self.observation_space = spaces.Dict({STATE_KEY: wrapped_observation_space})

        # Extend observation space with pixels.
        self.env.reset()
        pixels_spaces = {}
        for pixel_key in pixel_keys:
            pixels = self._render(**render_kwargs[pixel_key])
            pixels: np.ndarray = pixels[-1] if isinstance(pixels, List) else pixels

            if not hasattr(pixels, "dtype") or not hasattr(pixels, "shape"):
                raise TypeError(
                    f"Render method returns a {pixels.__class__.__name__}, but an array with dtype and shape is expected."
                    "Be sure to specify the correct render_mode."
                )

            if np.issubdtype(pixels.dtype, np.integer):
                low, high = (0, 255)
            elif np.issubdtype(pixels.dtype, np.float):
                low, high = (-float("inf"), float("inf"))
            else:
                raise TypeError(pixels.dtype)

            # START MYCODE
            # Convert observation to grayscale and resize
            # if self.causal:
            #     pixels_space = spaces.Box(shape=(self.citris.hparams.num_latents, self.citris.hparams.num_causal_vars),
            #                               low=-float("inf"), high=float("inf"), dtype=np.float32)
            # # else:
            if self.rgb:
                pixels_space = spaces.Box(shape=(self.shape[0], self.shape[1], 3),
                                          low=low, high=high, dtype=pixels.dtype)
            else:
                if self.keep_dim:
                    pixels_space = spaces.Box(shape=(self.shape[0], self.shape[1], 1),
                                              low=low, high=high, dtype=pixels.dtype)
                else:
                    pixels_space = spaces.Box(shape=(self.shape[0], self.shape[1]), low=low,
                                              high=high, dtype=pixels.dtype)

            pixels_spaces[pixel_key] = pixels_space
            # END MYCODE

        self.observation_space.spaces.update(pixels_spaces)

        self._pixels_only = pixels_only
        self._render_kwargs = render_kwargs
        self._pixel_keys = pixel_keys

    def observation(self, observation):
        """Updates the observations with the pixel observations and converts them to grayscale and resizes.

        Args:
            observation: The observation to add pixel observations for

        Returns:
            The updated, grayscale pixel observations
        """
        # START MYCODE
        pixel_observation = self._add_pixel_observation(observation)
        for key, value in pixel_observation.items():
            if not self.rgb:
                pixel_observation[key] = cv2.cvtColor(pixel_observation[key], cv2.COLOR_RGB2GRAY)
                if self.keep_dim:
                    pixel_observation[key] = np.expand_dims(pixel_observation[key], -1)
            pixel_observation[key] = cv2.resize(pixel_observation[key], self.shape[::-1], interpolation=cv2.INTER_AREA)
            if observation.ndim == 2:
                pixel_observation[key] = np.expand_dims(pixel_observation[key], -1)

        return pixel_observation
        # END MYCODE

    def _add_pixel_observation(self, wrapped_observation):
        if self._pixels_only:
            observation = collections.OrderedDict()
        elif self._observation_is_dict:
            observation = type(wrapped_observation)(wrapped_observation)
        else:
            observation = collections.OrderedDict()
            observation[STATE_KEY] = wrapped_observation

        pixel_observations = {
            pixel_key: self._render(**self._render_kwargs[pixel_key])
            for pixel_key in self._pixel_keys
        }

        observation.update(pixel_observations)

        return observation

    def render(self, *args, **kwargs):
        """Renders the environment."""
        render = self.env.render(*args, **kwargs)
        if isinstance(render, list):
            render = self.render_history + render
            self.render_history = []
        return render

    def _render(self, *args, **kwargs):
        render = self.env.render(*args, **kwargs)
        if isinstance(render, list):
            self.render_history += render
        return render


class ActionWrapper(gym.ActionWrapper):
    """Superclass of wrappers that can modify the action before :meth:`env.step`.

    If you would like to apply a function to the action before passing it to the base environment,
    you can simply inherit from :class:`ActionWrapper` and overwrite the method :meth:`action` to implement
    that transformation. The transformation defined in that method must take values in the base environment’s
    action space. However, its domain might differ from the original action space.
    In that case, you need to specify the new action space of the wrapper by setting :attr:`self.action_space` in
    the :meth:`__init__` method of your wrapper.

    Among others, Gymnasium provides the action wrappers :class:`ClipAction` and :class:`RescaleAction` for clipping and rescaling actions.
    """

    def __init__(self, env, batch_size, causal):
        """Constructor for the action wrapper."""
        super().__init__(env)
        self.dh = dh.DataHandling()
        self.batch_size = batch_size
        self.t = 0
        self.actions = []
        self.causal = causal

    def action(self, action):
        """
        Returns a modified action before :meth:`env.step` is called.
        :param action: The original :meth:`step` actions:
        :returns: The modified actions
        """
        # if self.causal:
        #     self.t += 1
        #     if self.t < self.batch_size:
        #         # Convert action to intervention
        #         intervention = (np.absolute(action) > 0).astype(int)
        #         if len(self.actions) == 0:
        #             self.actions = np.array(np.array([intervention], dtype=np.float32))
        #         else:
        #             self.actions = np.concatenate((self.actions, np.array([intervention])), axis=0, dtype=np.float32)
        #     elif self.batch_size == self.t:
        #         self.dh.batch_update_npz(self.actions, filename='intervention')
        #         self.actions = []
        #         self.t = 0
        return action


class RewardWrapper(gym.RewardWrapper):
    def __init__(self, env, func):
        super().__init__(env)
        self.func = func

    def reward(self, reward):
        return self.func(reward)

