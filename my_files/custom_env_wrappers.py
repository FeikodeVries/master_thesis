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
        keep_dim: bool = False
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
            self.observation_space = None
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
            if self.rgb:
                self.observation_space = spaces.Box(shape=[self.shape[0], self.shape[1], 3],
                                                    low=low, high=high, dtype=pixels.dtype)
            else:
                if self.keep_dim:
                    pixels_space = spaces.Box(shape=(self.shape[0], self.shape[1], 1),
                                              low=low, high=high, dtype=pixels.dtype)
                else:
                    pixels_space = spaces.Box(shape=(self.shape[0], self.shape[1]), low=low,
                                              high=high, dtype=pixels.dtype)

            # pixels_spaces[pixel_key] = pixels_space
            # END MYCODE

        # self.observation_space.spaces.update(pixels_spaces)

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


class ResizeObservation(gym.ObservationWrapper, gym.utils.RecordConstructorArgs):
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

    def __init__(self, env: gym.Env, shape: tuple[int, int] | int) -> None:
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

        assert isinstance(env.observation_space['pixels'], gym.spaces.Box), f"Expected the observation space to be Box, " \
                                                                  f"actual type: {type(env.observation_space['pixels'])}"
        dims = len(env.observation_space['pixels'].shape)
        assert (dims == 2 or dims == 3), f"Expected the observation space to have 2 or 3 dimensions, got: {dims}"

        obs_shape = self.shape + env.observation_space['pixels'].shape[2:]
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

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

        observation = cv2.resize(
            observation, self.shape[::-1], interpolation=cv2.INTER_AREA
        )
        return observation.reshape(self.observation_space['pixels'].shape)

