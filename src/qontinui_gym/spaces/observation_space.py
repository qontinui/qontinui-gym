"""Observation space builders for Qontinui environments.

This module provides utilities for building Gymnasium observation spaces
from QontinuiConfig configurations, supporting multi-modal observations
combining visual (screenshot) and symbolic (state) information.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from gymnasium import spaces

if TYPE_CHECKING:
    from qontinui_gym.config.loader import QontinuiConfig


@dataclass
class ObservationSpaceConfig:
    """Configuration for observation space construction.

    Attributes:
        include_screenshot: Include RGB screenshot in observations
        screenshot_width: Resized screenshot width
        screenshot_height: Resized screenshot height
        screenshot_grayscale: Use grayscale instead of RGB
        include_state_info: Include state machine information
        include_state_confidence: Include detection confidence scores
        include_available_actions: Include mask of valid actions
        include_action_history: Include recent action history
        action_history_length: Number of recent actions to include
        include_step_count: Include current step number
    """

    # Screenshot settings
    include_screenshot: bool = True
    screenshot_width: int = 224
    screenshot_height: int = 224
    screenshot_grayscale: bool = False

    # State information
    include_state_info: bool = True
    include_state_confidence: bool = True
    include_available_actions: bool = True

    # History
    include_action_history: bool = False
    action_history_length: int = 5

    # Episode tracking
    include_step_count: bool = True

    @classmethod
    def visual_only(cls) -> ObservationSpaceConfig:
        """Create config with only screenshot observation."""
        return cls(
            include_state_info=False,
            include_state_confidence=False,
            include_available_actions=False,
            include_action_history=False,
            include_step_count=False,
        )

    @classmethod
    def state_only(cls) -> ObservationSpaceConfig:
        """Create config with only state information (no screenshot)."""
        return cls(include_screenshot=False)

    @classmethod
    def full(cls) -> ObservationSpaceConfig:
        """Create config with all observation types enabled."""
        return cls(include_action_history=True)


def build_observation_space(
    config: QontinuiConfig,
    obs_config: ObservationSpaceConfig | None = None,
) -> spaces.Space[Any]:
    """Build a Gymnasium observation space from QontinuiConfig.

    Args:
        config: Parsed QontinuiConfig
        obs_config: Configuration for observation space construction

    Returns:
        Gymnasium Space object (Dict with multiple observation types)

    Examples:
        # Default multi-modal observation space
        space = build_observation_space(qontinui_config)

        # Visual-only observation space
        config = ObservationSpaceConfig.visual_only()
        space = build_observation_space(qontinui_config, config)
    """
    if obs_config is None:
        obs_config = ObservationSpaceConfig()

    space_dict: dict[str, spaces.Space[Any]] = {}

    # Screenshot observation
    if obs_config.include_screenshot:
        channels = 1 if obs_config.screenshot_grayscale else 3
        space_dict["screenshot"] = spaces.Box(
            low=0,
            high=255,
            shape=(obs_config.screenshot_height, obs_config.screenshot_width, channels),
            dtype=np.uint8,
        )

    # State information
    if obs_config.include_state_info:
        num_states = max(1, len(config.states))
        state_space_dict: dict[str, spaces.Space[Any]] = {
            "current_states": spaces.MultiBinary(num_states),
        }

        if obs_config.include_state_confidence:
            state_space_dict["confidence"] = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(num_states,),
                dtype=np.float32,
            )

        space_dict["state"] = spaces.Dict(state_space_dict)

    # Available actions mask
    if obs_config.include_available_actions:
        num_workflows = max(1, len(config.workflows))
        space_dict["available_actions"] = spaces.MultiBinary(num_workflows)

    # Step counter
    if obs_config.include_step_count:
        space_dict["step"] = spaces.Box(
            low=0,
            high=np.inf,
            shape=(1,),
            dtype=np.float32,
        )

    # Action history
    if obs_config.include_action_history:
        num_workflows = max(1, len(config.workflows))
        # +1 for "no action" placeholder
        space_dict["action_history"] = spaces.MultiDiscrete(
            [num_workflows + 1] * obs_config.action_history_length
        )

    return spaces.Dict(space_dict)


def build_observation(
    config: QontinuiConfig,
    obs_config: ObservationSpaceConfig,
    screenshot: npt.NDArray[np.uint8] | None = None,
    current_state_ids: list[str] | None = None,
    state_confidences: dict[str, float] | None = None,
    available_workflow_indices: list[int] | None = None,
    step: int = 0,
    action_history: list[int] | None = None,
) -> dict[str, Any]:
    """Build an observation dict from environment state.

    Args:
        config: QontinuiConfig for reference data
        obs_config: Observation space configuration
        screenshot: Raw screenshot as numpy array (will be resized)
        current_state_ids: List of currently active state IDs
        state_confidences: Mapping of state_id -> confidence score
        available_workflow_indices: Indices of valid workflows
        step: Current step number
        action_history: List of recent action indices

    Returns:
        Observation dict matching the observation space structure
    """
    observation: dict[str, Any] = {}

    num_states = max(1, len(config.states))
    num_workflows = max(1, len(config.workflows))

    # Screenshot
    if obs_config.include_screenshot:
        if screenshot is not None:
            processed = _process_screenshot(
                screenshot,
                obs_config.screenshot_width,
                obs_config.screenshot_height,
                obs_config.screenshot_grayscale,
            )
        else:
            # Blank screenshot
            channels = 1 if obs_config.screenshot_grayscale else 3
            processed = np.zeros(
                (obs_config.screenshot_height, obs_config.screenshot_width, channels),
                dtype=np.uint8,
            )
        observation["screenshot"] = processed

    # State information
    if obs_config.include_state_info:
        # Build state ID to index mapping
        state_id_to_idx = {s.id: i for i, s in enumerate(config.states)}

        # Current states as multi-binary
        current_states = np.zeros(num_states, dtype=np.int8)
        if current_state_ids:
            for state_id in current_state_ids:
                if state_id in state_id_to_idx:
                    current_states[state_id_to_idx[state_id]] = 1

        state_obs: dict[str, Any] = {"current_states": current_states}

        # Confidence scores
        if obs_config.include_state_confidence:
            confidence = np.zeros(num_states, dtype=np.float32)
            if state_confidences:
                for state_id, conf in state_confidences.items():
                    if state_id in state_id_to_idx:
                        confidence[state_id_to_idx[state_id]] = conf
            state_obs["confidence"] = confidence

        observation["state"] = state_obs

    # Available actions mask
    if obs_config.include_available_actions:
        available = np.ones(num_workflows, dtype=np.int8)
        if available_workflow_indices is not None:
            available = np.zeros(num_workflows, dtype=np.int8)
            for idx in available_workflow_indices:
                if 0 <= idx < num_workflows:
                    available[idx] = 1
        observation["available_actions"] = available

    # Step count
    if obs_config.include_step_count:
        observation["step"] = np.array([step], dtype=np.float32)

    # Action history
    if obs_config.include_action_history:
        history = np.zeros(obs_config.action_history_length, dtype=np.int64)
        # Fill with "no action" placeholder (num_workflows)
        history.fill(num_workflows)
        if action_history:
            # Fill from the end with most recent actions
            for i, action in enumerate(action_history[-obs_config.action_history_length :]):
                history[i] = action
        observation["action_history"] = history

    return observation


def _process_screenshot(
    screenshot: npt.NDArray[np.uint8],
    target_width: int,
    target_height: int,
    grayscale: bool,
) -> npt.NDArray[np.uint8]:
    """Process and resize screenshot.

    Args:
        screenshot: Raw screenshot as numpy array
        target_width: Target width
        target_height: Target height
        grayscale: Convert to grayscale

    Returns:
        Processed screenshot
    """
    try:
        from PIL import Image
    except ImportError:
        # Fallback: simple resize using numpy
        return _simple_resize(screenshot, target_width, target_height, grayscale)

    # Convert to PIL Image
    img = Image.fromarray(screenshot)

    # Resize
    img = img.resize((target_width, target_height), Image.Resampling.BILINEAR)

    # Convert to grayscale if needed
    if grayscale:
        img = img.convert("L")
        arr = np.array(img)
        return arr[:, :, np.newaxis]  # Add channel dimension

    # Ensure RGB (not RGBA)
    if img.mode == "RGBA":
        img = img.convert("RGB")

    return np.array(img)


def _simple_resize(
    screenshot: npt.NDArray[np.uint8],
    target_width: int,
    target_height: int,
    grayscale: bool,
) -> npt.NDArray[np.uint8]:
    """Simple resize fallback without PIL."""

    h, w = screenshot.shape[:2]

    # Calculate indices for simple nearest-neighbor resize
    x_indices = np.linspace(0, w - 1, target_width).astype(int)
    y_indices = np.linspace(0, h - 1, target_height).astype(int)

    resized = screenshot[np.ix_(y_indices, x_indices)]

    if grayscale:
        if len(resized.shape) == 3:
            # Convert RGB to grayscale using luminosity method
            resized = (
                0.299 * resized[:, :, 0]
                + 0.587 * resized[:, :, 1]
                + 0.114 * resized[:, :, 2]
            ).astype(np.uint8)
        result: npt.NDArray[np.uint8] = resized[:, :, np.newaxis]
        return result

    return np.asarray(resized, dtype=np.uint8)


def get_observation_metadata(
    config: QontinuiConfig,
    obs_config: ObservationSpaceConfig,
) -> dict[str, Any]:
    """Get metadata about the observation space.

    Useful for logging and debugging.

    Returns:
        Dict with observation space information
    """
    return {
        "include_screenshot": obs_config.include_screenshot,
        "screenshot_shape": (
            obs_config.screenshot_height,
            obs_config.screenshot_width,
            1 if obs_config.screenshot_grayscale else 3,
        )
        if obs_config.include_screenshot
        else None,
        "include_state_info": obs_config.include_state_info,
        "num_states": len(config.states),
        "state_names": [s.name for s in config.states],
        "include_available_actions": obs_config.include_available_actions,
        "num_workflows": len(config.workflows),
        "include_action_history": obs_config.include_action_history,
        "action_history_length": obs_config.action_history_length,
    }
