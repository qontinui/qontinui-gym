"""Action space builders for Qontinui environments.

This module provides utilities for building Gymnasium action spaces
from QontinuiConfig configurations, supporting both discrete and
hybrid (parameterized) action spaces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
from gymnasium import spaces

if TYPE_CHECKING:
    from qontinui_gym.config.loader import QontinuiConfig


class ActionSpaceMode(Enum):
    """Mode for action space construction."""

    DISCRETE = "discrete"  # Simple discrete: workflow index only
    HYBRID = "hybrid"  # Dict with action type + parameters
    PARAMETERIZED = "parameterized"  # Full parameterization


@dataclass
class ActionSpaceConfig:
    """Configuration for action space construction.

    Attributes:
        mode: Action space mode (discrete, hybrid, or parameterized)
        include_coordinates: Include click coordinate actions
        include_text_input: Include text typing actions
        include_scroll: Include scroll actions
        include_state_navigation: Include go-to-state actions
        screen_width: Screen width for coordinate bounds
        screen_height: Screen height for coordinate bounds
        include_categories: Only include workflows from these categories
        exclude_workflows: Exclude workflows with these names
    """

    mode: ActionSpaceMode = ActionSpaceMode.HYBRID
    include_coordinates: bool = True
    include_text_input: bool = True
    include_scroll: bool = True
    include_state_navigation: bool = True
    screen_width: int = 1920
    screen_height: int = 1080
    include_categories: list[str] | None = None
    exclude_workflows: list[str] = field(default_factory=list)
    max_text_length: int = 256
    max_scroll_clicks: int = 10
    max_wait_seconds: float = 10.0

    @classmethod
    def from_qontinui_config(cls, config: QontinuiConfig) -> ActionSpaceConfig:
        """Create action space config from QontinuiConfig.

        Extracts relevant settings like screen resolution from the config.
        """
        # Default values - could be extended to read from config settings
        return cls()

    @classmethod
    def discrete(cls) -> ActionSpaceConfig:
        """Create a simple discrete action space config."""
        return cls(mode=ActionSpaceMode.DISCRETE)

    @classmethod
    def hybrid(cls) -> ActionSpaceConfig:
        """Create a hybrid action space config with all action types."""
        return cls(mode=ActionSpaceMode.HYBRID)


def build_action_space(
    config: QontinuiConfig,
    action_config: ActionSpaceConfig | None = None,
) -> spaces.Space[Any]:
    """Build a Gymnasium action space from QontinuiConfig.

    Args:
        config: Parsed QontinuiConfig
        action_config: Configuration for action space construction

    Returns:
        Gymnasium Space object (Discrete or Dict)

    Examples:
        # Discrete action space (workflow selection only)
        config = ActionSpaceConfig.discrete()
        space = build_action_space(qontinui_config, config)
        # space is Discrete(num_workflows)

        # Hybrid action space
        config = ActionSpaceConfig.hybrid()
        space = build_action_space(qontinui_config, config)
        # space is Dict with action_type, workflow_idx, coordinates, etc.
    """
    if action_config is None:
        action_config = ActionSpaceConfig()

    # Filter workflows
    workflows = _filter_workflows(config, action_config)

    if action_config.mode == ActionSpaceMode.DISCRETE:
        return _build_discrete_space(workflows)

    return _build_hybrid_space(config, workflows, action_config)


def _filter_workflows(
    config: QontinuiConfig,
    action_config: ActionSpaceConfig,
) -> list[Any]:
    """Filter workflows based on configuration."""
    workflows = []

    for w in config.workflows:
        # Skip excluded workflows
        if w.name in action_config.exclude_workflows:
            continue

        # Filter by category if specified
        if action_config.include_categories is not None:
            if w.category not in action_config.include_categories:
                continue

        workflows.append(w)

    return workflows


def _build_discrete_space(workflows: list[Any]) -> spaces.Discrete:  # type: ignore[type-arg]
    """Build simple discrete action space."""
    num_actions = max(1, len(workflows))
    return spaces.Discrete(num_actions)


def _build_hybrid_space(
    config: QontinuiConfig,
    workflows: list[Any],
    action_config: ActionSpaceConfig,
) -> spaces.Dict:
    """Build hybrid action space with multiple action types.

    Structure:
        Dict({
            "action_type": Discrete(N),      # Which action category
            "workflow_idx": Discrete(W),      # Which workflow (for RUN_WORKFLOW)
            "coordinates": Box([0,0], [W,H]), # Click coordinates
            "scroll": Dict({direction, clicks})
            "text": Text(max_length)
            "state_idx": Discrete(S)          # For GO_TO_STATE
            "wait_duration": Box([0.1], [10]) # Wait duration
        })
    """
    space_dict: dict[str, spaces.Space[Any]] = {}

    # Build list of action types
    action_types = ["workflow"]  # Always available

    if action_config.include_coordinates:
        action_types.extend(["click", "double_click", "right_click"])

    if action_config.include_scroll:
        action_types.append("scroll")

    if action_config.include_text_input:
        action_types.append("type")

    if action_config.include_state_navigation and config.states:
        action_types.append("go_to_state")

    action_types.append("wait")

    # Action type selector
    space_dict["action_type"] = spaces.Discrete(len(action_types))

    # Workflow selector
    num_workflows = max(1, len(workflows))
    space_dict["workflow_idx"] = spaces.Discrete(num_workflows)

    # Coordinate actions
    if action_config.include_coordinates:
        space_dict["coordinates"] = spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array(
                [action_config.screen_width, action_config.screen_height],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

    # Scroll action
    if action_config.include_scroll:
        space_dict["scroll"] = spaces.Dict(
            {
                "direction": spaces.Discrete(4),  # up, down, left, right
                "clicks": spaces.Discrete(action_config.max_scroll_clicks),
            }
        )

    # Text input
    if action_config.include_text_input:
        space_dict["text"] = spaces.Text(
            max_length=action_config.max_text_length,
            min_length=0,
        )

    # State navigation
    if action_config.include_state_navigation and config.states:
        num_states = len(config.states)
        space_dict["state_idx"] = spaces.Discrete(num_states)

    # Wait duration
    space_dict["wait_duration"] = spaces.Box(
        low=np.array([0.1], dtype=np.float32),
        high=np.array([action_config.max_wait_seconds], dtype=np.float32),
        dtype=np.float32,
    )

    return spaces.Dict(space_dict)


def decode_action(
    action: int | dict[str, Any],
    config: QontinuiConfig,
    action_config: ActionSpaceConfig,
) -> tuple[str, dict[str, Any]]:
    """Decode an action into workflow name and parameters.

    Args:
        action: Action from the action space (int or dict)
        config: QontinuiConfig for resolving indices
        action_config: Action space configuration

    Returns:
        Tuple of (action_type, parameters dict)

    Examples:
        # Discrete action
        action_type, params = decode_action(3, config, config)
        # -> ("workflow", {"workflow_name": "ThirdWorkflow"})

        # Hybrid action
        action_type, params = decode_action(
            {"action_type": 0, "workflow_idx": 2},
            config, action_config
        )
        # -> ("workflow", {"workflow_name": "ThirdWorkflow"})
    """
    if isinstance(action, int):
        # Discrete mode: action is workflow index
        workflows = _filter_workflows(config, action_config)
        if action < len(workflows):
            return "workflow", {"workflow_name": workflows[action].name}
        return "workflow", {"workflow_name": ""}

    # Hybrid mode
    action_types = _get_action_types(config, action_config)
    action_type_idx = action.get("action_type", 0)
    action_type = action_types[action_type_idx] if action_type_idx < len(action_types) else "workflow"

    workflows = _filter_workflows(config, action_config)
    params: dict[str, Any] = {}

    if action_type == "workflow":
        workflow_idx = action.get("workflow_idx", 0)
        if workflow_idx < len(workflows):
            params["workflow_name"] = workflows[workflow_idx].name
        else:
            params["workflow_name"] = ""

    elif action_type in ("click", "double_click", "right_click"):
        coords = action.get("coordinates", np.array([0.0, 0.0]))
        params["x"] = float(coords[0])
        params["y"] = float(coords[1])
        params["click_type"] = action_type

    elif action_type == "scroll":
        scroll = action.get("scroll", {"direction": 0, "clicks": 1})
        directions = ["up", "down", "left", "right"]
        params["direction"] = directions[scroll.get("direction", 0)]
        params["clicks"] = scroll.get("clicks", 1) + 1  # 0-indexed to 1-indexed

    elif action_type == "type":
        params["text"] = action.get("text", "")

    elif action_type == "go_to_state":
        state_idx = action.get("state_idx", 0)
        if state_idx < len(config.states):
            params["state_name"] = config.states[state_idx].name
        else:
            params["state_name"] = ""

    elif action_type == "wait":
        wait_duration = action.get("wait_duration", np.array([1.0]))
        params["duration"] = float(wait_duration[0])

    return action_type, params


def _get_action_types(
    config: QontinuiConfig,
    action_config: ActionSpaceConfig,
) -> list[str]:
    """Get list of action types in order."""
    action_types = ["workflow"]

    if action_config.include_coordinates:
        action_types.extend(["click", "double_click", "right_click"])

    if action_config.include_scroll:
        action_types.append("scroll")

    if action_config.include_text_input:
        action_types.append("type")

    if action_config.include_state_navigation and config.states:
        action_types.append("go_to_state")

    action_types.append("wait")

    return action_types


def get_action_metadata(
    config: QontinuiConfig,
    action_config: ActionSpaceConfig,
) -> dict[str, Any]:
    """Get metadata about the action space.

    Useful for logging and debugging.

    Returns:
        Dict with action space information
    """
    workflows = _filter_workflows(config, action_config)
    action_types = _get_action_types(config, action_config)

    return {
        "mode": action_config.mode.value,
        "num_workflows": len(workflows),
        "workflow_names": [w.name for w in workflows],
        "action_types": action_types,
        "num_states": len(config.states),
        "screen_size": (action_config.screen_width, action_config.screen_height),
    }
