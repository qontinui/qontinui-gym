"""Core type definitions for qontinui-gym.

This module contains all the type definitions used throughout the package,
including dataclasses for configuration, results, and environment state.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ActionType(Enum):
    """Types of actions that can be executed in the environment."""

    WORKFLOW = "workflow"  # Execute a workflow by name
    CLICK = "click"  # Click at coordinates
    DOUBLE_CLICK = "double_click"  # Double-click at coordinates
    RIGHT_CLICK = "right_click"  # Right-click at coordinates
    SCROLL = "scroll"  # Scroll in a direction
    TYPE = "type"  # Type text
    GO_TO_STATE = "go_to_state"  # Navigate to a state
    WAIT = "wait"  # Wait for a duration


class ScrollDirection(Enum):
    """Scroll directions."""

    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"


@dataclass
class Coordinates:
    """Screen coordinates."""

    x: float
    y: float


@dataclass
class Region:
    """Screen region (bounding box)."""

    x: int
    y: int
    width: int
    height: int

    @property
    def center(self) -> Coordinates:
        """Get the center point of the region."""
        return Coordinates(
            x=self.x + self.width / 2,
            y=self.y + self.height / 2,
        )


@dataclass
class MonitorInfo:
    """Information about a display monitor."""

    index: int
    x: int
    y: int
    width: int
    height: int
    is_primary: bool
    position: str  # "primary", "left", "right", etc.
    name: str
    description: str


@dataclass
class WorkflowInfo:
    """Information about a workflow from the configuration."""

    id: str
    name: str
    description: str | None = None
    category: str | None = None


@dataclass
class StateInfo:
    """Information about a state from the configuration."""

    id: str
    name: str
    description: str | None = None
    image_count: int = 0
    is_initial: bool = False
    is_final: bool = False


@dataclass
class TransitionInfo:
    """Information about a state transition."""

    id: str
    from_state: str
    to_state: str
    workflows: list[str] = field(default_factory=list)
    priority: int = 0


@dataclass
class StateDetectionResult:
    """Result of state detection."""

    state_ids: list[str]
    state_names: list[str]
    confidences: dict[str, float]  # state_id -> confidence
    patterns_matched: list[str]


@dataclass
class ExecutionEvent:
    """An event that occurred during workflow execution."""

    event_type: str
    timestamp: float
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScreenshotInfo:
    """Information about a captured screenshot."""

    path: str | None
    width: int
    height: int
    timestamp: float
    monitor_index: int | None = None
