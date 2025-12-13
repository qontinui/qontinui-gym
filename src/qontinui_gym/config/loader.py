"""Configuration loader for QontinuiConfig JSON files.

This module handles loading and parsing QontinuiConfig exports from
qontinui-web for use in the Gymnasium environment.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qontinui_gym.types import StateInfo, TransitionInfo, WorkflowInfo

logger = logging.getLogger(__name__)


@dataclass
class ConfigMetadata:
    """Metadata from the configuration file."""

    name: str
    description: str | None = None
    author: str | None = None
    created: str | None = None
    modified: str | None = None
    version: str = "2.7.0"
    target_application: str | None = None


@dataclass
class ExecutionSettings:
    """Execution settings from the configuration."""

    default_timeout: int = 10000
    default_retry_count: int = 0
    failure_strategy: str = "continue"


@dataclass
class QontinuiConfig:
    """Parsed QontinuiConfig representing an automation configuration.

    This class holds all the information from a QontinuiConfig JSON export,
    including workflows, states, transitions, and settings.
    """

    metadata: ConfigMetadata
    workflows: list[WorkflowInfo]
    states: list[StateInfo]
    transitions: list[TransitionInfo]
    categories: list[str]
    settings: ExecutionSettings
    version: str
    raw_config: dict[str, Any]

    @property
    def workflow_names(self) -> list[str]:
        """Get list of all workflow names."""
        return [w.name for w in self.workflows]

    @property
    def state_names(self) -> list[str]:
        """Get list of all state names."""
        return [s.name for s in self.states]

    @property
    def initial_states(self) -> list[StateInfo]:
        """Get states marked as initial."""
        return [s for s in self.states if s.is_initial]

    @property
    def final_states(self) -> list[StateInfo]:
        """Get states marked as final."""
        return [s for s in self.states if s.is_final]

    def get_workflow_by_name(self, name: str) -> WorkflowInfo | None:
        """Get workflow by name."""
        for w in self.workflows:
            if w.name == name:
                return w
        return None

    def get_state_by_name(self, name: str) -> StateInfo | None:
        """Get state by name."""
        for s in self.states:
            if s.name == name:
                return s
        return None

    def get_state_by_id(self, state_id: str) -> StateInfo | None:
        """Get state by ID."""
        for s in self.states:
            if s.id == state_id:
                return s
        return None

    def get_workflows_by_category(self, category: str) -> list[WorkflowInfo]:
        """Get workflows in a specific category."""
        return [w for w in self.workflows if w.category == category]

    def get_transitions_from_state(self, state_id: str) -> list[TransitionInfo]:
        """Get all transitions from a given state."""
        return [t for t in self.transitions if t.from_state == state_id]

    def get_transitions_to_state(self, state_id: str) -> list[TransitionInfo]:
        """Get all transitions to a given state."""
        return [t for t in self.transitions if t.to_state == state_id]


def load_qontinui_config(config_path: str | Path) -> QontinuiConfig:
    """Load a QontinuiConfig JSON file.

    Args:
        config_path: Path to the JSON configuration file

    Returns:
        Parsed QontinuiConfig object

    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is not valid JSON
        ValueError: If config file is missing required fields
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    logger.info(f"Loading config from: {path}")

    with open(path, encoding="utf-8") as f:
        raw_config = json.load(f)

    return _parse_config(raw_config)


def _parse_config(raw: dict[str, Any]) -> QontinuiConfig:
    """Parse raw JSON config into QontinuiConfig object.

    Args:
        raw: Raw JSON dictionary

    Returns:
        Parsed QontinuiConfig object
    """
    version = raw.get("version", "1.0.0")

    # Parse metadata
    meta_raw = raw.get("metadata", {})
    metadata = ConfigMetadata(
        name=meta_raw.get("name", "Unnamed Config"),
        description=meta_raw.get("description"),
        author=meta_raw.get("author"),
        created=meta_raw.get("created"),
        modified=meta_raw.get("modified"),
        version=version,
        target_application=meta_raw.get("targetApplication"),
    )

    # Parse workflows
    workflows = []
    for w in raw.get("workflows", []):
        workflows.append(
            WorkflowInfo(
                id=w.get("id", ""),
                name=w.get("name", ""),
                description=w.get("description"),
                category=w.get("category"),
            )
        )

    # Parse states
    states = []
    for s in raw.get("states", []):
        state_images = s.get("stateImages", [])
        states.append(
            StateInfo(
                id=s.get("id", ""),
                name=s.get("name", ""),
                description=s.get("description"),
                image_count=len(state_images),
                is_initial=s.get("isInitial", False),
                is_final=s.get("isFinal", False),
            )
        )

    # Parse transitions
    transitions = []
    for t in raw.get("transitions", []):
        # Handle both OutgoingTransition and IncomingTransition
        if t.get("type") == "OutgoingTransition":
            transitions.append(
                TransitionInfo(
                    id=t.get("id", ""),
                    from_state=t.get("fromState", ""),
                    to_state=t.get("toState", ""),
                    workflows=t.get("workflows", []),
                    priority=t.get("priority", 0),
                )
            )

    # Parse settings
    settings_raw = raw.get("settings", {})
    execution_raw = settings_raw.get("execution", {})
    settings = ExecutionSettings(
        default_timeout=execution_raw.get("defaultTimeout", 10000),
        default_retry_count=execution_raw.get("defaultRetryCount", 0),
        failure_strategy=execution_raw.get("failureStrategy", "continue"),
    )

    # Categories
    categories = raw.get("categories", [])

    return QontinuiConfig(
        metadata=metadata,
        workflows=workflows,
        states=states,
        transitions=transitions,
        categories=categories,
        settings=settings,
        version=version,
        raw_config=raw,
    )


def get_state_graph(config: QontinuiConfig) -> dict[str, list[str]]:
    """Build adjacency list representation of state graph.

    Args:
        config: Parsed configuration

    Returns:
        Dict mapping state_id -> list of reachable state_ids
    """
    graph: dict[str, list[str]] = {s.id: [] for s in config.states}

    for t in config.transitions:
        if t.from_state in graph:
            graph[t.from_state].append(t.to_state)

    return graph


def compute_state_distances(
    config: QontinuiConfig,
    goal_states: list[str],
) -> dict[str, int]:
    """Compute shortest path distances from all states to goal states.

    Uses BFS to find shortest paths in the state graph.

    Args:
        config: Parsed configuration
        goal_states: List of goal state IDs

    Returns:
        Dict mapping state_id -> minimum distance to any goal state
        (-1 if unreachable)
    """
    try:
        import networkx as nx
    except ImportError:
        logger.warning("networkx not installed, using BFS fallback")
        return _compute_distances_bfs(config, goal_states)

    # Build directed graph
    graph: nx.DiGraph[str] = nx.DiGraph()
    for s in config.states:
        graph.add_node(s.id)

    for t in config.transitions:
        graph.add_edge(t.from_state, t.to_state)

    # Compute distances to goals (reverse graph for "distance TO goal")
    distances: dict[str, int] = {}
    graph_rev = graph.reverse()

    for goal_id in goal_states:
        if goal_id not in graph_rev:
            continue

        try:
            lengths = dict(nx.single_source_shortest_path_length(graph_rev, goal_id))
            for state_id, dist in lengths.items():
                if state_id not in distances:
                    distances[state_id] = dist
                else:
                    distances[state_id] = min(distances[state_id], dist)
        except nx.NetworkXError:
            pass

    # Mark unreachable states
    for s in config.states:
        if s.id not in distances:
            distances[s.id] = -1

    return distances


def _compute_distances_bfs(
    config: QontinuiConfig,
    goal_states: list[str],
) -> dict[str, int]:
    """BFS fallback for computing state distances without networkx."""
    from collections import deque

    # Build reverse adjacency list
    reverse_graph: dict[str, list[str]] = {s.id: [] for s in config.states}
    for t in config.transitions:
        if t.to_state in reverse_graph:
            reverse_graph[t.to_state].append(t.from_state)

    distances: dict[str, int] = {}

    for goal_id in goal_states:
        if goal_id not in reverse_graph:
            continue

        # BFS from goal
        queue: deque[tuple[str, int]] = deque([(goal_id, 0)])
        visited = {goal_id}

        while queue:
            state_id, dist = queue.popleft()

            if state_id not in distances:
                distances[state_id] = dist
            else:
                distances[state_id] = min(distances[state_id], dist)

            for neighbor in reverse_graph.get(state_id, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

    # Mark unreachable states
    for s in config.states:
        if s.id not in distances:
            distances[s.id] = -1

    return distances
