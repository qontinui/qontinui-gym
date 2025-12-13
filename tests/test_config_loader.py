"""Tests for qontinui_gym.config.loader module."""

import json
import tempfile
from pathlib import Path

import pytest

from qontinui_gym.config.loader import (
    ExecutionSettings,
    compute_state_distances,
    get_state_graph,
    load_qontinui_config,
)


class TestLoadQontinuiConfig:
    """Tests for load_qontinui_config function."""

    def test_load_valid_config(self, sample_config_file: Path) -> None:
        """Test loading a valid config file."""
        config = load_qontinui_config(sample_config_file)
        assert config.metadata.name == "Test Config"
        assert len(config.workflows) == 3
        assert len(config.states) == 3
        assert len(config.transitions) == 3

    def test_load_missing_file(self) -> None:
        """Test loading a non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_qontinui_config("/nonexistent/path/config.json")

    def test_load_invalid_json(self) -> None:
        """Test loading invalid JSON."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not valid json {{{")
            path = Path(f.name)

        with pytest.raises(json.JSONDecodeError):
            load_qontinui_config(path)

    def test_load_minimal_config(self) -> None:
        """Test loading a minimal config."""
        minimal = {"version": "1.0.0"}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(minimal, f)
            path = Path(f.name)

        config = load_qontinui_config(path)
        assert config.version == "1.0.0"
        assert config.metadata.name == "Unnamed Config"
        assert len(config.workflows) == 0
        assert len(config.states) == 0


class TestConfigMetadata:
    """Tests for ConfigMetadata dataclass."""

    def test_metadata_from_config(self, sample_config_file: Path) -> None:
        """Test metadata is parsed correctly."""
        config = load_qontinui_config(sample_config_file)
        assert config.metadata.name == "Test Config"
        assert config.metadata.description == "A test configuration"
        assert config.metadata.author == "Test Author"
        assert config.metadata.target_application == "Test App"


class TestExecutionSettings:
    """Tests for ExecutionSettings dataclass."""

    def test_settings_from_config(self, sample_config_file: Path) -> None:
        """Test execution settings are parsed correctly."""
        config = load_qontinui_config(sample_config_file)
        assert config.settings.default_timeout == 10000
        assert config.settings.default_retry_count == 3
        assert config.settings.failure_strategy == "continue"

    def test_settings_defaults(self) -> None:
        """Test default execution settings."""
        settings = ExecutionSettings()
        assert settings.default_timeout == 10000
        assert settings.default_retry_count == 0
        assert settings.failure_strategy == "continue"


class TestQontinuiConfig:
    """Tests for QontinuiConfig class."""

    def test_workflow_names(self, sample_config_file: Path) -> None:
        """Test workflow_names property."""
        config = load_qontinui_config(sample_config_file)
        names = config.workflow_names
        assert "Login" in names
        assert "Logout" in names
        assert "Navigate Home" in names

    def test_state_names(self, sample_config_file: Path) -> None:
        """Test state_names property."""
        config = load_qontinui_config(sample_config_file)
        names = config.state_names
        assert "Login Page" in names
        assert "Home Page" in names
        assert "Logout Complete" in names

    def test_initial_states(self, sample_config_file: Path) -> None:
        """Test initial_states property."""
        config = load_qontinui_config(sample_config_file)
        initial = config.initial_states
        assert len(initial) == 1
        assert initial[0].name == "Login Page"

    def test_final_states(self, sample_config_file: Path) -> None:
        """Test final_states property."""
        config = load_qontinui_config(sample_config_file)
        final = config.final_states
        assert len(final) == 1
        assert final[0].name == "Logout Complete"

    def test_get_workflow_by_name(self, sample_config_file: Path) -> None:
        """Test get_workflow_by_name method."""
        config = load_qontinui_config(sample_config_file)

        workflow = config.get_workflow_by_name("Login")
        assert workflow is not None
        assert workflow.id == "workflow-1"

        missing = config.get_workflow_by_name("Nonexistent")
        assert missing is None

    def test_get_state_by_name(self, sample_config_file: Path) -> None:
        """Test get_state_by_name method."""
        config = load_qontinui_config(sample_config_file)

        state = config.get_state_by_name("Home Page")
        assert state is not None
        assert state.id == "state-2"

        missing = config.get_state_by_name("Nonexistent")
        assert missing is None

    def test_get_state_by_id(self, sample_config_file: Path) -> None:
        """Test get_state_by_id method."""
        config = load_qontinui_config(sample_config_file)

        state = config.get_state_by_id("state-2")
        assert state is not None
        assert state.name == "Home Page"

        missing = config.get_state_by_id("nonexistent-id")
        assert missing is None

    def test_get_workflows_by_category(self, sample_config_file: Path) -> None:
        """Test get_workflows_by_category method."""
        config = load_qontinui_config(sample_config_file)

        auth_workflows = config.get_workflows_by_category("Authentication")
        assert len(auth_workflows) == 2

        nav_workflows = config.get_workflows_by_category("Navigation")
        assert len(nav_workflows) == 1

    def test_get_transitions_from_state(self, sample_config_file: Path) -> None:
        """Test get_transitions_from_state method."""
        config = load_qontinui_config(sample_config_file)

        # state-2 has two outgoing transitions
        transitions = config.get_transitions_from_state("state-2")
        assert len(transitions) == 2

        # state-3 has no outgoing transitions
        transitions = config.get_transitions_from_state("state-3")
        assert len(transitions) == 0

    def test_get_transitions_to_state(self, sample_config_file: Path) -> None:
        """Test get_transitions_to_state method."""
        config = load_qontinui_config(sample_config_file)

        # state-2 has one incoming transition from state-1
        # and one self-loop (from state-2 to state-2)
        transitions = config.get_transitions_to_state("state-2")
        assert len(transitions) == 2

        # state-1 has no incoming transitions
        transitions = config.get_transitions_to_state("state-1")
        assert len(transitions) == 0


class TestGetStateGraph:
    """Tests for get_state_graph function."""

    def test_build_graph(self, sample_config_file: Path) -> None:
        """Test building state graph."""
        config = load_qontinui_config(sample_config_file)
        graph = get_state_graph(config)

        assert "state-1" in graph
        assert "state-2" in graph
        assert "state-3" in graph

        # state-1 -> state-2
        assert "state-2" in graph["state-1"]

        # state-2 -> state-3 and state-2 -> state-2 (self-loop)
        assert "state-3" in graph["state-2"]
        assert "state-2" in graph["state-2"]

        # state-3 has no outgoing edges
        assert graph["state-3"] == []


class TestComputeStateDistances:
    """Tests for compute_state_distances function."""

    def test_distances_to_goal(self, sample_config_file: Path) -> None:
        """Test computing distances to goal states."""
        config = load_qontinui_config(sample_config_file)
        distances = compute_state_distances(config, ["state-3"])

        # state-3 is the goal, distance = 0
        assert distances["state-3"] == 0

        # state-2 -> state-3, distance = 1
        assert distances["state-2"] == 1

        # state-1 -> state-2 -> state-3, distance = 2
        assert distances["state-1"] == 2

    def test_distances_multiple_goals(self, sample_config_file: Path) -> None:
        """Test distances with multiple goal states."""
        config = load_qontinui_config(sample_config_file)
        distances = compute_state_distances(config, ["state-2", "state-3"])

        # Both are goals or 1 step from goal
        assert distances["state-2"] == 0  # state-2 is a goal
        assert distances["state-3"] == 0  # state-3 is a goal
        assert distances["state-1"] == 1  # state-1 -> state-2

    def test_distances_empty_goals(self, sample_config_file: Path) -> None:
        """Test distances with no goals."""
        config = load_qontinui_config(sample_config_file)
        distances = compute_state_distances(config, [])

        # All states should be unreachable (-1)
        assert all(d == -1 for d in distances.values())
