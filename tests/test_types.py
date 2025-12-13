"""Tests for qontinui_gym.types module."""

from qontinui_gym.types import (
    ActionType,
    Coordinates,
    ExecutionEvent,
    MonitorInfo,
    Region,
    ScreenshotInfo,
    ScrollDirection,
    StateDetectionResult,
    StateInfo,
    TransitionInfo,
    WorkflowInfo,
)


class TestActionType:
    """Tests for ActionType enum."""

    def test_action_types_exist(self) -> None:
        """Verify all expected action types exist."""
        assert ActionType.WORKFLOW.value == "workflow"
        assert ActionType.CLICK.value == "click"
        assert ActionType.DOUBLE_CLICK.value == "double_click"
        assert ActionType.RIGHT_CLICK.value == "right_click"
        assert ActionType.SCROLL.value == "scroll"
        assert ActionType.TYPE.value == "type"
        assert ActionType.GO_TO_STATE.value == "go_to_state"
        assert ActionType.WAIT.value == "wait"

    def test_action_type_count(self) -> None:
        """Verify the number of action types."""
        assert len(ActionType) == 8


class TestScrollDirection:
    """Tests for ScrollDirection enum."""

    def test_scroll_directions_exist(self) -> None:
        """Verify all scroll directions exist."""
        assert ScrollDirection.UP.value == "up"
        assert ScrollDirection.DOWN.value == "down"
        assert ScrollDirection.LEFT.value == "left"
        assert ScrollDirection.RIGHT.value == "right"


class TestCoordinates:
    """Tests for Coordinates dataclass."""

    def test_create_coordinates(self) -> None:
        """Test creating coordinates."""
        coords = Coordinates(x=100.5, y=200.5)
        assert coords.x == 100.5
        assert coords.y == 200.5

    def test_coordinates_equality(self) -> None:
        """Test coordinates equality."""
        c1 = Coordinates(x=10, y=20)
        c2 = Coordinates(x=10, y=20)
        assert c1 == c2


class TestRegion:
    """Tests for Region dataclass."""

    def test_create_region(self) -> None:
        """Test creating a region."""
        region = Region(x=0, y=0, width=100, height=50)
        assert region.x == 0
        assert region.y == 0
        assert region.width == 100
        assert region.height == 50

    def test_region_center(self) -> None:
        """Test region center calculation."""
        region = Region(x=0, y=0, width=100, height=50)
        center = region.center
        assert center.x == 50.0
        assert center.y == 25.0

    def test_region_center_with_offset(self) -> None:
        """Test region center with non-zero origin."""
        region = Region(x=100, y=200, width=50, height=50)
        center = region.center
        assert center.x == 125.0
        assert center.y == 225.0


class TestMonitorInfo:
    """Tests for MonitorInfo dataclass."""

    def test_create_monitor_info(self) -> None:
        """Test creating monitor info."""
        monitor = MonitorInfo(
            index=0,
            x=0,
            y=0,
            width=1920,
            height=1080,
            is_primary=True,
            position="primary",
            name="Monitor 1",
            description="Primary display",
        )
        assert monitor.index == 0
        assert monitor.width == 1920
        assert monitor.height == 1080
        assert monitor.is_primary is True


class TestWorkflowInfo:
    """Tests for WorkflowInfo dataclass."""

    def test_create_workflow_info(self) -> None:
        """Test creating workflow info."""
        workflow = WorkflowInfo(
            id="wf-1",
            name="Login",
            description="Login workflow",
            category="Auth",
        )
        assert workflow.id == "wf-1"
        assert workflow.name == "Login"
        assert workflow.description == "Login workflow"
        assert workflow.category == "Auth"

    def test_workflow_info_optional_fields(self) -> None:
        """Test workflow info with optional fields."""
        workflow = WorkflowInfo(id="wf-2", name="Test")
        assert workflow.description is None
        assert workflow.category is None


class TestStateInfo:
    """Tests for StateInfo dataclass."""

    def test_create_state_info(self) -> None:
        """Test creating state info."""
        state = StateInfo(
            id="state-1",
            name="Home",
            description="Home page",
            image_count=3,
            is_initial=True,
            is_final=False,
        )
        assert state.id == "state-1"
        assert state.name == "Home"
        assert state.image_count == 3
        assert state.is_initial is True
        assert state.is_final is False

    def test_state_info_defaults(self) -> None:
        """Test state info default values."""
        state = StateInfo(id="s1", name="Test")
        assert state.description is None
        assert state.image_count == 0
        assert state.is_initial is False
        assert state.is_final is False


class TestTransitionInfo:
    """Tests for TransitionInfo dataclass."""

    def test_create_transition_info(self) -> None:
        """Test creating transition info."""
        transition = TransitionInfo(
            id="t-1",
            from_state="state-1",
            to_state="state-2",
            workflows=["workflow-1"],
            priority=1,
        )
        assert transition.id == "t-1"
        assert transition.from_state == "state-1"
        assert transition.to_state == "state-2"
        assert transition.workflows == ["workflow-1"]
        assert transition.priority == 1

    def test_transition_info_defaults(self) -> None:
        """Test transition info default values."""
        transition = TransitionInfo(id="t-2", from_state="s1", to_state="s2")
        assert transition.workflows == []
        assert transition.priority == 0


class TestStateDetectionResult:
    """Tests for StateDetectionResult dataclass."""

    def test_create_detection_result(self) -> None:
        """Test creating state detection result."""
        result = StateDetectionResult(
            state_ids=["s1", "s2"],
            state_names=["Home", "Dashboard"],
            confidences={"s1": 0.95, "s2": 0.85},
            patterns_matched=["pattern1", "pattern2"],
        )
        assert len(result.state_ids) == 2
        assert result.confidences["s1"] == 0.95


class TestExecutionEvent:
    """Tests for ExecutionEvent dataclass."""

    def test_create_execution_event(self) -> None:
        """Test creating execution event."""
        event = ExecutionEvent(
            event_type="click",
            timestamp=1234567890.0,
            data={"x": 100, "y": 200},
        )
        assert event.event_type == "click"
        assert event.timestamp == 1234567890.0
        assert event.data["x"] == 100

    def test_execution_event_defaults(self) -> None:
        """Test execution event default data."""
        event = ExecutionEvent(event_type="wait", timestamp=0.0)
        assert event.data == {}


class TestScreenshotInfo:
    """Tests for ScreenshotInfo dataclass."""

    def test_create_screenshot_info(self) -> None:
        """Test creating screenshot info."""
        info = ScreenshotInfo(
            path="/tmp/screenshot.png",
            width=1920,
            height=1080,
            timestamp=1234567890.0,
            monitor_index=0,
        )
        assert info.path == "/tmp/screenshot.png"
        assert info.width == 1920
        assert info.monitor_index == 0

    def test_screenshot_info_defaults(self) -> None:
        """Test screenshot info with optional fields."""
        info = ScreenshotInfo(
            path=None,
            width=640,
            height=480,
            timestamp=0.0,
        )
        assert info.path is None
        assert info.monitor_index is None
