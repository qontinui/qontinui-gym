"""Tests for qontinui_gym.rewards module."""

import numpy as np
import pytest

from qontinui_gym.rewards.base import (
    ActionResult,
    ConstantReward,
    PatternMatch,
    StepInfo,
)
from qontinui_gym.rewards.components import (
    GoalConditionedReward,
    InvalidActionPenalty,
    ScreenChangeReward,
    StateReachReward,
    StateTransitionReward,
    StateVisitCountReward,
    StepPenalty,
)
from qontinui_gym.rewards.composers import (
    ComposedReward,
    CompositionMode,
    RewardBuilder,
)


class TestStepInfo:
    """Tests for StepInfo dataclass."""

    def test_create_step_info(self) -> None:
        """Test creating StepInfo."""
        step_info = StepInfo(
            previous_state_ids=frozenset(["s1"]),
            current_state_ids=frozenset(["s2"]),
            action_type="workflow",
            step_count=5,
        )
        assert step_info.previous_state_ids == frozenset(["s1"])
        assert step_info.current_state_ids == frozenset(["s2"])
        assert step_info.step_count == 5

    def test_state_changed(self) -> None:
        """Test state_changed property."""
        changed = StepInfo(
            previous_state_ids=frozenset(["s1"]),
            current_state_ids=frozenset(["s2"]),
        )
        assert changed.state_changed is True

        unchanged = StepInfo(
            previous_state_ids=frozenset(["s1"]),
            current_state_ids=frozenset(["s1"]),
        )
        assert unchanged.state_changed is False

    def test_transition_success(self) -> None:
        """Test transition_success property."""
        success = StepInfo(action_result=ActionResult(success=True))
        assert success.transition_success is True

        failure = StepInfo(action_result=ActionResult(success=False))
        assert failure.transition_success is False

    def test_workflow_name(self) -> None:
        """Test workflow_name property."""
        step_info = StepInfo(
            action_result=ActionResult(success=True, workflow_name="Login")
        )
        assert step_info.workflow_name == "Login"


class TestActionResult:
    """Tests for ActionResult dataclass."""

    def test_create_action_result(self) -> None:
        """Test creating ActionResult."""
        result = ActionResult(
            success=True,
            action_type="workflow",
            workflow_name="Login",
            duration_ms=500,
        )
        assert result.success is True
        assert result.workflow_name == "Login"
        assert result.duration_ms == 500

    def test_action_result_frozen(self) -> None:
        """Test that ActionResult is immutable."""
        result = ActionResult(success=True)
        with pytest.raises(AttributeError):
            result.success = False  # type: ignore[misc]


class TestPatternMatch:
    """Tests for PatternMatch dataclass."""

    def test_create_pattern_match(self) -> None:
        """Test creating PatternMatch."""
        match = PatternMatch(
            pattern_name="login_button",
            similarity=0.95,
            region=(100, 200, 50, 30),
            state_id="state-1",
        )
        assert match.pattern_name == "login_button"
        assert match.similarity == 0.95


class TestBaseRewardComponent:
    """Tests for BaseRewardComponent base class."""

    def test_constant_reward(self) -> None:
        """Test ConstantReward component."""
        component = ConstantReward(value=5.0)
        step_info = StepInfo()

        reward = component.compute(step_info)
        assert reward == 5.0

    def test_weighted_reward(self) -> None:
        """Test weight scaling."""
        component = ConstantReward(value=2.0, weight=3.0)
        step_info = StepInfo()

        reward = component.compute(step_info)
        assert reward == 6.0  # 2.0 * 3.0

    def test_statistics_tracking(self) -> None:
        """Test statistics are tracked correctly."""
        component = ConstantReward(value=1.0)
        step_info = StepInfo()

        component.compute(step_info)
        component.compute(step_info)

        stats = component.get_statistics()
        assert stats["call_count"] == 2
        assert stats["total_reward"] == 2.0
        assert stats["average_reward"] == 1.0

    def test_reset(self) -> None:
        """Test reset clears statistics."""
        component = ConstantReward(value=1.0)
        step_info = StepInfo()

        component.compute(step_info)
        component.reset()

        stats = component.get_statistics()
        assert stats["call_count"] == 0
        assert stats["total_reward"] == 0.0


class TestStepPenalty:
    """Tests for StepPenalty component."""

    def test_step_penalty(self) -> None:
        """Test step penalty returns constant negative reward."""
        component = StepPenalty(penalty=-0.01)
        step_info = StepInfo()

        reward = component.compute(step_info)
        assert reward == -0.01

    def test_step_penalty_default(self) -> None:
        """Test step penalty default value."""
        component = StepPenalty()
        step_info = StepInfo()

        reward = component.compute(step_info)
        assert reward == -0.001


class TestStateReachReward:
    """Tests for StateReachReward component."""

    def test_reach_target_state(self) -> None:
        """Test reward when reaching target state."""
        component = StateReachReward(
            target_states={"goal"},
            reward_per_state=10.0,
        )
        step_info = StepInfo(
            current_state_names=frozenset(["goal"]),
        )

        reward = component.compute(step_info)
        assert reward == 10.0

    def test_not_at_target(self) -> None:
        """Test no reward when not at target state."""
        component = StateReachReward(
            target_states={"goal"},
            reward_per_state=10.0,
        )
        step_info = StepInfo(
            current_state_names=frozenset(["other"]),
        )

        reward = component.compute(step_info)
        assert reward == 0.0

    def test_multiple_targets(self) -> None:
        """Test with multiple target states."""
        component = StateReachReward(
            target_states={"goal1", "goal2"},
            reward_per_state=5.0,
        )
        step_info = StepInfo(
            current_state_names=frozenset(["goal1", "goal2"]),
        )

        reward = component.compute(step_info)
        assert reward == 10.0  # 2 states * 5.0


class TestStateTransitionReward:
    """Tests for StateTransitionReward component."""

    def test_successful_transition(self) -> None:
        """Test reward for successful state transition."""
        component = StateTransitionReward(
            reward_on_transition=1.0,
            penalty_on_failure=-0.5,
        )
        step_info = StepInfo(
            previous_state_ids=frozenset(["s1"]),
            current_state_ids=frozenset(["s2"]),
            previous_state_names=frozenset(["state1"]),
            current_state_names=frozenset(["state2"]),
            action_result=ActionResult(success=True),
        )

        reward = component.compute(step_info)
        assert reward == 1.0

    def test_failed_transition(self) -> None:
        """Test penalty for failed transition."""
        component = StateTransitionReward(
            reward_on_transition=1.0,
            penalty_on_failure=-0.5,
        )
        step_info = StepInfo(
            previous_state_ids=frozenset(["s1"]),
            current_state_ids=frozenset(["s1"]),  # No change
            action_result=ActionResult(success=False),
        )

        reward = component.compute(step_info)
        assert reward == -0.5


class TestInvalidActionPenalty:
    """Tests for InvalidActionPenalty component."""

    def test_valid_action(self) -> None:
        """Test no penalty for valid action."""
        component = InvalidActionPenalty(penalty=-1.0)
        step_info = StepInfo(
            action_result=ActionResult(success=True),
        )

        reward = component.compute(step_info)
        assert reward == 0.0

    def test_invalid_action(self) -> None:
        """Test penalty for invalid action."""
        component = InvalidActionPenalty(penalty=-1.0)
        step_info = StepInfo(
            action_result=ActionResult(success=False, error="Invalid action"),
        )

        reward = component.compute(step_info)
        assert reward == -1.0


class TestGoalConditionedReward:
    """Tests for GoalConditionedReward component."""

    def test_goal_achieved(self) -> None:
        """Test reward when goal is achieved."""
        component = GoalConditionedReward(goal_reward=100.0)
        step_info = StepInfo(
            goal_state="goal",
            current_state_names=frozenset(["goal"]),
            goal_achieved=True,
        )

        reward = component.compute(step_info)
        assert reward == 100.0

    def test_goal_not_achieved(self) -> None:
        """Test no reward when goal not achieved."""
        component = GoalConditionedReward(goal_reward=100.0)
        step_info = StepInfo(
            goal_state="goal",
            current_state_names=frozenset(["other"]),
            goal_achieved=False,
        )

        reward = component.compute(step_info)
        assert reward == 0.0


class TestStateVisitCountReward:
    """Tests for StateVisitCountReward component."""

    def test_new_state_bonus(self) -> None:
        """Test bonus for visiting new state."""
        component = StateVisitCountReward(beta=1.0)
        step_info = StepInfo(
            current_state_ids=frozenset(["new_state"]),
        )

        # First visit should give bonus
        reward1 = component.compute(step_info)
        assert reward1 > 0.0

        # Second visit should give less bonus
        reward2 = component.compute(step_info)
        assert reward2 < reward1

    def test_exploration_reset(self) -> None:
        """Test exploration bonus resets correctly."""
        component = StateVisitCountReward(beta=1.0)
        step_info = StepInfo(
            current_state_ids=frozenset(["state"]),
        )

        component.compute(step_info)
        component.reset()

        # After reset, should be like first visit again
        reward = component.compute(step_info)
        assert reward == 1.0  # beta / sqrt(1)


class TestScreenChangeReward:
    """Tests for ScreenChangeReward component."""

    def test_screen_changed(
        self,
        sample_screenshot: np.ndarray,
        sample_screenshot_with_content: np.ndarray,
    ) -> None:
        """Test reward when screen changes."""
        component = ScreenChangeReward(
            change_threshold=0.01,
            reward_on_change=1.0,
        )
        step_info = StepInfo(
            screenshot_before=sample_screenshot,
            screenshot_after=sample_screenshot_with_content,
        )

        reward = component.compute(step_info)
        assert reward == 1.0

    def test_screen_unchanged(
        self,
        sample_screenshot: np.ndarray,
    ) -> None:
        """Test no reward when screen doesn't change."""
        component = ScreenChangeReward(
            change_threshold=0.01,
            reward_on_change=1.0,
        )
        step_info = StepInfo(
            screenshot_before=sample_screenshot,
            screenshot_after=sample_screenshot.copy(),
        )

        reward = component.compute(step_info)
        assert reward == 0.0  # No change means no reward


class TestComposedReward:
    """Tests for ComposedReward class."""

    def test_sum_composition(self) -> None:
        """Test summing multiple rewards."""
        composed = ComposedReward(
            components=[
                ConstantReward(value=1.0),
                ConstantReward(value=2.0),
                ConstantReward(value=3.0),
            ],
            mode=CompositionMode.SUM,
        )
        step_info = StepInfo()

        reward = composed(step_info)
        assert reward == 6.0

    def test_weighted_sum(self) -> None:
        """Test weighted sum composition."""
        composed = ComposedReward(
            components=[
                ConstantReward(value=1.0, weight=2.0),
                ConstantReward(value=1.0, weight=3.0),
            ],
            mode=CompositionMode.SUM,
        )
        step_info = StepInfo()

        reward = composed(step_info)
        assert reward == 5.0  # 1*2 + 1*3

    def test_min_composition(self) -> None:
        """Test min composition."""
        composed = ComposedReward(
            components=[
                ConstantReward(value=5.0),
                ConstantReward(value=2.0),
                ConstantReward(value=8.0),
            ],
            mode=CompositionMode.MIN,
        )
        step_info = StepInfo()

        reward = composed(step_info)
        assert reward == 2.0

    def test_max_composition(self) -> None:
        """Test max composition."""
        composed = ComposedReward(
            components=[
                ConstantReward(value=5.0),
                ConstantReward(value=2.0),
                ConstantReward(value=8.0),
            ],
            mode=CompositionMode.MAX,
        )
        step_info = StepInfo()

        reward = composed(step_info)
        assert reward == 8.0

    def test_reset_components(self) -> None:
        """Test reset propagates to all components."""
        c1 = ConstantReward(value=1.0)
        c2 = ConstantReward(value=2.0)
        composed = ComposedReward(components=[c1, c2])

        step_info = StepInfo()
        composed(step_info)
        composed.reset()

        assert c1.get_statistics()["call_count"] == 0
        assert c2.get_statistics()["call_count"] == 0


class TestRewardBuilder:
    """Tests for RewardBuilder fluent API."""

    def test_builder_basic(self) -> None:
        """Test basic builder usage."""
        reward = RewardBuilder().step_penalty(-0.01).build()

        step_info = StepInfo()
        assert reward(step_info) == -0.01

    def test_builder_goal_reaching(self) -> None:
        """Test goal reaching builder method."""
        reward = RewardBuilder().goal_reaching({"goal"}, reward=100.0).build()

        # At goal
        at_goal = StepInfo(current_state_names=frozenset(["goal"]))
        assert reward(at_goal) == 100.0

        # Not at goal
        not_at_goal = StepInfo(current_state_names=frozenset(["other"]))
        assert reward(not_at_goal) == 0.0

    def test_builder_exploration(self) -> None:
        """Test exploration bonus builder method."""
        reward = RewardBuilder().exploration_bonus(beta=2.0).build()

        step_info = StepInfo(current_state_ids=frozenset(["s1"]))
        first = reward(step_info)
        assert first == 2.0  # 2.0 / sqrt(1)

    def test_builder_transition_reward(self) -> None:
        """Test transition reward builder method."""
        reward = (
            RewardBuilder()
            .transition_reward(success_reward=1.0, failure_penalty=-0.5)
            .build()
        )

        success = StepInfo(
            previous_state_ids=frozenset(["s1"]),
            current_state_ids=frozenset(["s2"]),
            previous_state_names=frozenset(["state1"]),
            current_state_names=frozenset(["state2"]),
            action_result=ActionResult(success=True),
        )
        assert reward(success) == 1.0

    def test_builder_chain(self) -> None:
        """Test chaining multiple builder methods."""
        reward = (
            RewardBuilder().step_penalty(-0.001).exploration_bonus(beta=0.1).build()
        )

        step_info = StepInfo(current_state_ids=frozenset(["s1"]))
        result = reward(step_info)

        # Should be step_penalty + exploration_bonus
        expected = -0.001 + 0.1  # 0.1 / sqrt(1)
        assert abs(result - expected) < 1e-6

    def test_builder_custom_component(self) -> None:
        """Test adding custom component."""
        custom = ConstantReward(value=42.0)
        reward = RewardBuilder().custom(custom).build()

        step_info = StepInfo()
        assert reward(step_info) == 42.0
