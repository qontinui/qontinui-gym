"""Synchronous HTTP client for qontinui-runner.

This module provides a synchronous HTTP client for communicating with
the qontinui-runner application, which handles workflow execution and
state management.
"""

from __future__ import annotations

import logging
import os
import platform
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import requests
from PIL import Image
from requests.exceptions import ConnectionError, Timeout

from qontinui_gym.types import MonitorInfo, StateDetectionResult

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30.0
EXECUTION_TIMEOUT = 300.0
DEFAULT_PORT = 9876


@dataclass
class WorkflowResult:
    """Result of workflow execution."""

    success: bool
    workflow_name: str
    duration_ms: int = 0
    error: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class GoToStateResult:
    """Result of state navigation."""

    success: bool
    target_states: list[str]
    screenshot_path: str | None = None
    duration_ms: int = 0
    error: str | None = None
    monitors_used: list[int] | None = None


def _get_windows_host() -> str:
    """Get the Windows host IP for WSL environments.

    Returns localhost for native Windows/Mac, or the Windows host IP for WSL.
    """
    # Check environment variable first
    env_host = os.environ.get("QONTINUI_RUNNER_HOST")
    if env_host:
        return env_host

    # Detect WSL
    if platform.system() == "Linux":
        try:
            resolv_path = Path("/etc/resolv.conf")
            if resolv_path.exists():
                content = resolv_path.read_text()
                # Look for nameserver line (Windows host in WSL2)
                match = re.search(r"nameserver\s+(\d+\.\d+\.\d+\.\d+)", content)
                if match:
                    return match.group(1)
        except Exception:
            pass

    return "localhost"


def _convert_wsl_path(path: str) -> str:
    """Convert WSL path to Windows path if needed.

    Args:
        path: Path that may be a WSL path (/mnt/c/...)

    Returns:
        Windows-compatible path (C:\\...)
    """
    if path.startswith("/mnt/") and len(path) > 6:
        drive = path[5].upper()
        rest = path[7:].replace("/", "\\")
        return f"{drive}:\\{rest}"
    return path


class RunnerClient:
    """Synchronous HTTP client for qontinui-runner.

    This client provides methods to interact with the qontinui-runner
    HTTP API for workflow execution, state navigation, and monitoring.

    Example:
        client = RunnerClient()
        if client.health():
            client.load_config("/path/to/config.json")
            result = client.run_workflow("LoginWorkflow")
            print(f"Success: {result.success}")
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        execution_timeout: float = EXECUTION_TIMEOUT,
    ):
        """Initialize the runner client.

        Args:
            host: Runner host (default: auto-detect for WSL)
            port: Runner port (default: 9876)
            timeout: Default timeout for API calls in seconds
            execution_timeout: Timeout for workflow execution in seconds
        """
        self.host = host or _get_windows_host()
        self.port = port or int(os.environ.get("QONTINUI_RUNNER_PORT", DEFAULT_PORT))
        self.base_url = f"http://{self.host}:{self.port}"
        self.timeout = timeout
        self.execution_timeout = execution_timeout
        self._session = requests.Session()
        self._monitors_cache: list[MonitorInfo] | None = None

    def health(self) -> bool:
        """Check if the runner is healthy and responding.

        Returns:
            True if runner is healthy, False otherwise
        """
        try:
            resp = self._session.get(
                f"{self.base_url}/health",
                timeout=self.timeout,
            )
            return bool(resp.status_code == 200)
        except (ConnectionError, Timeout):
            return False

    def status(self) -> dict[str, Any]:
        """Get the current status of the runner.

        Returns:
            Status dict with keys: executor_running, executor_state,
            config_loaded, config_path

        Raises:
            ConnectionError: If unable to connect to runner
            requests.HTTPError: If runner returns an error
        """
        resp = self._session.get(
            f"{self.base_url}/status",
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        return cast(dict[str, Any], data.get("data", {}))

    def load_config(self, config_path: str) -> bool:
        """Load a workflow configuration file.

        Args:
            config_path: Path to the QontinuiConfig JSON file

        Returns:
            True if config loaded successfully

        Raises:
            FileNotFoundError: If config file doesn't exist
            requests.HTTPError: If runner returns an error
        """
        # Convert WSL path if needed
        windows_path = _convert_wsl_path(config_path)

        logger.info(f"Loading config: {windows_path}")
        resp = self._session.post(
            f"{self.base_url}/load-config",
            json={"config_path": windows_path},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        return bool(data.get("success", False))

    def run_workflow(
        self,
        workflow_name: str,
        monitor: int | str | None = None,
        monitors: list[int | str] | None = None,
        timeout_seconds: int | None = None,
    ) -> WorkflowResult:
        """Execute a workflow by name.

        Args:
            workflow_name: Name of the workflow to execute
            monitor: Single monitor (index or position like "left", "primary")
            monitors: List of monitors for multi-monitor capture
            timeout_seconds: Execution timeout (default: execution_timeout)

        Returns:
            WorkflowResult with execution status and details
        """
        start_time = time.time()
        timeout = timeout_seconds or int(self.execution_timeout)

        payload: dict[str, Any] = {
            "workflow_name": workflow_name,
            "timeout_seconds": timeout,
        }

        # Resolve monitor(s)
        if monitors is not None:
            indices = [self._resolve_monitor(m) for m in monitors]
            payload["monitor_indices"] = indices
        elif monitor is not None:
            payload["monitor_index"] = self._resolve_monitor(monitor)

        try:
            resp = self._session.post(
                f"{self.base_url}/run-workflow",
                json=payload,
                timeout=timeout + 30,  # Extra buffer for HTTP
            )
            resp.raise_for_status()
            resp_data: dict[str, Any] = resp.json()
            data: dict[str, Any] = resp_data.get("data", {})

            duration_ms = int((time.time() - start_time) * 1000)

            return WorkflowResult(
                success=data.get("success", False),
                workflow_name=workflow_name,
                duration_ms=duration_ms,
                error=data.get("error"),
                events=data.get("events", []),
            )
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            return WorkflowResult(
                success=False,
                workflow_name=workflow_name,
                duration_ms=duration_ms,
                error=str(e),
            )

    def go_to_state(
        self,
        state_names: list[str],
        monitor: int | str | None = None,
        take_screenshot: bool = True,
        timeout_seconds: float = 120.0,
    ) -> GoToStateResult:
        """Navigate to specific states using the state machine.

        Args:
            state_names: List of target state names
            monitor: Monitor to use (index or position)
            take_screenshot: Whether to capture screenshot after navigation
            timeout_seconds: Navigation timeout

        Returns:
            GoToStateResult with navigation status
        """
        start_time = time.time()

        payload: dict[str, Any] = {
            "state_names": state_names,
            "take_screenshot": take_screenshot,
            "timeout_seconds": timeout_seconds,
        }

        if monitor is not None:
            payload["monitor_index"] = self._resolve_monitor(monitor)

        try:
            resp = self._session.post(
                f"{self.base_url}/go-to-state",
                json=payload,
                timeout=timeout_seconds + 30,
            )
            resp.raise_for_status()
            resp_data_gts: dict[str, Any] = resp.json()
            data_gts: dict[str, Any] = resp_data_gts.get("data", {})

            duration_ms = int((time.time() - start_time) * 1000)

            return GoToStateResult(
                success=bool(data_gts.get("success", False)),
                target_states=state_names,
                screenshot_path=data_gts.get("screenshot_path"),
                duration_ms=duration_ms,
                error=data_gts.get("error"),
                monitors_used=data_gts.get("monitors_used"),
            )
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            return GoToStateResult(
                success=False,
                target_states=state_names,
                duration_ms=duration_ms,
                error=str(e),
            )

    def list_monitors(self, refresh: bool = False) -> list[MonitorInfo]:
        """Get list of available monitors.

        Args:
            refresh: Force refresh of cached monitor list

        Returns:
            List of MonitorInfo objects
        """
        if self._monitors_cache is not None and not refresh:
            return self._monitors_cache

        resp = self._session.get(
            f"{self.base_url}/monitors",
            timeout=self.timeout,
        )
        resp.raise_for_status()
        resp_data_mon: dict[str, Any] = resp.json()
        data_mon: dict[str, Any] = resp_data_mon.get("data", {})

        monitors: list[MonitorInfo] = []
        for m in data_mon.get("monitors", []):
            monitors.append(
                MonitorInfo(
                    index=m.get("index", 0),
                    x=m.get("x", 0),
                    y=m.get("y", 0),
                    width=m.get("width", 1920),
                    height=m.get("height", 1080),
                    is_primary=m.get("is_primary", False),
                    position=m.get("position", "unknown"),
                    name=m.get("name", f"Monitor {m.get('index', 0)}"),
                    description=m.get("description", ""),
                )
            )

        self._monitors_cache = monitors
        return monitors

    def list_states(self) -> list[dict[str, Any]]:
        """Get list of available states from loaded config.

        Returns:
            List of state dictionaries with id, name, description, etc.
        """
        resp = self._session.get(
            f"{self.base_url}/list-states",
            timeout=self.timeout,
        )
        resp.raise_for_status()
        resp_data_ls: dict[str, Any] = resp.json()
        data_ls: dict[str, Any] = resp_data_ls.get("data", {})
        states: list[dict[str, Any]] = data_ls.get("states", [])
        return states

    def list_images(self) -> list[dict[str, Any]]:
        """Get list of all images from loaded config.

        Returns:
            List of image dictionaries with id, name, state info, etc.
        """
        resp = self._session.get(
            f"{self.base_url}/list-images",
            timeout=self.timeout,
        )
        resp.raise_for_status()
        resp_data_li: dict[str, Any] = resp.json()
        data_li: dict[str, Any] = resp_data_li.get("data", {})
        images: list[dict[str, Any]] = data_li.get("images", [])
        return images

    def click_image(
        self,
        image_name: str,
        click_type: str = "single",
        timeout_seconds: float = 30.0,
    ) -> dict[str, Any]:
        """Click on a specific image element.

        Args:
            image_name: Name of the StateImage to click
            click_type: "single", "double", or "right"
            timeout_seconds: Timeout for finding and clicking

        Returns:
            Result dict with success status and click location
        """
        resp = self._session.post(
            f"{self.base_url}/click-image",
            json={
                "image_name": image_name,
                "click_type": click_type,
                "timeout_seconds": timeout_seconds,
            },
            timeout=timeout_seconds + 10,
        )
        resp.raise_for_status()
        resp_data_ci: dict[str, Any] = resp.json()
        return cast(dict[str, Any], resp_data_ci.get("data", {}))

    def stop_execution(self) -> bool:
        """Stop the currently running workflow.

        Returns:
            True if stop command was acknowledged
        """
        try:
            resp = self._session.post(
                f"{self.base_url}/stop-execution",
                timeout=self.timeout,
            )
            resp.raise_for_status()
            resp_data_se: dict[str, Any] = resp.json()
            return bool(resp_data_se.get("success", False))
        except Exception:
            return False

    def take_screenshot(
        self,
        monitor: int | str | None = None,
    ) -> npt.NDArray[np.uint8] | None:
        """Capture current screen state.

        Args:
            monitor: Monitor to capture (index or position)

        Returns:
            Screenshot as numpy array (RGB), or None if failed

        Note:
            This uses go_to_state with empty state list to trigger screenshot.
        """
        result = self.go_to_state(
            state_names=[],
            monitor=monitor,
            take_screenshot=True,
            timeout_seconds=10.0,
        )

        if result.screenshot_path:
            try:
                img = Image.open(result.screenshot_path)
                return np.array(img, dtype=np.uint8)
            except Exception as e:
                logger.warning(f"Failed to load screenshot: {e}")

        return None

    def detect_state(
        self,
        monitor: int | str | None = None,
    ) -> StateDetectionResult:
        """Detect the current application state.

        Args:
            monitor: Monitor to analyze

        Returns:
            StateDetectionResult with detected states and confidences
        """
        # Use list-states endpoint which returns current state info
        states = self.list_states()

        # This is a simplified implementation - full state detection
        # would require a dedicated endpoint in the runner
        return StateDetectionResult(
            state_ids=[s.get("id", "") for s in states if s.get("active", False)],
            state_names=[s.get("name", "") for s in states if s.get("active", False)],
            confidences={},
            patterns_matched=[],
        )

    def close(self) -> None:
        """Close the HTTP session."""
        self._session.close()

    def _resolve_monitor(self, monitor: int | str) -> int:
        """Resolve monitor name/position to index.

        Args:
            monitor: Monitor index or position string

        Returns:
            Monitor index
        """
        if isinstance(monitor, int):
            return monitor

        monitors = self.list_monitors()
        for m in monitors:
            if m.position.lower() == monitor.lower():
                return m.index
            if m.name.lower() == monitor.lower():
                return m.index

        # Default to primary
        logger.warning(f"Monitor '{monitor}' not found, using primary")
        for m in monitors:
            if m.is_primary:
                return m.index

        return 0
