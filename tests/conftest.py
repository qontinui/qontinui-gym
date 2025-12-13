"""Shared fixtures for qontinui-gym tests."""

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest


@pytest.fixture
def sample_config_dict() -> dict[str, Any]:
    """Sample QontinuiConfig as a dictionary."""
    return {
        "version": "2.7.0",
        "metadata": {
            "name": "Test Config",
            "description": "A test configuration",
            "author": "Test Author",
            "created": "2024-01-01T00:00:00Z",
            "modified": "2024-01-01T00:00:00Z",
            "targetApplication": "Test App",
        },
        "workflows": [
            {
                "id": "workflow-1",
                "name": "Login",
                "description": "Login to the application",
                "category": "Authentication",
            },
            {
                "id": "workflow-2",
                "name": "Logout",
                "description": "Logout from the application",
                "category": "Authentication",
            },
            {
                "id": "workflow-3",
                "name": "Navigate Home",
                "description": "Navigate to home page",
                "category": "Navigation",
            },
        ],
        "states": [
            {
                "id": "state-1",
                "name": "Login Page",
                "description": "The login page",
                "stateImages": ["img1.png"],
                "isInitial": True,
                "isFinal": False,
            },
            {
                "id": "state-2",
                "name": "Home Page",
                "description": "The home page",
                "stateImages": ["img2.png", "img3.png"],
                "isInitial": False,
                "isFinal": False,
            },
            {
                "id": "state-3",
                "name": "Logout Complete",
                "description": "Successfully logged out",
                "stateImages": [],
                "isInitial": False,
                "isFinal": True,
            },
        ],
        "transitions": [
            {
                "id": "trans-1",
                "type": "OutgoingTransition",
                "fromState": "state-1",
                "toState": "state-2",
                "workflows": ["workflow-1"],
                "priority": 0,
            },
            {
                "id": "trans-2",
                "type": "OutgoingTransition",
                "fromState": "state-2",
                "toState": "state-3",
                "workflows": ["workflow-2"],
                "priority": 0,
            },
            {
                "id": "trans-3",
                "type": "OutgoingTransition",
                "fromState": "state-2",
                "toState": "state-2",
                "workflows": ["workflow-3"],
                "priority": 1,
            },
        ],
        "categories": ["Authentication", "Navigation"],
        "settings": {
            "execution": {
                "defaultTimeout": 10000,
                "defaultRetryCount": 3,
                "failureStrategy": "continue",
            }
        },
    }


@pytest.fixture
def sample_config_file(sample_config_dict: dict[str, Any]) -> Path:
    """Create a temporary config file."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(sample_config_dict, f)
        return Path(f.name)


@pytest.fixture
def sample_screenshot() -> np.ndarray:
    """Create a sample screenshot array."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_screenshot_with_content() -> np.ndarray:
    """Create a sample screenshot with some content."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add some variation
    img[100:200, 100:200] = [255, 0, 0]  # Red square
    img[200:300, 300:400] = [0, 255, 0]  # Green square
    return img
