#!/usr/bin/env python3
"""Pre-commit hook to block commit messages with Claude attribution."""

import sys


def main() -> int:
    """Check commit message for Claude attribution patterns."""
    if len(sys.argv) < 2:
        return 0

    commit_msg_file = sys.argv[1]
    try:
        with open(commit_msg_file, encoding="utf-8") as f:
            commit_msg = f.read()
    except FileNotFoundError:
        return 0

    # Check for Claude attribution patterns
    if "Generated with [Claude Code]" in commit_msg:
        print("ERROR: Commit message contains Claude attribution")
        print("Please remove the 'Generated with [Claude Code]' line")
        print("from your commit message")
        return 1

    if "Co-Authored-By: Claude" in commit_msg:
        print("ERROR: Commit message contains Claude co-author attribution")
        print("Please remove the 'Co-Authored-By: Claude' line")
        print("from your commit message")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
