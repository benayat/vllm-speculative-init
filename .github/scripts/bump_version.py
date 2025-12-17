#!/usr/bin/env python3
"""
Bump version script for automated PyPI publishing workflow.

This script:
1. Reads the current version from pyproject.toml
2. Determines the bump level from PR labels and commit messages
3. Bumps the version accordingly
4. Writes the new version back to pyproject.toml
5. Commits and pushes the change

Environment variables expected:
- GITHUB_EVENT_PATH: Path to GitHub event JSON
- GITHUB_TOKEN: GitHub token for pushing
- GITHUB_REPOSITORY: Repository name (owner/repo)
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

try:
    # Python 3.11+ has tomllib in standard library
    import tomllib
except ImportError:
    # Python < 3.11: use tomli (installed via pip)
    import tomli as tomllib


def read_pyproject_toml(path):
    """Read and parse pyproject.toml file."""
    with open(path, "rb") as f:
        return tomllib.load(f)


def write_pyproject_toml(path, data, new_version):
    """Write updated version to pyproject.toml while preserving formatting."""
    with open(path, "r") as f:
        content = f.read()
    
    # Use regex to replace version line
    new_content = re.sub(
        r'^version\s*=\s*"[^"]*"',
        f'version = "{new_version}"',
        content,
        flags=re.MULTILINE
    )
    
    with open(path, "w") as f:
        f.write(new_content)


def parse_version(version_str):
    """Parse a semantic version string into (major, minor, patch)."""
    parts = version_str.split(".")
    if len(parts) != 3:
        raise ValueError(f"Invalid version format: {version_str}")
    return tuple(int(p) for p in parts)


def bump_version(version_str, bump_type):
    """Bump version according to bump_type (major, minor, patch)."""
    major, minor, patch = parse_version(version_str)
    
    if bump_type == "major":
        return f"{major + 1}.0.0"
    elif bump_type == "minor":
        return f"{major}.{minor + 1}.0"
    elif bump_type == "patch":
        return f"{major}.{minor}.{patch + 1}"
    else:
        raise ValueError(f"Invalid bump type: {bump_type}")


def get_bump_level_from_label(labels):
    """
    Determine bump level from PR labels.
    Returns 'major', 'minor', 'patch', or None if no relevant label found.
    """
    label_names = [label.get("name", "").lower() for label in labels]
    
    if "major" in label_names:
        return "major"
    elif "minor" in label_names:
        return "minor"
    elif "patch" in label_names:
        return "patch"
    
    return None


def get_bump_level_from_commits(commits, pr_title="", pr_body=""):
    """
    Determine bump level from commit messages, PR title, and PR body.
    
    Rules:
    - "BREAKING CHANGE" or "feat!" -> major
    - "feat:" -> minor
    - Everything else -> patch
    """
    all_text = "\n".join([
        pr_title,
        pr_body,
        *[commit.get("message", "") for commit in commits]
    ])
    
    # Check for breaking changes
    if "BREAKING CHANGE" in all_text or "BREAKING-CHANGE" in all_text:
        return "major"
    
    # Check for feat! pattern (conventional commit with !)
    if re.search(r'\b\w+!\s*:', all_text):
        return "major"
    
    # Check for feat: (new feature)
    if re.search(r'\bfeat\s*:', all_text, re.IGNORECASE):
        return "minor"
    
    # Default to patch
    return "patch"


def get_event_data():
    """Read and parse GitHub event data."""
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if not event_path:
        raise ValueError("GITHUB_EVENT_PATH not set")
    
    with open(event_path, "r") as f:
        return json.load(f)


def get_commits_from_event(event_data):
    """Extract commits from GitHub event data."""
    commits = []
    
    # For push events
    if "commits" in event_data:
        commits = event_data["commits"]
    
    # For pull_request events
    elif "pull_request" in event_data:
        pr = event_data["pull_request"]
        # We'll use the PR title and body as signals
        # Note: The actual commits are not in the event payload for pull_request events
        # We would need to fetch them via API, but for simplicity we'll use title/body
        commits = [{
            "message": pr.get("title", "") + "\n" + pr.get("body", "")
        }]
    
    return commits


def should_skip_bump(event_data):
    """
    Check if we should skip the bump (e.g., if this commit is itself a version bump).
    """
    # Check for push events with [skip ci] in the head commit
    if "head_commit" in event_data:
        head_commit = event_data["head_commit"]
        message = head_commit.get("message", "")
        if "[skip ci]" in message or "[ci skip]" in message:
            return True
    
    # Check for pull_request events
    if "pull_request" in event_data:
        pr = event_data["pull_request"]
        if not pr.get("merged", False):
            return True
    
    return False


def run_command(cmd, check=True):
    """Run a shell command and return output."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=check)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result


def main():
    """Main entry point."""
    print("=== Version Bump Script ===")
    
    # Read event data
    try:
        event_data = get_event_data()
    except Exception as e:
        print(f"Error reading event data: {e}")
        print("This script must be run in GitHub Actions with GITHUB_EVENT_PATH set")
        sys.exit(1)
    
    # Check if we should skip
    if should_skip_bump(event_data):
        print("Skipping version bump (already a CI commit or unmerged PR)")
        sys.exit(0)
    
    # Get pyproject.toml path
    repo_root = Path(__file__).parent.parent.parent
    pyproject_path = repo_root / "pyproject.toml"
    
    if not pyproject_path.exists():
        print(f"Error: pyproject.toml not found at {pyproject_path}")
        sys.exit(1)
    
    # Read current version
    try:
        data = read_pyproject_toml(pyproject_path)
        current_version = data["project"]["version"]
        print(f"Current version: {current_version}")
    except Exception as e:
        print(f"Error reading pyproject.toml: {e}")
        sys.exit(1)
    
    # Determine bump level
    bump_level = None
    
    # Priority 1: PR labels
    if "pull_request" in event_data:
        pr = event_data["pull_request"]
        labels = pr.get("labels", [])
        bump_level = get_bump_level_from_label(labels)
        if bump_level:
            print(f"Bump level from PR label: {bump_level}")
    
    # Priority 2: Commit messages and PR title/body
    if not bump_level:
        commits = get_commits_from_event(event_data)
        pr_title = ""
        pr_body = ""
        
        if "pull_request" in event_data:
            pr = event_data["pull_request"]
            pr_title = pr.get("title", "")
            pr_body = pr.get("body", "")
        
        bump_level = get_bump_level_from_commits(commits, pr_title, pr_body)
        print(f"Bump level from commits: {bump_level}")
    
    # Bump version
    new_version = bump_version(current_version, bump_level)
    print(f"New version: {new_version}")
    
    # Write new version
    try:
        write_pyproject_toml(pyproject_path, data, new_version)
        print(f"Updated {pyproject_path}")
    except Exception as e:
        print(f"Error writing pyproject.toml: {e}")
        sys.exit(1)
    
    # Configure git
    run_command(["git", "config", "user.name", "github-actions[bot]"])
    run_command(["git", "config", "user.email", "github-actions[bot]@users.noreply.github.com"])
    
    # Commit and push
    commit_message = f"ci: bump version to {new_version} [skip ci]"
    
    run_command(["git", "add", str(pyproject_path)])
    run_command(["git", "commit", "-m", commit_message])
    
    # Push using GITHUB_TOKEN
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("Error: GITHUB_TOKEN not set")
        sys.exit(1)
    
    repo = os.environ.get("GITHUB_REPOSITORY")
    if not repo:
        print("Error: GITHUB_REPOSITORY not set")
        sys.exit(1)
    
    # Push with authentication
    # Get the current branch name
    branch_result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True,
        check=True
    )
    current_branch = branch_result.stdout.strip()
    
    push_url = f"https://x-access-token:{token}@github.com/{repo}.git"
    run_command(["git", "push", push_url, f"HEAD:{current_branch}"])
    
    print(f"Successfully bumped version to {new_version}")


if __name__ == "__main__":
    main()
