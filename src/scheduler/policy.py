"""Small, versioned agent policies loaded from JSON files."""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AgentPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str = Field(min_length=1)
    system_prompt: str = Field(min_length=1)


def load_policy(path: str | Path) -> AgentPolicy:
    with Path(path).open(encoding="utf-8") as policy_file:
        return AgentPolicy.model_validate(json.load(policy_file))


def save_policy(policy: AgentPolicy, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(policy.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )


def load_active_policy(policy_directory: str | Path) -> AgentPolicy:
    """Load the explicitly promoted policy, falling back to v1 on first run."""
    directory = Path(policy_directory)
    pointer = directory / "active.json"
    if not pointer.exists():
        return load_policy(directory / "v1.json")

    active: Any = json.loads(pointer.read_text(encoding="utf-8"))
    if not isinstance(active, dict) or set(active) != {"version", "file"}:
        raise ValueError("Active policy pointer must contain only version and file.")
    filename = active["file"]
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError("Active policy file must be a filename within the policy directory.")
    policy = load_policy(directory / filename)
    if policy.version != active["version"]:
        raise ValueError("Active policy pointer version does not match its policy file.")
    return policy


def activate_policy(policy: AgentPolicy, policy_directory: str | Path) -> Path:
    """Atomically point future live runs at a previously saved policy version."""
    directory = Path(policy_directory)
    policy_path = directory / f"{policy.version}.json"
    if not policy_path.exists():
        save_policy(policy, policy_path)
    elif load_policy(policy_path) != policy:
        raise ValueError(f"Policy version {policy.version} already exists with different content.")

    pointer = directory / "active.json"
    temporary = pointer.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps({"version": policy.version, "file": policy_path.name}, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(pointer)
    return policy_path
