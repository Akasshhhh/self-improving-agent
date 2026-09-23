"""Small, versioned agent policies loaded from JSON files."""

import json
from pathlib import Path

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

