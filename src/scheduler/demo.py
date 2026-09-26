"""Create an isolated, reproducible workspace for a recorded live demo."""

import argparse
from pathlib import Path

from .policy import activate_policy, load_policy
from .repository import SchedulingRepository


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a fresh clinic-agent demo workspace.")
    parser.add_argument("directory", help="New directory, for example artifacts/demo-001")
    parser.add_argument("--base-policy", default="policies/v1.json")
    args = parser.parse_args()
    workspace = Path(args.directory)
    if workspace.exists():
        parser.error(f"workspace already exists: {workspace}")
    workspace.mkdir(parents=True)
    for name in ("traces", "proposals", "evaluation-traces", "regressions"):
        (workspace / name).mkdir()
    base = load_policy(args.base_policy)
    activate_policy(base, workspace / "policies")
    repository = SchedulingRepository(workspace / "scheduler.db")
    repository.close()
    print(f"Demo workspace ready: {workspace}")
    print(f"Chat: scheduler-agent --workspace {workspace}")
    print(f"Review: scheduler-admin --workspace {workspace} list")


if __name__ == "__main__":
    main()
