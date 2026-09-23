"""Write completed, JSON-serializable run traces to disk."""

import json
from pathlib import Path
from typing import Any


class TraceStore:
    def __init__(self, directory: str | Path) -> None:
        self._directory = Path(directory)

    def save(self, trace: dict[str, Any]) -> Path:
        self._directory.mkdir(parents=True, exist_ok=True)
        target = self._directory / f"{trace['run_id']}.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")
        temporary.replace(target)
        return target
