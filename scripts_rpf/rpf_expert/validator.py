"""Lightweight validation for expert episode records."""

from __future__ import annotations


REQUIRED_RECORD_KEYS = {
    "robot_state",
    "expert_trajectory_world",
    "expert_trajectory_local",
    "expert_action",
    "labels",
}


def validate_record(record: dict) -> None:
    missing = REQUIRED_RECORD_KEYS - set(record)
    if missing:
        raise ValueError(f"Episode record missing keys: {sorted(missing)}")
    if not record["expert_trajectory_world"]:
        raise ValueError("expert_trajectory_world is empty")
    if not record["expert_trajectory_local"]:
        raise ValueError("expert_trajectory_local is empty")
