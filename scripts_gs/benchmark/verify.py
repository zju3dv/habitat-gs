"""Verify public bundle checksums, schemas and optionally local scene assets."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def verify_files(root, files):
    root = root.resolve()
    for name, expected in files.items():
        path = (root / name).resolve()
        if Path(name).is_absolute() or not path.is_relative_to(root):
            raise ValueError(f"unsafe manifest path: {name}")
        if not path.is_file():
            raise ValueError(f"missing file: {name}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected["sha256"] or path.stat().st_size != expected["bytes"]:
            raise ValueError(f"checksum/size mismatch: {name}")
    return len(files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--scene-root", type=Path)
    args = parser.parse_args()
    from jsonschema import Draft202012Validator

    manifest = json.loads((args.bundle / "manifest.json").read_text())
    count = verify_files(args.bundle, manifest["files"])
    tasks = {}
    for task in ("objectnav", "multi-objectnav"):
        folder = args.bundle / task / "v1"
        schema = json.loads((folder / "episode_v1.schema.json").read_text())
        validator = Draft202012Validator(schema)
        episodes = json.loads((folder / "episodes.json").read_text())["episodes"]
        for episode in episodes:
            validator.validate(episode)
        if len(episodes) != 150 or len({e["episode_id"] for e in episodes}) != 150:
            raise ValueError(f"expected 150 distinct episodes: {task}")
        tasks[task] = len(episodes)
    assets = None
    if args.scene_root:
        scene_manifest = json.loads((args.bundle / "scene-assets/v1/manifest.json").read_text())
        assets = verify_files(args.scene_root, scene_manifest["files"])
    print(json.dumps({"verified_bundle_files": count, "episodes": tasks, "verified_scene_files": assets}))


if __name__ == "__main__":
    main()
