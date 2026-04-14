#!/usr/bin/env python3
"""Adapt a habitat-baselines actor-critic checkpoint for the
``ddppo.pretrained_weights`` load path.

Background
----------
Habitat-baselines saves checkpoints whose ``state_dict`` keys are the
direct module names (``net.*``, ``critic.*``, ``action_distribution.*``)
because ``self._actor_critic.state_dict()`` is called on the actor-critic
module itself. When such a checkpoint is loaded as ``pretrained_weights``,
``single_agent_access_mgr.py`` blindly strips a 13-char ``actor_critic.``
prefix from every key, mangling them.

This helper rewrites the keys with an explicit ``actor_critic.`` prefix
so that the strip turns them back into the original names. Checkpoints
that already carry the prefix are passed through unchanged.

Usage
-----
    python scripts_gs/_adapt_pretrained_ckpt.py SRC.pth DST.pth
"""
import sys

import torch


def main() -> None:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} SRC.pth DST.pth", file=sys.stderr)
        sys.exit(2)
    src, dst = sys.argv[1], sys.argv[2]

    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    if "state_dict" not in ckpt:
        print(
            f"ERROR: {src} has no 'state_dict' key (top-level keys: "
            f"{list(ckpt.keys())})",
            file=sys.stderr,
        )
        sys.exit(1)

    sd = ckpt["state_dict"]
    if not any(k.startswith("actor_critic.") for k in sd):
        ckpt["state_dict"] = {f"actor_critic.{k}": v for k, v in sd.items()}

    torch.save(ckpt, dst)


if __name__ == "__main__":
    main()
