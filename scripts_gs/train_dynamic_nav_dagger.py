"""Online behavior cloning of the tracking oracle into PointNavResNetPolicy.

Teacher-forced collection with truncated BPTT: each env step the policy net is run
(keeping the graph), the oracle action is the cross-entropy label AND the action sent
to the env; every SEG_T steps the accumulated loss is backpropagated and the hidden
state detached. Saves habitat-format checkpoints for eval / ddppo warm-start.
"""
import os
import sys

import numpy as np
import torch

from habitat.config.default_structured_configs import HabitatConfigPlugin, register_hydra_plugin
from habitat_baselines.config.default_structured_configs import HabitatBaselinesConfigPlugin

register_hydra_plugin(HabitatBaselinesConfigPlugin)
register_hydra_plugin(HabitatConfigPlugin)

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from habitat.config.default import patch_config


def repo_root() -> str:
    """habitat-gs repo root (this file lives in ``<root>/scripts_gs``).

    Set ``HABITAT_GS_ROOT`` to override, e.g. when the scripts are run from a
    relocated copy that is not next to ``data/``.
    """
    root = os.environ.get("HABITAT_GS_ROOT") or os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
    return os.path.abspath(os.path.expanduser(root))


ROOT = repo_root()
CONFIG_DIR = os.path.join(ROOT, "data", "scene_datasets", "gs_scenes", "configs")

OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/dyn_track_bc"
# Resolved against the repo root, matching the documented invocation
# ``python scripts_gs/train_dynamic_nav_dagger.py output/dyn_track_dagger 90000``
# (README.md:1005), which is run from the repo root.
OUT = OUT if os.path.isabs(OUT) else os.path.join(ROOT, OUT)
TOTAL_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 60000
N_ENVS = 2
SEG_T = 64
LR = 2.5e-4
TEACHER_FORCING_STEPS = 15000   # after this many env steps, switch to student-driven collection


def oracle_action_idx(obs) -> int:
    d, bearing, _v, _r = [float(x) for x in np.asarray(obs["track_target_state"])]
    deg = float(np.degrees(bearing))
    if d <= 0.0:
        return 0  # move_forward
    if abs(deg) > 20:
        return 1 if deg > 0 else 2
    if d > 2.1:
        return 0
    if abs(deg) > 8:
        return 1 if deg > 0 else 2
    if d > 1.85:
        return 0
    return 1 if deg > 0 else 2


def main() -> None:
    if not os.path.isdir(CONFIG_DIR):
        raise SystemExit(
            f"GS config dir not found: {CONFIG_DIR}\n"
            "Run this from a habitat-gs checkout with the GS scene dataset in place, "
            "or point HABITAT_GS_ROOT at one."
        )
    # The task configs reference `data/scene_datasets/...` relatively, and habitat
    # resolves those against the process cwd when the envs are constructed below.
    # Done here rather than at import time so that importing this module (e.g. to
    # reuse oracle_action_idx) does not silently move the caller's cwd.
    os.chdir(ROOT)

    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="ddppo_dynamic_track_gs_train")
    cfg = patch_config(cfg)
    OmegaConf.set_readonly(cfg, None)
    cfg.habitat_baselines.num_environments = N_ENVS

    from habitat_baselines.common.habitat_env_factory import HabitatVectorEnvFactory

    envs = HabitatVectorEnvFactory().construct_envs(
        cfg, workers_ignore_signals=False, enforce_scenes_greater_eq_environments=False,
        is_first_rank=True,
    )

    from habitat_baselines.rl.ddppo.policy import PointNavResNetPolicy

    policy = PointNavResNetPolicy(
        observation_space=envs.observation_spaces[0], action_space=envs.action_spaces[0],
        hidden_size=512, rnn_type="LSTM", num_recurrent_layers=2, backbone="resnet18",
        normalize_visual_inputs=True,
    ).cuda()
    policy.train()
    opt = torch.optim.Adam(policy.parameters(), lr=LR)
    num_rec = policy.net.num_recurrent_layers

    def batch_obs(obs_list):
        return {
            k: torch.as_tensor(np.stack([np.asarray(o[k]) for o in obs_list])).cuda()
            for k in obs_list[0].keys()
        }

    obs = envs.reset()
    hidden = torch.zeros(N_ENVS, num_rec, 512, device="cuda")
    prev_act = torch.zeros(N_ENVS, 1, dtype=torch.long, device="cuda")
    not_done = torch.zeros(N_ENVS, 1, dtype=torch.bool, device="cuda")

    steps_done = 0
    seg_steps = 0
    seg_loss = 0.0
    seg_correct = 0
    seg_count = 0
    seg_id = 0
    loss_ema = None
    acc_ema = None
    os.makedirs(f"{OUT}/checkpoints", exist_ok=True)

    while steps_done < TOTAL_STEPS:
        b = batch_obs(obs)
        labels = torch.tensor(
            [oracle_action_idx(o) for o in obs], dtype=torch.long, device="cuda"
        ).unsqueeze(1)
        features, hidden, _aux = policy.net(b, hidden, prev_act, not_done)
        dist = policy.action_distribution(features)
        # class-balanced CE (turn actions dominate the oracle's action marginal);
        # unambiguous shapes: (N,3) log-softmax gathered at the (N,1) labels.
        logp_all = torch.log_softmax(dist.logits, dim=-1)
        lp = logp_all.gather(1, labels).view(-1)
        w = torch.where(labels.view(-1) == 0, 1.6, 1.0)  # upweight move_forward
        seg_loss = seg_loss + (-(w * lp).sum() / w.sum())
        with torch.no_grad():
            pred = dist.probs.argmax(dim=-1, keepdim=True)
            if pred.dim() > labels.dim():
                pred = pred.squeeze(-1)
            seg_correct += int((pred.view(-1) == labels.view(-1)).sum().item())
            seg_count += N_ENVS

        # DAgger: teacher-forced warmup, then student-driven collection (oracle labels)
        if steps_done < TEACHER_FORCING_STEPS:
            exec_actions = labels.view(-1).tolist()
        else:
            with torch.no_grad():
                exec_actions = dist.sample().view(-1).tolist()
        outputs = envs.step([int(a) for a in exec_actions])
        obs, _r, dones, _i = [list(x) for x in zip(*outputs)]
        d = torch.tensor(dones, dtype=torch.bool, device="cuda").unsqueeze(1)
        not_done = ~d
        prev_act = torch.tensor(exec_actions, dtype=torch.long, device="cuda").unsqueeze(1)
        prev_act[d] = 0
        hidden = hidden * (~d).view(N_ENVS, 1, 1)
        steps_done += N_ENVS
        seg_steps += 1

        if seg_steps >= SEG_T:
            opt.zero_grad()
            (seg_loss / seg_steps).backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            opt.step()
            lv = float(seg_loss.item()) / seg_steps
            av = seg_correct / max(seg_count, 1)
            loss_ema = lv if loss_ema is None else 0.9 * loss_ema + 0.1 * lv
            acc_ema = av if acc_ema is None else 0.9 * acc_ema + 0.1 * av
            hidden = hidden.detach()
            seg_loss = 0.0
            seg_steps = 0
            seg_correct = 0
            seg_count = 0
            seg_id += 1
            if seg_id % 20 == 0:
                print(f"steps {steps_done:>7}/{TOTAL_STEPS}  bc_loss {loss_ema:.4f}  "
                      f"oracle-match {acc_ema:.3f}", flush=True)
            if seg_id % 100 == 0:
                sd = {f"actor_critic.{k}": v for k, v in policy.state_dict().items()}
                torch.save({"state_dict": sd, "config": None},
                           f"{OUT}/checkpoints/latest.pth")

    sd = {f"actor_critic.{k}": v for k, v in policy.state_dict().items()}
    torch.save({"state_dict": sd, "config": None}, f"{OUT}/checkpoints/latest.pth")
    envs.close()
    print("BC DONE", flush=True)


if __name__ == "__main__":
    main()
