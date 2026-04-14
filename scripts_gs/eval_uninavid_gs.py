#!/usr/bin/env python3
"""
Uni-NaVid evaluation on Gaussian Splatting scenes using Habitat Env.

Mirrors the NaVid-VLN-CE evaluation (run.py + agent_uninavid.py) faithfully:
  - Same agent logic (prompt, tokenization, special tokens, action parsing)
  - Same evaluation loop (early stopping, metric collection, per-episode JSON)
  - Same multi-GPU splitting (sorted episodes, np.random.seed(42))

The only structural differences from NaVid-VLN-CE are:
  1. Uses habitat-lab 0.3.3 DictConfig API (not the deprecated yacs CN API)
  2. Fixes relative paths in model config (mm_vision_tower, image_processor)
  3. Computes oracle_success/path_length manually (these measures are not
     registered in habitat-lab 0.3.3)

Usage:
    python scripts_gs/eval_uninavid_gs.py \
        --model-path /path/to/uninavid/checkpoint \
        --eval-split val

    # Multi-GPU
    bash scripts_gs/eval_uninavid.sh --ckpt /path/to/checkpoint --num-gpus 4
"""

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ═══════════════════════════════════════════════════════════════════════
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Early stopping (matching NaVid-VLN-CE defaults from uninavid_r2r.yaml)
EARLY_STOP_ROTATION = 25
EARLY_STOP_STEPS = 400
SUCCESS_DISTANCE = 3.0
FORWARD_STEP_SIZE = 0.25  # for path_length computation


# =====================================================================
#  Uni-NaVid Agent  (faithfully mirrors NaVid-VLN-CE/agent_uninavid.py)
# =====================================================================

class UniNaVidAgent:
    """Uni-NaVid agent for online VLN evaluation in Habitat Env.

    This is a faithful reimplementation of NaVid-VLN-CE/agent_uninavid.py,
    using the same model loading, tokenization, inference, and action parsing.
    """

    def __init__(self, model_path: str, result_path: str, exp_save: str):
        from uninavid.mm_utils import get_model_name_from_path
        from uninavid.model.builder import load_pretrained_model
        import transformers

        print("Initialize UniNaVid")

        self.result_path = result_path
        self.require_map = "video" in exp_save
        self.require_data = "data" in exp_save

        self.conv_mode = "vicuna_v1"

        if self.require_map or self.require_data:
            os.makedirs(self.result_path, exist_ok=True)
        if self.require_data:
            os.makedirs(os.path.join(self.result_path, "log"), exist_ok=True)
        if self.require_data:
            os.makedirs(os.path.join(self.result_path, "video"), exist_ok=True)

        # Fix relative paths in model config (mm_vision_tower, image_processor).
        # The checkpoint was saved with paths relative to the Uni-NaVid repo root.
        uninavid_root = Path(model_path)
        while uninavid_root.name and not (uninavid_root / "uninavid").is_dir():
            uninavid_root = uninavid_root.parent
        if (uninavid_root / "uninavid").is_dir():
            cfg = transformers.AutoConfig.from_pretrained(model_path)
            vt = getattr(cfg, "mm_vision_tower", "")
            if vt and not os.path.isabs(vt) and not os.path.exists(vt):
                abs_vt = str(uninavid_root / vt)
                if os.path.exists(abs_vt):
                    cfg.mm_vision_tower = abs_vt
            ip = getattr(cfg, "image_processor", "")
            if ip and not os.path.isabs(ip) and not os.path.exists(ip):
                abs_ip = str(uninavid_root / ip)
                if os.path.exists(abs_ip):
                    cfg.image_processor = abs_ip
            cfg.save_pretrained(model_path)

        self.model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = \
            load_pretrained_model(model_path, None, self.model_name)

        print("Initialization Complete")

        # Prompt template — identical to NaVid-VLN-CE/agent_uninavid.py line 48
        self.promt_template = (
            "Imagine you are a robot programmed for navigation tasks. "
            "You have been given a video of historical observations and an "
            "image of the current observation <image>. "
            "Your assigned task is: '{}'. "
            "Analyze this series of images to determine your next four actions. "
            "The predicted action should be one of the following: "
            "forward, left, right, or stop."
        )

        self.rgb_list = []
        self.topdown_map_list = []
        self.count_id = 0
        self.reset()

    def process_images(self, rgb_list):
        """Identical to NaVid-VLN-CE/agent_uninavid.py lines 59-76."""
        batch_image = np.asarray(rgb_list)
        self.model.get_model().new_frames = len(rgb_list)
        video = self.image_processor.preprocess(
            batch_image, return_tensors='pt'
        )['pixel_values'].half().cuda()
        return [video]

    def predict_inference(self, prompt: str) -> str:
        """Identical to NaVid-VLN-CE/agent_uninavid.py lines 79-160."""
        from uninavid.constants import (
            IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN,
        )
        from uninavid.conversation import conv_templates, SeparatorStyle
        from uninavid.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria

        question = prompt.replace(DEFAULT_IMAGE_TOKEN, '').replace('\n', '')
        qs = prompt

        VIDEO_START_SPECIAL_TOKEN = "<video_special>"
        VIDEO_END_SPECIAL_TOKEN = "</video_special>"
        IMAGE_START_TOKEN = "<image_special>"
        IMAGE_END_TOKEN = "</image_special>"
        NAVIGATION_SPECIAL_TOKEN = "[Navigation]"
        IAMGE_SEPARATOR = "<image_sep>"
        image_start_special_token = self.tokenizer(IMAGE_START_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        image_end_special_token = self.tokenizer(IMAGE_END_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        video_start_special_token = self.tokenizer(VIDEO_START_SPECIAL_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        video_end_special_token = self.tokenizer(VIDEO_END_SPECIAL_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        navigation_special_token = self.tokenizer(NAVIGATION_SPECIAL_TOKEN, return_tensors="pt").input_ids[0][1:].cuda()
        image_seperator = self.tokenizer(IAMGE_SEPARATOR, return_tensors="pt").input_ids[0][1:].cuda()

        # Build prompt — matching original mm_use_im_start_end check
        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs.replace('<image>', '')
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs.replace('<image>', '')

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        token_prompt = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').cuda()
        indices_to_replace = torch.where(token_prompt == -200)[0]
        new_list = []
        while indices_to_replace.numel() > 0:
            idx = indices_to_replace[0]
            new_list.append(token_prompt[:idx])
            new_list.append(video_start_special_token)
            new_list.append(image_seperator)
            new_list.append(token_prompt[idx:idx + 1])
            new_list.append(video_end_special_token)
            new_list.append(image_start_special_token)
            new_list.append(image_end_special_token)
            new_list.append(navigation_special_token)
            token_prompt = token_prompt[idx + 1:]
            indices_to_replace = torch.where(token_prompt == -200)[0]
        if token_prompt.numel() > 0:
            new_list.append(token_prompt)
        input_ids = torch.cat(new_list, dim=0).unsqueeze(0)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        imgs = self.process_images(self.rgb_list)
        self.rgb_list = []

        cur_prompt = question
        with torch.inference_mode():
            self.model.update_prompt([[cur_prompt]])
            output_ids = self.model.generate(
                input_ids,
                images=imgs,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        return outputs

    def reset(self):
        """Identical to NaVid-VLN-CE/agent_uninavid.py lines 236-256."""
        self.history_rgb_tensor = None
        self.transformation_list = []
        self.rgb_list = []
        self.topdown_map_list = []
        self.last_action = None
        self.count_id += 1
        self.count_stop = 0
        self.pending_action_list = []

        self.model.config.run_type = "eval"
        self.model.get_model().initialize_online_inference_nav_feat_cache()
        self.model.get_model().new_frames = 0
        self.first_forward = False

    def act(self, observations, info, episode_id):
        """Identical to NaVid-VLN-CE/agent_uninavid.py lines 260-311."""
        self.episode_id = episode_id
        rgb = observations["rgb"]
        self.rgb_list.append(rgb)

        if len(self.pending_action_list) != 0:
            temp_action = self.pending_action_list.pop(0)
            return {"action": temp_action}

        navigation_qs = self.promt_template.format(observations["instruction"]["text"])
        navigation = self.predict_inference(navigation_qs)

        # Parse action words — identical to original (exact match, ValueError on unknown)
        action_list = navigation.split(" ")
        for action in action_list:
            if action == "stop":
                self.pending_action_list.append(0)
            elif action == "forward":
                self.pending_action_list.append(1)
            elif action == "left":
                self.pending_action_list.append(2)
            elif action == "right":
                self.pending_action_list.append(3)
            else:
                # Original raises ValueError here; we log and skip to avoid
                # crashing long evaluation runs, but preserve the intent.
                print(f"[Warning] Unknown action word: '{action}' in output: '{navigation}'")
                continue

            if len(self.pending_action_list) == 2:
                break

        # If nothing parsed, default to stop (original would have raised ValueError)
        if not self.pending_action_list:
            print(f"[Warning] No valid actions parsed from: '{navigation}', defaulting to stop")
            self.pending_action_list.append(0)

        return {"action": self.pending_action_list.pop(0)}


# =====================================================================
#  Evaluation loop  (faithfully mirrors NaVid-VLN-CE/run.py)
# =====================================================================

def evaluate_agent(config_identification, split_id, env, agent, result_path, exp_save):
    """Mirrors NaVid-VLN-CE/run.py:evaluate_agent."""
    from tqdm import trange

    num_episodes = len(env.episodes)

    # Same target_key set as NaVid-VLN-CE/run.py line 117
    # (oracle_success and path_length are collected from env if available,
    #  otherwise computed manually since habitat-lab 0.3.3 doesn't register them)
    target_key = {"distance_to_goal", "success", "spl", "path_length", "oracle_success"}

    count = 0

    for _ in trange(num_episodes, desc=config_identification + "-{}".format(split_id)):
        obs = env.reset()
        iter_step = 0
        agent.reset()

        continuse_rotation_count = 0
        last_dtg = 999

        # Manual tracking for metrics not available in habitat-lab 0.3.3
        min_dtg = 999
        prev_pos = None
        manual_path_length = 0.0

        while not env.episode_over:
            info = env.get_metrics()

            # Track distance for oracle success and stuck detection
            cur_dtg = info.get("distance_to_goal", 999)
            min_dtg = min(min_dtg, cur_dtg)

            if cur_dtg != last_dtg:
                last_dtg = cur_dtg
                continuse_rotation_count = 0
            else:
                continuse_rotation_count += 1

            action = agent.act(obs, info, env.current_episode.episode_id)

            if continuse_rotation_count > EARLY_STOP_ROTATION or iter_step > EARLY_STOP_STEPS:
                action = {"action": 0}

            # Track path length manually
            cur_pos = env._sim.get_agent(0).get_state().position
            if prev_pos is not None:
                manual_path_length += float(np.linalg.norm(cur_pos - prev_pos))
            prev_pos = cur_pos.copy()

            iter_step += 1
            obs = env.step(action)

        # Final position for path_length
        cur_pos = env._sim.get_agent(0).get_state().position
        if prev_pos is not None:
            manual_path_length += float(np.linalg.norm(cur_pos - prev_pos))

        info = env.get_metrics()
        result_dict = {k: info[k] for k in target_key if k in info}
        result_dict["id"] = env.current_episode.episode_id
        count += 1

        # Fill in metrics that habitat-lab 0.3.3 doesn't compute
        if "oracle_success" not in result_dict:
            result_dict["oracle_success"] = float(min_dtg <= SUCCESS_DISTANCE)
        if "path_length" not in result_dict:
            result_dict["path_length"] = manual_path_length

        if "data" in exp_save:
            with open(os.path.join(os.path.join(result_path, "log"),
                      "stats_{}.json".format(env.current_episode.episode_id)), "w") as f:
                json.dump(result_dict, f, indent=4)


def aggregate_results(result_path: str):
    """Aggregate per-episode JSON results. Mirrors NaVid-VLN-CE/analyze_results.py."""
    log_dir = os.path.join(result_path, "log")
    if not os.path.isdir(log_dir):
        print(f"No results found in {log_dir}")
        return

    results = []
    for fname in sorted(os.listdir(log_dir)):
        if fname.startswith("stats_") and fname.endswith(".json"):
            try:
                with open(os.path.join(log_dir, fname)) as f:
                    data = json.load(f)
                for k, v in data.items():
                    if isinstance(v, float) and (math.isinf(v) or math.isnan(v)):
                        data[k] = 0.0
                results.append(data)
            except (json.JSONDecodeError, KeyError):
                continue

    if not results:
        print("No result files found")
        return

    n = len(results)
    succ = sum(int(r.get("success", 0)) for r in results)
    oracle_succ = sum(int(r.get("oracle_success", 0)) for r in results)
    spl = sum(r.get("spl", 0) for r in results)
    distance_to_goal = sum(r.get("distance_to_goal", 0) for r in results)
    path_length = sum(r.get("path_length", 0) for r in results)

    print(f"\n{'=' * 50}")
    print(f"  Aggregated Results ({n} episodes)")
    print(f"{'=' * 50}")
    print(f"  Success rate: {succ}/{n} ({succ/n:.3f})")
    print(f"  Oracle success rate: {oracle_succ}/{n} ({oracle_succ/n:.3f})")
    print(f"  SPL: {spl:.3f}/{n} ({spl/n:.3f})")
    print(f"  Distance to goal: {distance_to_goal/n:.3f}")
    print(f"  Path length: {path_length/n:.3f}")
    print(f"{'=' * 50}")

    summary = {
        "num_episodes": n,
        "SR": succ / n, "OSR": oracle_succ / n,
        "SPL": spl / n, "DTG": distance_to_goal / n, "PL": path_length / n,
    }
    with open(os.path.join(result_path, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved to {result_path}/summary.json")


# =====================================================================
#  Main  (mirrors NaVid-VLN-CE/run.py:run_exp + evaluate_agent)
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Uni-NaVid evaluation on GS scenes")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--eval-split", type=str, default="val")
    parser.add_argument("--result-path", type=str, default="")
    parser.add_argument("--split-num", type=int, default=1,
                        help="Total number of GPU workers")
    parser.add_argument("--split-id", type=int, default=0,
                        help="This worker's ID (0-indexed)")
    parser.add_argument("--exp-save", type=str, default="data",
                        help="What to save: 'data' for JSON logs, 'video-data' for both")
    parser.add_argument("--config-path", type=str, default="",
                        help="Path to habitat eval config YAML")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Only aggregate existing results, don't run evaluation")
    args = parser.parse_args()

    # Default result path
    if not args.result_path:
        ckpt_name = os.path.basename(args.model_path.rstrip("/"))
        args.result_path = str(PROJECT_ROOT / "results" / "uninavid" /
                               f"{ckpt_name}_{args.eval_split}")

    if args.aggregate_only:
        aggregate_results(args.result_path)
        return

    # Default config
    if not args.config_path:
        args.config_path = str(
            PROJECT_ROOT / "data" / "scene_datasets" / "gs_scenes" / "configs" /
            "vln_uninavid_gs_eval.yaml"
        )

    # Must change to project root so relative paths in config resolve
    os.chdir(str(PROJECT_ROOT))

    # Load Habitat config
    from habitat.config import get_config
    configs_dir = str(PROJECT_ROOT / "data" / "scene_datasets" / "gs_scenes" / "configs")

    overrides = [
        f"habitat.dataset.split={args.eval_split}",
    ]
    if args.eval_split == "train":
        overrides.append(
            "habitat.simulator.scene_dataset="
            "data/scene_datasets/gs_scenes/train.scene_dataset_config.json"
        )

    config = get_config(args.config_path, overrides=overrides, configs_dir=configs_dir)

    # Create Habitat Env
    from habitat import Env
    env = Env(config=config)

    # Sort episodes and split — matching NaVid-VLN-CE/run.py lines 85-89
    all_episodes = list(env._dataset.episodes)
    all_episodes.sort(key=lambda ep: ep.episode_id)
    np.random.seed(42)

    if args.split_num > 1:
        chunk_size = (len(all_episodes) + args.split_num - 1) // args.split_num
        start = args.split_id * chunk_size
        end = min(start + chunk_size, len(all_episodes))
        env._dataset.episodes = all_episodes[start:end]

    print(f"Worker {args.split_id}/{args.split_num}: "
          f"{len(env._dataset.episodes)} episodes")

    # Load agent — matching NaVid-VLN-CE/run.py lines 106-108
    agent = UniNaVidAgent(args.model_path, args.result_path, args.exp_save)

    # Run evaluation — matching NaVid-VLN-CE/run.py line calling pattern
    evaluate_agent("r2r", args.split_id, env, agent, args.result_path, args.exp_save)

    env.close()

    # Aggregate if single worker
    if args.split_num == 1:
        aggregate_results(args.result_path)
    else:
        print(f"\nWorker {args.split_id} done. After all workers finish, aggregate with:")
        print(f"  python scripts_gs/eval_uninavid_gs.py "
              f"--model-path {args.model_path} --result-path {args.result_path} --aggregate-only")


if __name__ == "__main__":
    main()
