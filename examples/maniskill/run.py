#!/usr/bin/env python3
"""
Control loop runner for the sim factory.

Usage:
    python run.py --config workcell.yaml [--scene scene.yaml] [--steps 1000]
"""

import argparse
import sys
import os

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from maniskill.factory import build_entity


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config",  default="workcell.yaml")
    p.add_argument("--scene",   default=None)
    p.add_argument("--steps",   type=int, default=500)
    p.add_argument("--backend", default="auto")
    p.add_argument(
        "--render-mode",
        default  = "human",
        choices  = ["human", "rgb_array", "sensors", "none"],
    )
    p.add_argument(
        "--obs-mode",
        default  = "sensor_data",
        choices  = ["state", "rgb", "depth", "rgb+depth", "sensor_data"],
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    config_path = os.path.join(os.path.dirname(__file__), args.config)
    scene_path  = (
        os.path.join(os.path.dirname(__file__), args.scene)
        if args.scene else None
    )
    render_mode = None if args.render_mode == "none" else args.render_mode

    entity = build_entity(
        config_path       = config_path,
        scene_config_path = scene_path,
        render_mode       = render_mode,
        sim_backend       = args.backend,
        obs_mode          = args.obs_mode,
    )

    obs, info = entity.reset(seed=0)
    print(f"obs keys:    {list(obs.keys())}")
    print(f"action space: {entity._env.action_space}")

    # -- set initial zero targets on all joint-position controllers ----------
    num_envs = entity._env.unwrapped.num_envs
    ctrl_mgr = entity.controller_manager
    for name in ["left_arm", "right_arm", "left_gripper", "right_gripper"]:
        ctrl = ctrl_mgr[name]
        n    = len(ctrl.controlled_joints)
        ctrl.set_target(torch.zeros(num_envs, n))

    # -- control loop --------------------------------------------------------
    for step in range(args.steps):
        obs, reward, terminated, truncated, info = entity.step()

        if terminated.any() or truncated.any():
            obs, info = entity.reset()

        if render_mode == "human":
            entity._env.render()

    entity._env.close()


if __name__ == "__main__":
    main()
