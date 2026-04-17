from __future__ import annotations

import numpy as np
import sapien
import gymnasium as gym

from mani_skill.agents.base_agent   import BaseAgent, Keyframe
from mani_skill.agents.registration import register_agent
from mani_skill.agents.controllers  import PDJointPosControllerConfig
from mani_skill.envs.sapien_env     import BaseEnv
from mani_skill.utils.registration  import register_env
from mani_skill.utils               import sapien_utils
from mani_skill.sensors.camera      import CameraConfig
from mani_skill.utils.structs.types import SimConfig as ManiSimConfig
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.scene_builder.kitchen_counter import KitchenCounterSceneBuilder

import torch
from mani_skill.utils.structs.pose import Pose as ManiPose

from .config import SimConfig, parse_pose

BUILTIN_SCENES = {
    "kitchen_counter": KitchenCounterSceneBuilder,
}

class SimEnv(BaseEnv):

    def __init__(self, *args, sim_cfg: SimConfig, robot_uid: str, **kwargs):
        self.sim_cfg   = sim_cfg
        self.dr_actors = {}   # name -> (actor, base_pose) for pose reset each episode
        super().__init__(*args, robot_uids=robot_uid, **kwargs)

    @property
    def _default_sim_config(self):
        from mani_skill.utils.structs.types import GPUMemoryConfig
        import math
        # scale buffers with num_envs — each env adds contacts proportionally
        n         = self.num_envs
        exp       = max(22, math.ceil(math.log2(n * 2**19)))
        p_exp     = max(20, math.ceil(math.log2(n * 2**18)))
        fl_exp    = max(26, math.ceil(math.log2(n * 2**24)))
        return ManiSimConfig(
            gpu_memory_config = GPUMemoryConfig(
                max_rigid_contact_count      = 2**exp,
                max_rigid_patch_count        = 2**p_exp,
                found_lost_pairs_capacity    = 2**fl_exp,
            )
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(
            eye    = self.sim_cfg.render.view.eye,
            target = self.sim_cfg.render.view.target,
        )
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        v    = self.sim_cfg.render.view
        pose = sapien_utils.look_at(eye=v.eye, target=v.target)
        return CameraConfig("render_camera", pose, v.width, v.height, 1.0, 0.01, 100)

    def _load_agent(self, options: dict, initial_agent_poses=None, build_separate=False):
        super()._load_agent(
            options,
            initial_agent_poses = parse_pose(self.sim_cfg.robot.pose),
            build_separate      = build_separate,
        )

    def _load_scene(self, options: dict) -> None:
        bg = self.sim_cfg.background

        if bg.type == "ground":
            import math
            grid      = math.ceil(math.sqrt(self.num_envs))
            span      = (grid - 1) * self._default_sim_config.spacing
            floor_w   = int(span + 20)   # 10m margin on each side
            build_ground(self.scene, floor_width=floor_w)

        elif bg.type == "urdf":
            assert bg.path, "background type='urdf' requires a 'path'"
            loader             = self.scene.create_urdf_loader()
            loader.fix_root_link = True
            loader.load(bg.path)

        elif bg.type in BUILTIN_SCENES:
            BUILTIN_SCENES[bg.type](self).build()

        for obj in self.sim_cfg.scene.objects:
            pose = parse_pose(obj.pose)

            if obj.type == "urdf":
                loader               = self.scene.create_urdf_loader()
                loader.fix_root_link = obj.static
                art                  = loader.load(obj.path)
                art.initial_pose     = pose
                self.dr_actors[obj.name] = (art, obj.pose)

            elif obj.type == "box":
                mat   = sapien.render.RenderMaterial(base_color=obj.color)
                b     = self.scene.create_actor_builder()
                b.set_initial_pose(pose)
                b.add_box_collision(half_size=obj.size)
                b.add_box_visual(half_size=obj.size, material=mat)
                actor = b.build_static(obj.name) if obj.static else b.build(obj.name)
                self.dr_actors[obj.name] = (actor, obj.pose)

            elif obj.type == "sphere":
                mat   = sapien.render.RenderMaterial(base_color=obj.color)
                b     = self.scene.create_actor_builder()
                b.set_initial_pose(pose)
                b.add_sphere_collision(radius=obj.radius)
                b.add_sphere_visual(radius=obj.radius, material=mat)
                actor = b.build_static(obj.name) if obj.static else b.build(obj.name)
                self.dr_actors[obj.name] = (actor, obj.pose)

            elif obj.type == "capsule":
                mat   = sapien.render.RenderMaterial(base_color=obj.color)
                b     = self.scene.create_actor_builder()
                b.set_initial_pose(pose)
                b.add_capsule_collision(radius=obj.radius, half_length=obj.half_length)
                b.add_capsule_visual(radius=obj.radius, half_length=obj.half_length,
                                     material=mat)
                actor = b.build_static(obj.name) if obj.static else b.build(obj.name)
                self.dr_actors[obj.name] = (actor, obj.pose)

            else:
                raise ValueError(f"Unknown object type '{obj.type}'")

    def _load_lighting(self, options: dict) -> None:
        lighting = self.sim_cfg.lighting
        self.scene.set_ambient_light(lighting.ambient)
        for src in lighting.sources:
            if src.type == "directional":
                self.scene.add_directional_light(
                    src.direction, src.color, shadow=src.shadow,
                )
            elif src.type == "point":
                self.scene.add_point_light(
                    src.position, src.color, shadow=src.shadow,
                )
            elif src.type == "spot":
                self.scene.add_spot_light(
                    src.position, src.direction,
                    src.inner_fov, src.outer_fov,
                    src.color, shadow=src.shadow,
                )
            else:
                raise ValueError(f"Unknown light type '{src.type}'")

    def _initialize_episode(self, env_idx, options: dict) -> None:
        self.agent.robot.set_pose(parse_pose(self.sim_cfg.robot.pose))
        dr = self.sim_cfg.randomization

        n = self.num_envs
        for name, (actor, base_pose) in self.dr_actors.items():
            bx = float(base_pose[0])
            by = float(base_pose[1])
            bz = float(base_pose[2])
            bq = base_pose[3:] if len(base_pose) == 7 else [1.0, 0.0, 0.0, 0.0]

            obj_dr = next((o for o in dr.objects if o.name == name), None)
            if obj_dr is not None:
                xs   = np.random.uniform(*obj_dr.x,   size=n) if obj_dr.x   else np.full(n, bx)
                ys   = np.random.uniform(*obj_dr.y,   size=n) if obj_dr.y   else np.full(n, by)
                zs   = np.random.uniform(*obj_dr.z,   size=n) if obj_dr.z   else np.full(n, bz)
                yaws = np.random.uniform(*obj_dr.yaw, size=n) if obj_dr.yaw else None
            else:
                xs   = np.full(n, bx)
                ys   = np.full(n, by)
                zs   = np.full(n, bz)
                yaws = None

            ps = torch.tensor(np.stack([xs, ys, zs], axis=1), dtype=torch.float32)

            if yaws is not None:
                # yaw around Z: q = [cos(θ/2), 0, 0, sin(θ/2)] in [qw,qx,qy,qz]
                half = yaws / 2.0
                qs   = torch.tensor(
                    np.stack([np.cos(half), np.zeros(n), np.zeros(n), np.sin(half)], axis=1),
                    dtype=torch.float32,
                )
            else:
                qs = torch.tensor(bq, dtype=torch.float32).unsqueeze(0).expand(n, -1)

            actor.set_pose(ManiPose.create_from_pq(p=ps, q=qs))

        # Light DR: re-add lights with randomized intensity each episode.
        # We scale the base color by the sampled factor.
        for light_dr in dr.lights:
            src = self.sim_cfg.lighting.sources[light_dr.index]
            scale = float(np.random.uniform(*light_dr.intensity))
            color = [c * scale for c in src.color]
            if src.type == "directional":
                self.scene.add_directional_light(src.direction, color)
            elif src.type == "point":
                self.scene.add_point_light(src.position, color)
            elif src.type == "spot":
                self.scene.add_spot_light(
                    src.position, src.direction,
                    src.inner_fov, src.outer_fov, color,
                )

    def evaluate(self) -> dict:
        import torch
        n = self.num_envs
        return {
            "success": torch.zeros(n, dtype=torch.bool, device=self.device),
            "fail":    torch.zeros(n, dtype=torch.bool, device=self.device),
        }

    def _get_obs_extra(self, info: dict) -> dict:
        return {}

    def compute_dense_reward(self, obs, action, info):
        import torch
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info)


class ControllerManager:
    """
    Builds and runs a ManiSkill environment from a SimConfig.
    No control — robot holds rest pose. Ctrl-C exits the loop.
    """

    def __init__(self, cfg: SimConfig) -> None:
        self.cfg = cfg
        self.env = self.build_env()

    def build_env(self) -> gym.Env:
        robot_uid = "sim_robot"
        env_id    = "SimEnv-v0"

        robot_cfg = self.cfg.robot

        def controller_configs(self_agent):
            joints = [j.name for j in self_agent.robot.get_active_joints()]
            return dict(
                pd_joint_pos = PDJointPosControllerConfig(
                    joints,
                    lower            = None,
                    upper            = None,
                    stiffness        = 1e3,
                    damping          = 1e2,
                    force_limit      = 100.0,
                    normalize_action = False,
                )
            )

        robot_cls = type(robot_uid, (BaseAgent,), dict(
            uid                      = robot_uid,
            urdf_path                = robot_cfg.urdf,
            fix_root_link            = robot_cfg.fix_root,
            load_multiple_collisions = True,
            keyframes                = dict(
                rest=Keyframe(qpos=None, pose=parse_pose(robot_cfg.pose))
            ),
            _controller_configs = property(controller_configs),
        ))
        register_agent()(robot_cls)

        register_env(env_id, max_episode_steps=10_000)(SimEnv)

        render_mode = "human" if self.cfg.render.gui else "rgb_array"
        return gym.make(
            env_id,
            num_envs                 = self.cfg.num_envs,
            obs_mode                 = "state",
            control_mode             = "pd_joint_pos",
            render_mode              = render_mode,
            sim_backend              = self.cfg.sim_backend,
            parallel_in_single_scene = self.cfg.render.gui and self.cfg.num_envs > 1,
            sim_cfg                  = self.cfg,
            robot_uid                = robot_uid,
        )

    def run(self) -> None:
        import time
        obs, _       = self.env.reset()
        n            = self.env.unwrapped.num_envs
        action       = np.zeros((n, *self.env.single_action_space.shape))
        control_dt   = self.env.unwrapped.control_timestep
        try:
            while True:
                t0 = time.perf_counter()
                self.env.step(action)
                if self.cfg.render.gui:
                    self.env.render()
                if self.cfg.real_time:
                    elapsed = time.perf_counter() - t0
                    remaining = control_dt - elapsed
                    if remaining > 0:
                        time.sleep(remaining)
        except KeyboardInterrupt:
            pass

    def close(self) -> None:
        self.env.close()
