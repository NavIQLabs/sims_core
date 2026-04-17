from __future__ import annotations

from dataclasses import dataclass, field

import sapien
import yaml


@dataclass
class RobotConfig:
    urdf:     str
    fix_root: bool        = True
    pose:     list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


@dataclass
class ObjectConfig:
    name:        str
    type:        str                  # "urdf" | "box" | "sphere" | "capsule"
    static:      bool                 = True
    pose:        list[float]          = field(default_factory=lambda: [0.0, 0.0, 0.0])
    path:        str | None           = None    # urdf type
    size:        list[float] | None   = None    # box: half-extents [x, y, z]
    radius:      float | None         = None    # sphere / capsule
    half_length: float | None         = None    # capsule
    color:       list[float]          = field(default_factory=lambda: [0.7, 0.7, 0.7, 1.0])


@dataclass
class LightSourceConfig:
    type:      str                    # "directional" | "point" | "spot"
    color:     list[float]            = field(default_factory=lambda: [1.0, 1.0, 1.0])
    shadow:    bool                   = False
    direction: list[float] | None     = None    # directional / spot
    position:  list[float] | None     = None    # point / spot
    inner_fov: float | None           = None    # spot
    outer_fov: float | None           = None    # spot


@dataclass
class LightingConfig:
    ambient: list[float]             = field(default_factory=lambda: [0.3, 0.3, 0.3])
    sources: list[LightSourceConfig] = field(default_factory=list)


# Background types:
#   "void"            - empty, no ground (default)
#   "ground"          - checkered infinite ground plane
#   "kitchen_counter" - built-in ManiSkill kitchen counter scene
#   "urdf"            - load an arbitrary URDF as background
@dataclass
class BackgroundConfig:
    type: str        = "void"
    path: str | None = None   # required when type="urdf"


# Per-object domain randomization: each reset samples a new pose
# within the given xyz ranges. Missing axes keep the default pose value.
@dataclass
class ObjectRandomization:
    name: str                         # must match an ObjectConfig name
    x:    list[float] | None = None   # [min, max]
    y:    list[float] | None = None
    z:    list[float] | None = None
    yaw:  list[float] | None = None   # [min, max] radians, rotation around Z


# Per-light domain randomization.
# intensity: scale the RGB color by a uniform sample in [min, max]
@dataclass
class LightRandomization:
    index:     int              # index into lighting.sources
    intensity: list[float] = field(default_factory=lambda: [1.0, 1.0])


@dataclass
class DomainRandomization:
    objects: list[ObjectRandomization] = field(default_factory=list)
    lights:  list[LightRandomization]  = field(default_factory=list)


@dataclass
class ViewConfig:
    eye:    list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    target: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.3])
    width:  int         = 1000
    height: int         = 1000


@dataclass
class RenderConfig:
    gui:  bool       = True
    view: ViewConfig = field(default_factory=ViewConfig)


@dataclass
class SceneDescription:
    objects: list[ObjectConfig] = field(default_factory=list)


@dataclass
class SimConfig:
    robot:         RobotConfig
    scene:         SceneDescription    = field(default_factory=SceneDescription)
    background:    BackgroundConfig    = field(default_factory=BackgroundConfig)
    lighting:      LightingConfig      = field(default_factory=LightingConfig)
    randomization: DomainRandomization = field(default_factory=DomainRandomization)
    render:        RenderConfig        = field(default_factory=RenderConfig)
    num_envs:      int                 = 1
    sim_backend:   str                 = "auto"  # auto | cpu | gpu | cuda:0 etc.


def parse_pose(p: list[float]) -> sapien.Pose:
    assert len(p) in (3, 7), f"pose must be [x,y,z] or [x,y,z,qw,qx,qy,qz], got {p}"
    if len(p) == 3:
        return sapien.Pose(p=p)
    return sapien.Pose(p=p[:3], q=p[3:])


def load(path: str) -> SimConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)

    assert "robot" in raw, "config missing 'robot' section"

    robot = RobotConfig(
        urdf     = raw["robot"]["urdf"],
        fix_root = raw["robot"].get("fix_root", True),
        pose     = raw["robot"].get("pose",     [0.0, 0.0, 0.0]),
    )

    scene_raw = raw.get("scene", {})
    scene = SceneDescription(objects=[
        ObjectConfig(
            name        = o["name"],
            type        = o["type"],
            static      = o.get("static", True),
            pose        = o.get("pose",   [0.0, 0.0, 0.0]),
            path        = o.get("path",   None),
            size        = o.get("size",   None),
            radius      = o.get("radius", None),
            half_length = o.get("half_length", None),
            color       = o.get("color",  [0.7, 0.7, 0.7, 1.0]),
        )
        for o in scene_raw.get("objects", [])
    ])

    bg_raw = raw.get("background", {})
    background = BackgroundConfig(
        type = bg_raw.get("type", "void"),
        path = bg_raw.get("path", None),
    )

    lighting_raw = raw.get("lighting", {})
    lighting = LightingConfig(
        ambient = lighting_raw.get("ambient", [0.3, 0.3, 0.3]),
        sources = [
            LightSourceConfig(
                type      = s["type"],
                color     = s.get("color",     [1.0, 1.0, 1.0]),
                shadow    = s.get("shadow",    False),
                direction = s.get("direction", None),
                position  = s.get("position",  None),
                inner_fov = s.get("inner_fov", None),
                outer_fov = s.get("outer_fov", None),
            )
            for s in lighting_raw.get("sources", [])
        ],
    )

    dr_raw = raw.get("randomization", {})
    randomization = DomainRandomization(
        objects=[
            ObjectRandomization(
                name = o["name"],
                x    = o.get("x",   None),
                y    = o.get("y",   None),
                z    = o.get("z",   None),
                yaw  = o.get("yaw", None),
            )
            for o in dr_raw.get("objects", [])
        ],
        lights=[
            LightRandomization(
                index     = l["index"],
                intensity = l.get("intensity", [1.0, 1.0]),
            )
            for l in dr_raw.get("lights", [])
        ],
    )

    render_raw = raw.get("render", {})
    view_raw   = render_raw.get("camera", {})
    render = RenderConfig(
        gui  = render_raw.get("gui", True),
        view = ViewConfig(
            eye    = view_raw.get("eye",    [1.0, 1.0, 1.0]),
            target = view_raw.get("target", [0.0, 0.0, 0.3]),
            width  = view_raw.get("width",  1000),
            height = view_raw.get("height", 1000),
        ),
    )

    return SimConfig(
        robot         = robot,
        scene         = scene,
        background    = background,
        lighting      = lighting,
        randomization = randomization,
        render        = render,
        num_envs      = int(raw.get("num_envs",    1)),
        sim_backend   = str(raw.get("sim_backend", raw.get("backend", "auto"))),
    )
