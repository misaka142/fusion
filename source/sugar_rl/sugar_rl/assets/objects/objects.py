from isaaclab.assets.rigid_object.rigid_object_cfg import RigidObjectCfg
from isaaclab.sim.spawners.from_files import UsdFileCfg
import isaaclab.sim as sim_utils


BIGBOX_CFG = RigidObjectCfg(
    spawn=UsdFileCfg(
        usd_path="descriptions/objects/big_box/obj_aligned.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False
        ),
        scale=(1.0, 1.0, 1.0),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.75),

    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0)
    )
)


CHAIR_CFG = RigidObjectCfg(
    spawn=UsdFileCfg(
        usd_path="descriptions/objects/chair/obj_aligned.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False
        ),
        scale=(1.0, 1.0, 1.0),
        mass_props=sim_utils.MassPropertiesCfg(mass=3.0),

    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0)
    )
)

BOTTLE_CFG = RigidObjectCfg(
    spawn=UsdFileCfg(
        usd_path="descriptions/objects/bottle/obj_aligned.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            angular_damping=0.2
        ),
        scale=(1.0, 1.0, 1.0),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.75),

    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0)
    )
)

SMALLBOX_CFG = RigidObjectCfg(
    spawn=UsdFileCfg(
        usd_path="descriptions/objects/small_box/obj_aligned.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False
        ),
        scale=(1.0, 1.0, 1.0),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.5),

    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0)
    )
)


def apply_dataset_box_spec(env_cfg, *motion_folders: str | None) -> bool:
    """Override `env_cfg.scene.obj` with the dataset's aligned convex-hull USD asset when
    `box_spec.json` is present (e.g. `data/OmniCarryBox` from `OmniRetarget_Dataset`).

    Fail-fast: refuses silent fallback to `CuboidCfg` if `box_spec.json` is detected without
    a companion `.usda`/`.usd` asset, because a cuboid approximation causes >42% hand/body penetration.
    Preserves 100% original `SMALLBOX_CFG` USD behavior when running standard SUGAR datasets.
    """
    import json
    from pathlib import Path

    for folder in motion_folders:
        if not folder:
            continue
        p = Path(folder)
        for cand in (p / "box_spec.json", p.parent / "box_spec.json"):
            if not cand.is_file():
                continue
            meta = json.loads(cand.read_text(encoding="utf-8"))
            mass = float(meta.get("mass_kg", 0.5))
            usd_rel = str(meta.get("usd_file", "obj_aligned.usda"))
            usd_candidates = [
                cand.parent / usd_rel,
                cand.parent / "obj_aligned.usda",
                cand.parent / "obj_aligned.usd",
            ]
            usd_path = next((u for u in usd_candidates if u.is_file()), None)
            if usd_path is None:
                raise FileNotFoundError(
                    f"[Dataset Box Override] Found `{cand}`, but no convex-hull USD asset exists "
                    f"(checked {[str(x) for x in usd_candidates]}). "
                    f"CuboidCfg fallback is disabled to prevent severe penetration; please re-run "
                    f"`scripts/sugar_rl/convert_omniretarget_to_sugar.py`."
                )

            prim_path = getattr(env_cfg.scene.obj, "prim_path", "{ENV_REGEX_NS}/Obj")
            env_cfg.scene.obj = RigidObjectCfg(
                prim_path=prim_path,
                spawn=UsdFileCfg(
                    usd_path=str(usd_path.resolve()),
                    scale=(1.0, 1.0, 1.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        disable_gravity=False,
                        max_depenetration_velocity=1.0,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            )
            print(
                f"[Dataset Box Override] Loaded `{cand}` ({meta.get('dataset', 'custom')}): "
                f"UsdFileCfg convexHull usd_path=`{usd_path.resolve()}`, mass={mass} kg, "
                f"max_depenetration_velocity=1.0"
            )
            return True
    return False
