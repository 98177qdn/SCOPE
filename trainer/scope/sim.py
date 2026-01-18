from model.scope import SCOPE
from envs.UAVEnv import UAVEnv
import numpy as np
import warnings
import time

from trainer.scope.dynamics_model import MultiAgentDynamicModel

import mujoco
import mujoco.viewer as mjviewer

warnings.filterwarnings('ignore')

done_flag = False
final_frame = 0

SECONDS_PER_FRAME = 2.0 / 30.0


def test():
    n_actions = 2
    future_steps = 3
    seq_len = 5

    # === env initialize ===
    env = UAVEnv(dim_actions=n_actions)
    n_hunters = env.num_hunters
    # get dim of observation
    first_agent_key = list(env.hunters_observation_space.keys())[0]
    obs_dim = env.hunters_observation_space[first_agent_key].shape[0]
    preds_buffer = []

    # observation concatenate
    actor_dims = [obs_dim + future_steps * obs_dim for _ in range(n_hunters)]
    critic_dims = sum(actor_dims)

    # === initialize agent ===
    agents = SCOPE(actor_dims, critic_dims, n_hunters, n_actions, env,
                   fc1=128, fc2=128, alpha=0.0001, beta=0.002,
                   scenario='UAV_Round_up', chkpt_dir='../../tmp/scope/')

    agents.load_checkpoint()
    print('---- Evaluating ----')

    # === initialize SPDM ===
    SPDM = MultiAgentDynamicModel(
        n_hunters, seq_len, obs_dim, future_steps,
        chkpt_dir='../../tmp/scope/', name='DLinear.pt'
    )
    SPDM.load_model()

    # === reset ===
    hunters_obs, _ = env.reset()
    rewards = np.zeros(env.num_hunters)
    obs_records = [hunters_obs]

    total_steps = 0
    max_frames = 400
    done_flag = False

    # load MuJoCo
    mj_model = mujoco.MjModel.from_xml_path('../../crazyfile/scene.xml')
    mj_data = mujoco.MjData(mj_model)

    # === MuJoCo config ===
    ENV_UAV_RADIUS = 0.03
    MUJOCO_UAV_RADIUS = 0.035
    POS_SCALE = MUJOCO_UAV_RADIUS / ENV_UAV_RADIUS  # ≈ 1.1666667

    hunter_joint_names = [f"hunter{i}_root" for i in range(n_hunters)]
    hunter_qposadr = []

    for name in hunter_joint_names:
        try:
            jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        except Exception as e:
            raise RuntimeError(
                f"please confirm scene.xml exist <joint name=\"{name}\" type=\"free\"/>."
            ) from e
        hunter_qposadr.append(mj_model.jnt_qposadr[jid])

    try:
        target_jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, "target_root")
        target_qposadr = mj_model.jnt_qposadr[target_jid]
    except Exception as e:
        raise RuntimeError(
            "please confirm scene.xml exist <joint name=\"target_root\" type=\"free\"/>."
        ) from e

    MAX_MJ_OBS = 10
    obs_joint_names = [f"obs{i}_root" for i in range(MAX_MJ_OBS)]
    obs_geom_names = [f"obs{i}_geom" for i in range(MAX_MJ_OBS)]

    obs_qposadr = []
    obs_geom_ids = []

    for jname, gname in zip(obs_joint_names, obs_geom_names):
        try:
            jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            gid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, gname)
        except Exception:
            continue
        obs_qposadr.append(mj_model.jnt_qposadr[jid])
        obs_geom_ids.append(gid)

    HUNTER_RGBA = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)  # 绿色
    TARGET_RGBA = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)  # 红色

    MAX_TRAIL_POINTS = 100       # trajectory points
    TRAIL_EVERY_N_FRAMES = 1

    TRAIL_WIDTH_ENV = 1      # linewidth

    hunter_trails = [list() for _ in range(n_hunters)]
    target_trail = []

    def _push_point(trail_list, p, maxlen):
        trail_list.append(p)
        if len(trail_list) > maxlen:
            del trail_list[0:len(trail_list) - maxlen]

    def _draw_polyline(viewer, points, rgba, width):
        if len(points) < 2:
            return
        scn = viewer.user_scn
        for p0, p1 in zip(points[:-1], points[1:]):
            if scn.ngeom >= scn.maxgeom:
                break

            g = scn.geoms[scn.ngeom]
            scn.ngeom += 1

            mujoco.mjv_initGeom(
                g,
                type=mujoco.mjtGeom.mjGEOM_LINE,
                size=np.array([width, 0.0, 0.0], dtype=np.float64),
                pos=np.zeros(3, dtype=np.float64),
                mat=np.eye(3, dtype=np.float64).ravel(),
                rgba=np.array(rgba, dtype=np.float32),
            )

            mujoco.mjv_connector(
                g,
                mujoco.mjtGeom.mjGEOM_LINE,
                float(width),
                np.asarray(p0, dtype=np.float64),
                np.asarray(p1, dtype=np.float64),
            )
            g.rgba[:] = rgba

    def draw_trails(viewer):
        scn = viewer.user_scn
        scn.ngeom = 0

        width_mj = TRAIL_WIDTH_ENV * POS_SCALE

        # hunters：green
        for i in range(n_hunters):
            _draw_polyline(viewer, hunter_trails[i], HUNTER_RGBA, width_mj)

        # target：red
        _draw_polyline(viewer, target_trail, TARGET_RGBA, width_mj)

    def sync_mujoco_from_env(record_trail=False, maxlen=150):
        # ---- hunters ----
        for i in range(n_hunters):
            pos = env.hunters[i].pos
            x_env = float(pos[0])
            y_env = float(pos[1])
            z_env = 0.1 if len(pos) < 3 else float(pos[2])

            x_mj = x_env * POS_SCALE
            y_mj = y_env * POS_SCALE
            z_mj = z_env * POS_SCALE

            adr = hunter_qposadr[i]
            mj_data.qpos[adr + 0] = x_mj
            mj_data.qpos[adr + 1] = y_mj
            mj_data.qpos[adr + 2] = z_mj

            # trajectory record
            if record_trail:
                _push_point(hunter_trails[i], np.array([x_mj, y_mj, z_mj], dtype=np.float64), maxlen)

            vel = env.hunters[i].vel
            vx = float(vel[0])
            vy = float(vel[1])
            speed = np.hypot(vx, vy)

            if speed > 1e-3:
                yaw = np.arctan2(vy, vx)
                half = 0.5 * yaw
                qw = np.cos(half)
                qz = np.sin(half)

                mj_data.qpos[adr + 3] = qw
                mj_data.qpos[adr + 4] = 0.0
                mj_data.qpos[adr + 5] = 0.0
                mj_data.qpos[adr + 6] = qz

        # ---- target ----
        if hasattr(env, "targets") and len(env.targets) > 0:
            t_pos = env.targets[0].pos
            tx_env = float(t_pos[0])
            ty_env = float(t_pos[1])
            tz_env = 0.1 if len(t_pos) < 3 else float(t_pos[2])

            tx_mj = tx_env * POS_SCALE
            ty_mj = ty_env * POS_SCALE
            tz_mj = tz_env * POS_SCALE

            adr_t = target_qposadr
            mj_data.qpos[adr_t + 0] = tx_mj
            mj_data.qpos[adr_t + 1] = ty_mj
            mj_data.qpos[adr_t + 2] = tz_mj

            # trajectory record
            if record_trail:
                _push_point(target_trail, np.array([tx_mj, ty_mj, tz_mj], dtype=np.float64), maxlen)

            t_vel = env.targets[0].vel
            tvx = float(t_vel[0])
            tvy = float(t_vel[1])
            tspeed = np.hypot(tvx, tvy)

            if tspeed > 1e-3:
                yaw_t = np.arctan2(tvy, tvx)
                half_t = 0.5 * yaw_t
                qw_t = np.cos(half_t)
                qz_t = np.sin(half_t)

                mj_data.qpos[adr_t + 3] = qw_t
                mj_data.qpos[adr_t + 4] = 0.0
                mj_data.qpos[adr_t + 5] = 0.0
                mj_data.qpos[adr_t + 6] = qz_t

        # ---- obstacles ----
        if hasattr(env, "obstacles") and len(obs_qposadr) > 0:
            num_env_obs = len(env.obstacles)
            num_mj_obs = len(obs_qposadr)
            num_sync = min(num_env_obs, num_mj_obs)

            for k in range(num_sync):
                obs = env.obstacles[k]
                o_pos = obs.pos
                ox_env = float(o_pos[0])
                oy_env = float(o_pos[1])
                oz_env = 0.05 if len(o_pos) < 3 else float(o_pos[2])

                ox_mj = ox_env * POS_SCALE
                oy_mj = oy_env * POS_SCALE
                oz_mj = oz_env * POS_SCALE

                adr = obs_qposadr[k]
                mj_data.qpos[adr + 0] = ox_mj
                mj_data.qpos[adr + 1] = oy_mj
                mj_data.qpos[adr + 2] = oz_mj

                if hasattr(obs, "radius") and k < len(obs_geom_ids):
                    r_env = float(obs.radius)
                    r_mj = r_env * POS_SCALE
                    gid = obs_geom_ids[k]
                    mj_model.geom_size[gid][0] = r_mj

            for k in range(num_sync, len(obs_qposadr)):
                adr = obs_qposadr[k]
                mj_data.qpos[adr + 0] = 0.0
                mj_data.qpos[adr + 1] = 0.0
                mj_data.qpos[adr + 2] = -10.0

        mujoco.mj_forward(mj_model, mj_data)

    with mjviewer.launch_passive(mj_model, mj_data) as mj_viewer:
        print("MuJoCo viewer started for 3D visualization.")

        maxgeom = mj_viewer.user_scn.maxgeom
        safe_points = max(10, (maxgeom // (n_hunters + 1)) + 1)
        MAX_TRAIL_POINTS = min(MAX_TRAIL_POINTS, safe_points)
        print(f"[trail] user_scn.maxgeom={maxgeom}, MAX_TRAIL_POINTS set to {MAX_TRAIL_POINTS}")

        center_xy = 2.0 * POS_SCALE

        cam = mj_viewer.cam
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[0] = center_xy
        cam.lookat[1] = center_xy
        cam.lookat[2] = 0.0
        cam.distance = 5.3 * POS_SCALE
        cam.azimuth = 0.0
        cam.elevation = -90.0

        for frame in range(max_frames):

            frame_start_time = time.time()

            if not mj_viewer.is_running():
                print("MuJoCo viewer closed by user, stopping test loop.")
                break

            preds = SPDM.update_and_predict(hunters_obs)
            preds_buffer.append(preds)

            aug_obs = [
                np.concatenate([hunters_obs[i], preds[i].reshape(-1)])
                for i in range(n_hunters)
            ]

            hunters_actions, modified = agents.agents_choose_action(
                aug_obs, total_steps, True
            )
            targets_actions = env.targets_choose_action()

            hunters_next_obs, _, hunters_rewards, _, hunter_dones, _ = env.step(
                hunters_actions, targets_actions, modified
            )
            rewards += hunters_rewards

            step_mse = np.nan
            if len(preds_buffer) >= future_steps and len(obs_records) >= future_steps + 1:
                pred_block = np.array(list(preds_buffer[-future_steps].values()))
                true_block = np.stack([obs_records[-future_steps + t] for t in range(future_steps)])
                true_block = np.transpose(true_block, (1, 0, 2))
                diff = pred_block - true_block
                step_mse = float(np.mean(diff ** 2))

            print(f"Frame {frame} | Rewards: {str(rewards.tolist()):<90} | MSE: {step_mse}")

            record = (frame % TRAIL_EVERY_N_FRAMES == 0)
            sync_mujoco_from_env(record_trail=record, maxlen=MAX_TRAIL_POINTS)
            draw_trails(mj_viewer)
            mj_viewer.sync()

            hunters_obs = hunters_next_obs
            obs_records.append(hunters_obs)
            total_steps += 1

            if any(hunter_dones):
                global final_frame
                final_frame = frame
                print("Round-up finished in", frame, "steps.")
                break

            # 控制每帧时长
            elapsed = time.time() - frame_start_time
            sleep_t = SECONDS_PER_FRAME - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    print("Collision has occurred ", env.collision_num, " times")


if __name__ == '__main__':
    test()
