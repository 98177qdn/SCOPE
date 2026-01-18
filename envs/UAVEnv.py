import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import matplotlib.image as mpimg
import os
from gymnasium import spaces

import torch
from envs.math_tools import *
import matplotlib.backends.backend_agg as agg
import random

boundary_length = 4.0
boundary_width = 4.0
boundary_height = 4.0

eps = 1e-6

# Definition of UAV agent
class UAVAgent:
    def __init__(self, number, pos, v_max, a_max, num_lasers, L_sensor, d_safe, dim, vel=None):
        self.number = number  # UAV id
        self.pos = np.array(pos, dtype=np.float32)  # UAV position
        self.last_pos = np.array(pos, dtype=np.float32)  # UAV position of last step
        self.prev_pos = np.array(pos, dtype=np.float32)  # UAV position of preview test
        self.v_max = v_max  # max velocity
        self.a_max = a_max  # max acceleration
        self.num_lasers = num_lasers  # the quantity of lasers
        self.L_sensor = L_sensor  # the length of lasers
        self.lasers = np.full((num_lasers,), L_sensor, dtype=np.float32)  # lasers list
        self.safe_radius = d_safe
        self.dim = dim

        self.collision_num = 0

        if vel is not None:
            vel = np.array(vel, dtype=np.float32)
            speed = np.linalg.norm(vel)
            if speed > self.v_max:
                vel = vel / speed * self.v_max
            self.vel = vel
        else:
            self.vel = np.zeros(self.dim, dtype=np.float32)

    def update_velocity(self, action, time_step):
        self.vel += action * time_step
        speed = np.linalg.norm(self.vel)
        if speed > self.v_max:
            self.vel = self.vel / speed * self.v_max

    def get_preview_position(self, time_step):
        self.prev_pos = self.pos + self.vel * time_step


# Hunter UAV
class Hunter(UAVAgent):
    def __init__(self, number, pos, v_max, a_max, num_lasers, L_sensor, d_safe, dim, interference_radius, interference_strength, **kwargs):
        super().__init__(number=number, pos=pos, v_max=v_max, a_max=a_max, num_lasers=num_lasers, L_sensor=L_sensor, d_safe=d_safe, dim=dim, **kwargs)
        self.interference_radius = interference_radius
        self.interference_strength = interference_strength

# Target UAV
class Target(UAVAgent):
    def __init__(self, number, pos, v_max, a_max, num_lasers, L_sensor, d_safe, dim, interference_decay_rate,**kwargs):
        super().__init__(number=number, pos=pos, v_max=v_max, a_max=a_max, num_lasers=num_lasers, L_sensor=L_sensor, d_safe=d_safe, dim=dim, **kwargs)
        self.max_interference = 100
        self.interference = 0
        self.interference_decay_rate = interference_decay_rate

    def interference_decay(self):
        self.interference = self.interference * (1 - self.interference_decay_rate)

# Definition of obstacle
class Obstacle:
    def __init__(self, length, width):
        self.pos = np.array([
            np.random.uniform(0.45, length - 0.55),
            np.random.uniform(0.6, width - 0.55)
        ])
        angle = np.random.uniform(0, 2 * np.pi)
        speed = 0.02  # Speed is zero
        self.vel = np.array([np.cos(angle), np.sin(angle)]) * speed
        self.radius = np.random.uniform(0.1, 0.15)

class UAVEnv:
    def __init__(self, dim_actions, length=boundary_length, width=boundary_width, height=boundary_height, num_obstacle=7, num_hunters=4, num_targets=1):
        self.dim_actions = dim_actions
        self.length = length  # x length of boundary
        self.width = width  # y width of boundary

        if self.dim_actions == 2:
            self.height = 0
        elif self.dim_actions == 3:
            self.height = height  # z height of boundary
        else:
            raise Exception("Dimension must be 2 or 3")

        self.num_obstacle, self.num_hunters, self.num_targets = num_obstacle, num_hunters, num_targets
        self.num_agents = num_hunters + num_targets

        self.time_step = 0.5 # update time step
        self.d_capture = 0.2  # distance about capture
        self.d_escape = 0.5  # distance for target to escape
        self.corner_capture = 0.2
        self.boundary_capture = 0.05
        self.dim_actions = dim_actions

        self.safe_value = np.zeros(num_hunters, dtype=np.float32)

        self.v_max = 0.12
        self.v_max_e = 0.12
        self.a_max = 0.04
        self.a_max_e = 0.035
        self.d_safe = 0.03
        self.d_safe_e = 0.04
        self.L_sensor = 0.25
        self.num_lasers = 16 # num of laser beams

        # The decay rate of interference
        self.interference_decay_rate = 0.1
        self.interference_radius = 0.2
        self.interference_strength = 2

        self.targets = None
        self.hunters = None
        self.obstacles = None
        self.hunters_history_positions = [[] for _ in range(num_hunters)]
        self.targets_history_positions = [[] for _ in range(num_targets)]

        # action_space and observation_space
        self.hunters_action_space = {f'hunter_{i}': spaces.Box(low=-np.inf, high=np.inf, shape=(dim_actions,)) for i in
                             range(self.num_hunters)}
        self.targets_action_space = {f'target_{i}': spaces.Box(low=-np.inf, high=np.inf, shape=(dim_actions,)) for i in
                             range(self.num_hunters)}

        self.hunters_observation_space = {f'agent_{i}': spaces.Box(low=-np.inf, high=np.inf, shape=(2 * dim_actions + dim_actions * (num_hunters - 1) + self.num_lasers + 2 * num_targets,)) for i in
                                  range(self.num_hunters)}
        self.targets_observation_space = {f'agent_{i}':spaces.Box(low=-np.inf, high=np.inf, shape=(2 * dim_actions + self.num_lasers + 2 * num_hunters,)) for i in range(self.num_targets)}

    def reset(self):
        seed = random.randint(1, 1000)
        random.seed(seed)
        # print(f"seed: {seed}")

        # Initialize agents and obstacles
        self.hunters = []
        self.targets = []
        self.hunters_history_positions = [[] for _ in range(self.num_hunters)]
        self.targets_history_positions = [[] for _ in range(self.num_targets)]
        self.obstacles = [Obstacle(self.length, self.width) for _ in range(self.num_obstacle)]
        self.safe_value = np.zeros(self.num_hunters, dtype=np.float32)

        self.collision_num = 0

        # Define the low and high bounds for the hunters and targets' positions based on dim_actions
        if self.dim_actions == 3:
            low_hunter = np.array([0.1, 0.1, 0.1])
            high_hunter = np.array([self.length - 0.1, self.width - 0.1, self.height - 0.1])

            low_target = np.array([self.length / 2 - 0.3, self.width / 2 - 0.3, self.height / 2 - 0.3])
            high_target = np.array([self.length / 2 + 0.3, self.width / 2 + 0.3, self.height / 2 + 0.3])
        else:
            low_hunter = np.array([0.1, 0.1])
            high_hunter = np.array([self.length - 0.1, 0.4])

            low_target = np.array([self.length / 2 - 0.3, self.length / 2 - 0.3])
            high_target = np.array([self.length / 2 + 0.3, self.length / 2 + 0.3])


        # Generate Hunters
        for i in range(self.num_hunters):
            while True:
                # Sample random positions using numpy's random.uniform for both 2D and 3D
                summon_test_pos = np.random.uniform(low_hunter, high_hunter)

                # Check for conflicts with other hunters
                no_conflict_hunters = all(
                    np.linalg.norm(hunter.pos - summon_test_pos) > 2 * self.d_safe
                    for hunter in self.hunters
                )

                # Check for conflicts with obstacles
                no_conflict_obstacles = all(
                    np.linalg.norm(obstacle.pos - summon_test_pos) > self.d_safe + obstacle.radius
                    for obstacle in self.obstacles
                )

                if no_conflict_hunters and no_conflict_obstacles:
                    self.hunters.append(Hunter(
                        number=i,
                        pos=summon_test_pos,
                        v_max=self.v_max,
                        a_max=self.a_max,
                        num_lasers=self.num_lasers,
                        L_sensor=self.L_sensor,
                        d_safe=self.d_safe,
                        dim=self.dim_actions,
                        interference_radius=self.interference_radius,
                        interference_strength=self.interference_strength
                    ))
                    break

        # Generate Targets
        for i in range(self.num_targets):
            while True:
                summon_test_pos = np.random.uniform(low_target, high_target)

                # Check for conflicts with obstacles
                no_conflict_obstacles = all(
                    np.linalg.norm(obstacle.pos - summon_test_pos) > self.d_safe_e + obstacle.radius
                    for obstacle in self.obstacles
                )

                if no_conflict_obstacles:
                    self.targets.append(Target(
                        number=i,
                        pos=summon_test_pos,
                        v_max=self.v_max_e,
                        a_max=self.a_max_e,
                        num_lasers=self.num_lasers,
                        L_sensor=self.L_sensor,
                        d_safe=self.d_safe_e,
                        dim=self.dim_actions,
                        interference_decay_rate = self.interference_decay_rate
                    ))
                    break

        # Update lasers
        self.update_hunters_lasers()
        self.update_targets_lasers()

        # obs is list of agent_obs
        hunters_obs, targets_obs = self.get_multiUAVs_observation()
        return hunters_obs, targets_obs

    def targets_choose_action(self):
        targets_actions = []
        N_dirs = 16

        # === direction ===
        if self.dim_actions == 2:
            angles = np.linspace(0, 2 * np.pi, N_dirs, endpoint=False)
            dir_vectors = np.array([[np.cos(a), np.sin(a)] for a in angles])

        elif self.dim_actions == 3:
            phi = np.linspace(0, 2 * np.pi, int(np.sqrt(N_dirs)), endpoint=False)
            theta = np.linspace(0, np.pi, int(np.sqrt(N_dirs)), endpoint=False)
            dir_vectors = []
            for t in theta:
                for p in phi:
                    x = np.sin(t) * np.cos(p)
                    y = np.sin(t) * np.sin(p)
                    z = np.cos(t)
                    dir_vectors.append([x, y, z])
            dir_vectors = np.array(dir_vectors)
            N_dirs = dir_vectors.shape[0]
        else:
            raise ValueError("Unsupported dim_actions")

        for target in self.targets:
            threat_per_dir = np.zeros(N_dirs)

            # === hunters threaten ===
            for hunter in self.hunters:
                delta = hunter.pos - target.pos
                dist = np.linalg.norm(delta)
                dist = max(dist, eps)  # 防止 0 距离
                if dist < self.d_escape:
                    dir = delta / dist
                    strength = min(1.0, 0.1 / (dist ** 2 + eps))
                    cosines = dir_vectors @ dir
                    cosines = np.clip(cosines, 0.0, 1.0)
                    threat_per_dir += strength * cosines

            # === obstacles threaten ===
            for obs in self.obstacles:
                delta = obs.pos - target.pos
                dist = np.linalg.norm(delta) - obs.radius
                dist = max(dist, 0.01)  # 避免负数和 0
                if dist < self.d_escape:
                    dir = delta / dist
                    strength = min(1.0, 0.02 / (dist ** 2 + eps))
                    cosines = dir_vectors @ dir
                    cosines = np.clip(cosines, 0.0, 1.0)
                    threat_per_dir += strength * cosines

            # === border threaten ===
            if self.dim_actions == 2:
                target.pos = np.clip(target.pos, [0, 0], [self.length, self.width])
                x, y = target.pos
                boundaries = [
                    (np.array([0, y]), x),
                    (np.array([self.length, y]), self.length - x),
                    (np.array([x, 0]), y),
                    (np.array([x, self.width]), self.width - y)
                ]
            else:
                target.pos = np.clip(target.pos, [0, 0, 0], [self.length, self.width, self.height])
                x, y, z = target.pos
                boundaries = [
                    (np.array([0, y, z]), x),
                    (np.array([self.length, y, z]), self.length - x),
                    (np.array([x, 0, z]), y),
                    (np.array([x, self.width, z]), self.width - y),
                    (np.array([x, y, 0]), z),
                    (np.array([x, y, self.height]), self.height - z)
                ]

            for boundary_point, dist in boundaries:
                dist = max(dist, 0.01)
                if dist < self.d_escape:
                    delta = boundary_point - target.pos
                    dir = delta / dist
                    strength = min(1.0, 0.03 / (dist ** 2 + eps))
                    cosines = dir_vectors @ dir
                    cosines = np.clip(cosines, 0.0, 1.0)
                    threat_per_dir += strength * cosines

            # === escape ===
            threat_per_dir = np.sqrt(np.maximum(threat_per_dir, 0))  # 防止 sqrt 负数
            min_threat = np.min(threat_per_dir)
            total_threat = np.sum(threat_per_dir) + eps

            if min_threat > 0.5:
                acc = -target.vel * 0.5
            else:
                weighted_threat_dirs = dir_vectors * threat_per_dir[:, None]
                threat_vector = np.sum(weighted_threat_dirs, axis=0)
                threat_norm = np.linalg.norm(threat_vector)
                if threat_norm < eps:
                    threat_vector = np.zeros_like(threat_vector)
                    threat_norm = 1.0

                min_idx = np.argmin(threat_per_dir)
                min_dir = dir_vectors[min_idx]

                avg_dir = -threat_vector / threat_norm
                w = 0.6
                escape_dir = w * min_dir + (1 - w) * avg_dir

                # === Tangential disturbance ===
                if self.dim_actions == 2:
                    tangent = np.array([-escape_dir[1], escape_dir[0]])
                else:
                    rand_vec = np.random.randn(3)
                    tangent = np.cross(escape_dir, rand_vec)
                    if np.linalg.norm(tangent) < eps:
                        tangent = np.cross(escape_dir, [1, 0, 0])
                tangent_norm = np.linalg.norm(tangent)
                if tangent_norm > eps:
                    tangent /= tangent_norm
                else:
                    tangent = np.zeros_like(escape_dir)

                theta = 0.1 + 0.3 * total_threat / N_dirs
                escape_dir = (1 - theta) * escape_dir + theta * tangent
                norm_escape = np.linalg.norm(escape_dir)
                if norm_escape > eps:
                    escape_dir /= norm_escape
                else:
                    escape_dir = np.zeros_like(escape_dir)

                acc_magnitude = self.a_max_e / (1 + np.exp(-5 * (total_threat - 1)))
                acc = escape_dir * acc_magnitude * 0.5

                for obs in self.obstacles:
                    delta = target.pos - obs.pos
                    dist = np.linalg.norm(delta) - obs.radius - target.safe_radius
                    if dist < self.L_sensor:
                        force = 0.03 / (1 + np.exp(10 * (dist - 0.5)))
                        dir = delta / (max(dist, eps))
                        acc += dir * force

            # === Boundary rebound force ===
            margin = 1.0
            k_wall = 0.03
            if self.dim_actions == 2:
                x, y = target.pos
                if x < margin:
                    acc[0] += k_wall / (x + eps)
                elif x > self.length - margin:
                    acc[0] -= k_wall / (self.length - x + eps)
                if y < margin:
                    acc[1] += k_wall / (y + eps)
                elif y > self.width - margin:
                    acc[1] -= k_wall / (self.width - y + eps)
            else:
                x, y, z = target.pos
                if x < margin:
                    acc[0] += k_wall / (x + eps)
                elif x > self.length - margin:
                    acc[0] -= k_wall / (self.length - x + eps)
                if y < margin:
                    acc[1] += k_wall / (y + eps)
                elif y > self.width - margin:
                    acc[1] -= k_wall / (self.width - y + eps)
                if z < margin:
                    acc[2] += k_wall / (z + eps)
                elif z > self.height - margin:
                    acc[2] -= k_wall / (self.height - z + eps)

            # Limit maximum acceleration
            norm_acc = np.linalg.norm(acc)
            if norm_acc > self.a_max_e:
                acc = acc / norm_acc * self.a_max_e

            assert not np.any(np.isnan(acc)), f"NaN detected in acc: {acc}"
            targets_actions.append(acc)

        return targets_actions

    # Collision determination of obstacle
    def UAV_collision_with_obstacle(self, uavAgent: UAVAgent):
        is_collided_obstacles = False
        modified_distance = []
        direction = uavAgent.prev_pos - uavAgent.pos
        max_distance = np.linalg.norm(direction)

        obs_safe_value = []
        boundary_safe_value = []

        for obs in self.obstacles:
            distance_to_obs = np.linalg.norm(obs.pos - uavAgent.prev_pos)
            r = uavAgent.safe_radius + obs.radius
            isInObs = distance_to_obs < r

            obs_safe_value.append(r - distance_to_obs)

            # Determine whether a collision has occurred
            if isInObs:
                is_collided_obstacles = True
                self.collision_num += 1

                if max_distance < eps:
                    modified_distance.append(0.0)
                    continue

                angle = np.arctan2(direction[1], direction[0])

                # Calculate the distance to collision position
                dist_to_collision = check_obs_intersection(uavAgent.pos, angle, obs.pos, r, max_distance)
                # Record modified position
                modified_distance.append(dist_to_collision)

        # Prevent UAV from entering obstacles
        if modified_distance:
            # if max_distance > eps:
                uavAgent.prev_pos = uavAgent.pos + min(modified_distance) * (direction / (max_distance + eps))
            # else:
            #     uavAgent.prev_pos = uavAgent.pos  # 无位移

        # Detection boundary
        for dim in [0, 1]:
            boundary = self.length if dim == 0 else self.width
            boundary_safe_value.append(uavAgent.safe_radius - uavAgent.prev_pos[dim])
            boundary_safe_value.append(uavAgent.safe_radius + uavAgent.prev_pos[dim] - boundary)

            if uavAgent.prev_pos[dim] - uavAgent.safe_radius < 0:
                is_collided_obstacles = True
                uavAgent.prev_pos[dim] = uavAgent.safe_radius
                uavAgent.vel[dim] = 0.0

            elif uavAgent.prev_pos[dim] + uavAgent.safe_radius > boundary:
                is_collided_obstacles = True
                uavAgent.prev_pos[dim] = boundary - uavAgent.safe_radius
                uavAgent.vel[dim] = 0.0
        if uavAgent in self.hunters:
            self.safe_value[uavAgent.number] += np.clip(max(obs_safe_value),0.0, None) + np.clip(max(boundary_safe_value),0.0, None) + float(is_collided_obstacles)

        return is_collided_obstacles


    # Collision determination of UAVs
    def UAV_collision_with_UAVs(self, uavAgent: UAVAgent, hunter_Collided_uavs, target_Collided_uavs):
        is_collided_uavs = False

        agent_safe_value = []
        # Check all hunters
        for hunter in self.hunters:
            if uavAgent is not hunter:  # pass itself
                distance_to_hunter = np.linalg.norm(uavAgent.prev_pos - hunter.prev_pos)
                r = uavAgent.safe_radius + hunter.safe_radius
                isCollided = distance_to_hunter < r

                agent_safe_value.append(r - distance_to_hunter)

                # Determine whether a collision has occurred
                if isCollided:
                    # Update hunters collision table
                    hunter_Collided_uavs[hunter.number], is_collided_uavs = True, True
                    self.collision_num += 1

                    # Prevent UAV from entering each other
                    uavAgent_test_pos, hunter_test_pos = compute_collision_positions(
                        uavAgent.pos, hunter.pos, uavAgent.vel, hunter.vel, uavAgent.safe_radius, hunter.safe_radius, self.time_step
                    )

                    uavAgent.prev_pos = select_nearest(uavAgent.pos, uavAgent_test_pos, uavAgent.prev_pos)
                    hunter.prev_pos = select_nearest(hunter.pos, hunter_test_pos, hunter.prev_pos)

        # Check all targets
        for target in self.targets:
            if uavAgent is not target:  # pass itself
                distance_to_target = np.linalg.norm(uavAgent.prev_pos - target.prev_pos)
                r = uavAgent.safe_radius + target.safe_radius
                isCollided = distance_to_target < r

                agent_safe_value.append(r - distance_to_target)

                # Determine whether a collision has occurred
                if isCollided:
                    # Update targets collision table
                    target_Collided_uavs[target.number], is_collided_uavs = True, True
                    self.collision_num += 1

                    # Prevent UAV from entering each other
                    uavAgent_test_pos, target_test_pos = compute_collision_positions(
                        uavAgent.pos, target.pos, uavAgent.vel, target.vel, uavAgent.safe_radius, target.safe_radius, self.time_step
                    )

                    uavAgent.prev_pos = select_nearest(uavAgent.pos, uavAgent_test_pos, uavAgent.prev_pos)
                    target.prev_pos = select_nearest(target.pos, target_test_pos, target.prev_pos)

        if uavAgent in self.hunters:
            self.safe_value[uavAgent.number] += np.clip(max(agent_safe_value),0.0, None) + float(is_collided_uavs)

        return is_collided_uavs

    # Step for only one target
    def step(self, hunters_actions, targets_actions, modified):
        if self.num_targets != 1:
            raise Exception("This step function is used for only one agent")

        # Update velocity and prev_pos for hunters
        for i in range(self.num_hunters):
            self.hunters[i].update_velocity(hunters_actions[i], self.time_step)
            self.hunters[i].get_preview_position(self.time_step)

        # Update velocity and prev_pos for targets
        for i in range(self.num_targets):
            self.targets[i].update_velocity(targets_actions[i], self.time_step)
            self.targets[i].get_preview_position(self.time_step)

        # Update obstacle positions
        for obs in self.obstacles:
            obs.pos += obs.vel * self.time_step

            # Check for boundary collisions and adjust velocities
            for dim in [0, 1]:
                boundary = self.length if dim == 0 else self.width

                if obs.pos[dim] - obs.radius < 0:
                    obs.pos[dim] = obs.radius
                    obs.vel[dim] *= -1
                elif obs.pos[dim] + obs.radius > boundary:
                    obs.pos[dim] = boundary - obs.radius
                    obs.vel[dim] *= -1

        self.safe_value = np.zeros(self.num_hunters, dtype=np.float32)

        # Dealing with collision of obstacles
        hunter_Collided_obstacles = [False] * self.num_hunters
        target_Collided_obstacles = [False] * self.num_targets

        for hunter in self.hunters:
            self.safe_value[hunter.number] += modified[hunter.number]
            hunter_Collided_obstacles[hunter.number] = self.UAV_collision_with_obstacle(hunter)

        for target in self.targets:
            target_Collided_obstacles[target.number] = self.UAV_collision_with_obstacle(target)

        # Dealing with collisions of UAVs
        hunter_Collided_uavs = [False] * self.num_hunters
        target_Collided_uavs = [False] * self.num_targets
        for hunter in self.hunters:
            hunter_Collided_uavs[hunter.number] = self.UAV_collision_with_UAVs(hunter, hunter_Collided_uavs,
                                                                               target_Collided_uavs)

        for target in self.targets:
            target_Collided_uavs[target.number] = self.UAV_collision_with_UAVs(target, hunter_Collided_uavs,
                                                                               target_Collided_uavs)
        # Update positions
        for hunter in self.hunters:
            hunter.pos = hunter.prev_pos

        for target in self.targets:
            target.pos = target.prev_pos

        # Update interference
        self.update_interference()

        # Update lasers
        self.update_hunters_lasers()
        self.update_targets_lasers()

        # Calculate rewards
        hunters_rewards, targets_rewards, hunter_dones = self.cal_rewards_dones(hunter_Collided_obstacles, target_Collided_obstacles, hunter_Collided_uavs, target_Collided_uavs)
        hunters_next_obs, targets_next_obs = self.get_multiUAVs_observation()

        return hunters_next_obs, targets_next_obs, hunters_rewards, targets_rewards, hunter_dones, self.safe_value

    def get_multiUAVs_observation(self):
        hunters_total_obs = []

        # Get hunters' observation
        for hunter in self.hunters:
            # S_self
            S_self = np.concatenate([
                hunter.pos / np.array([self.length, self.width, self.height][:hunter.dim]),
                hunter.vel / hunter.v_max
            ])  # dim == 2 * dim_actions
            S_self = np.concatenate([S_self, np.array(hunter.lasers)])  # dim == self.num_lasers

            # S_ally
            S_ally = np.concatenate([
                np.array([ally.pos[0] / self.length, ally.pos[1] / self.width])
                for ally in self.hunters if ally is not hunter
            ])

            # S_target
            targets_pos = np.array([target.pos for target in self.targets])  # shape: [num_target, 2]

            # extend hunter's position
            hunter_pos_expand = np.expand_dims(hunter.pos, axis=0).repeat(targets_pos.shape[0], axis=0)

            # difference vector
            delta_pos = targets_pos - hunter_pos_expand  # shape: [num_target, 2]

            # Calculate distances and angles
            distances = np.linalg.norm(delta_pos, axis=1) / np.linalg.norm(np.array([self.length, self.width]), axis=0)  # shape: [num_target]
            angles = np.arctan2(delta_pos[:, 1], delta_pos[:, 0])  # shape: [num_target]
            S_target = np.concatenate((distances, angles))  # shape: [2 * num_target]

            # Combine all observation
            single_obs = np.concatenate([S_self, S_ally, S_target])
            hunters_total_obs.append(single_obs)

        hunters_total_obs = np.stack(hunters_total_obs, axis=0)

        # The following can be removed when targets are using predefined rules
        target_total_obs = []

        for target in self.targets:
            # S_self
            S_self = np.array([
                target.pos[0] / self.length,
                target.pos[1] / self.length,
                target.vel[0] / target.v_max,
                target.vel[1] / target.v_max
            ])  # dim == 4
            S_self = np.concatenate([S_self, np.array(target.lasers)])  # dim == self.num_lasers

            # S_evade
            hunters_pos = np.array([hunter.pos for hunter in self.hunters])  # shape: [num_hunter, 2]

            # extend target's position
            target_pos_expand = np.expand_dims(target.pos, axis=0).repeat(hunters_pos.shape[0], axis=0)

            # difference vector
            delta = hunters_pos - target_pos_expand  # shape: [num_hunter, 2]

            # Calculate distances and angles
            distances = np.linalg.norm(delta, axis=1) / np.linalg.norm(np.array([self.length, self.length]), axis=0)  # shape: [num_hunter]
            angles = np.arctan2(delta[:, 1], delta[:, 0])  # shape: [num_hunter]
            S_evade = np.concatenate((distances, angles))  # shape: [2 * num_hunter]

            # Combine all observation
            single_obs = np.concatenate([S_self, S_evade])
            target_total_obs.append(single_obs)

        target_total_obs = np.stack(target_total_obs, axis=0)

        return hunters_total_obs, target_total_obs

    # Rewards for only one target
    def cal_rewards_dones(self, hunter_Collided_obstacles, target_Collided_obstacles, hunter_Collided_uavs,
                          target_Collided_uavs):
        # Label of whether game is finished
        hunters_dones = [False] * self.num_hunters

        # Rewards of hunters and targets
        hunters_rewards = np.zeros(self.num_hunters)
        target_rewards = np.zeros(self.num_targets)

        # Rewards weight
        mu = {
            'sync': 0.5, 'near': 0.9, 'safe_target': 0.7, 'safe_hunter': 1, 'co_dis': 0.5, 'finish': 1.0,
            'escape': 0.8, 'angle': 1.0, 'encircle': 0.4, 'dis': 0.6, 'close': 1.2, 'center': 0.2, 'tardis': 0.7, 'interference': 1
        }

        ## ================ Param Initialization ================ ##
        # Hunters' position list
        hunters_pos = np.array([hunter.pos for hunter in self.hunters])
        hunters_vel = np.array([hunter.vel for hunter in self.hunters])

        # Targets' position list
        targets_pos = np.array([target.pos for target in self.targets])
        targets_vel = np.array([target.vel for target in self.targets])

        # Calculate cluster_center
        hunters_cluster_centers = np.mean(hunters_pos, axis=0)

        # Get the distance from hunters to targets
        hunters_dists_to_targets = np.linalg.norm(hunters_pos[:, None] - targets_pos, axis=2)

        # Get the distance to the nearest target
        min_dists = np.min(hunters_dists_to_targets, axis=1)
        target_indices = np.argmin(hunters_dists_to_targets, axis=1)

        ## Calculate away factors
        # Get safe radius of hunters and targets
        hunter_radius = np.array([hunter.safe_radius for hunter in self.hunters])
        target_radius = np.array([target.safe_radius for target in self.targets])

        target_radius = target_radius[target_indices]  # shape: [num_hunters]
        min_dists_except_radius = min_dists - (hunter_radius + target_radius)  # shape: [num_hunters]

        # Factors that increase with distance
        away_factor = 2 * min_dists_except_radius / (0.2 + min_dists_except_radius)

        # Factor that declines with distance
        close_factor = np.exp(-1 * away_factor)

        ## ================ Hunters Rewards ================ ##
        # ---------------- 1. Collision rewards ----------------
        r_safe = np.array([
            (np.min(hunter.lasers) - hunter.L_sensor) / hunter.L_sensor
            for hunter in self.hunters
        ])
        # Numerical collision situation
        hunter_obstacle_collision_flags = np.array(hunter_Collided_obstacles, dtype=np.float32)
        hunter_uavs_collision_flags = np.array(hunter_Collided_uavs, dtype=np.float32)

        hunter_uavs_collision_flags *= away_factor
        r_safe += -2.0 * (hunter_obstacle_collision_flags + hunter_uavs_collision_flags)
        hunters_rewards += mu['safe_hunter'] * r_safe

        # ---------------- 2. Interval rewards ----------------
        interval_with_ally = np.linalg.norm(hunters_pos[:, None] - hunters_pos, axis=2)
        hunter_safe_radius = np.array([hunter.safe_radius for hunter in self.hunters])
        hunter_safe_matrix = hunter_safe_radius[:, None] + hunter_safe_radius[None, :]
        interval_with_ally -= hunter_safe_matrix

        np.fill_diagonal(interval_with_ally, np.inf)  # 排除自己
        min_interval = np.min(interval_with_ally, axis=1)
        r_sync = (5.66 - 4 * (min_interval + 0.56) - 1 / (0.5 * (min_interval + 0.56))) * away_factor
        hunters_rewards += mu['sync'] * r_sync

        # ---------------- 3. Capture rewards ----------------
        # Here is only suitable for single agents, target number is 0
        single_target = self.targets[0]
        Sum_S = calculate_all_triangle(hunters_pos, single_target.pos)
        S_round = calculate_polygon_area(hunters_pos)

        # velocity rewards
        single_target_pos = single_target.pos
        hunters_vel_magnitude = np.linalg.norm(hunters_vel, axis=1)
        hunter_v_max = np.array([hunter.v_max for hunter in self.hunters])
        direction_vectors = single_target_pos - hunters_pos
        cos_v_d = torch.nn.functional.cosine_similarity(torch.tensor(hunters_vel), torch.tensor(direction_vectors), dim=1)
        cos_v_d = cos_v_d.numpy()
        r_near = hunters_vel_magnitude / hunter_v_max * cos_v_d * away_factor
        hunters_rewards += mu['near'] * r_near

        # distance rewards
        r_dis = np.exp(-6 * min_dists_except_radius)
        hunters_rewards += mu['dis'] * r_dis

        # ---------------- 4. Cooperation rewards ----------------
        # Encircle rewards
        r_encircle = np.exp(-4 * (Sum_S / S_round - 1)) * close_factor
        hunters_rewards += mu['encircle'] * r_encircle

        # Surround rewards
        r_close = -np.log(1 + S_round) * away_factor
        hunters_rewards += mu['close'] * r_close

        # Angle rewards
        angles = calculate_angles(hunters_pos, single_target_pos)
        ideal_angle = 2 * np.pi / self.num_hunters
        dev = np.abs(angles - ideal_angle)
        log_prod = np.sum(np.log(dev + eps)) / self.num_hunters  # eps 防止 log(0)
        aui = np.exp(log_prod)
        r_angles = np.exp(-2 * aui)
        hunters_rewards += mu['angle'] * r_angles * close_factor

        ## ================ Completion Judgment ================ ##
        # Here is only suitable for single agents, target number is 0
        hunters_rewards += mu['interference'] * sum([target.interference / target.max_interference / self.num_targets for target in self.targets])
        # ---------------- Jamming success ----------------
        if all(target.interference >= target.max_interference for target in self.targets) and not any(hunter_Collided_obstacles) and not any(hunter_Collided_uavs):
            hunters_rewards += mu['finish'] * 50
            hunters_dones = [True] * self.num_hunters

        # ===== Debug: whether nan occur ===== #
        if np.isnan(hunters_rewards).any() or np.isnan(target_rewards).any():
            print("❌ NaN detected in rewards!")

            # print hunters' rewards
            print("== Hunters Reward Breakdown ==")
            print("r_safe:", r_safe)
            print("away_factor:", away_factor)
            print("r_sync:", r_sync)
            print("r_near:", r_near)
            print("r_dis:", r_dis)
            print("r_encircle:", r_encircle)
            print("r_close:", r_close)
            print("r_angles:", r_angles)

            # 其他中间变量
            print("min_dists:", min_dists)
            print("min_dists_except_radius:", min_dists_except_radius)
            print("Sum_S:", Sum_S, "S_round:", S_round)
            print("target_pos:", single_target_pos)
            print("hunters_pos:", hunters_pos)

            raise ValueError("NaN detected in rewards! Check debug info above.")

        return hunters_rewards, target_rewards, hunters_dones

    # Update the length of hunters lasers
    def update_hunters_lasers(self):
        for hunter in self.hunters:
            # Initialize hunter's lasers
            hunter_current_lasers = np.full(hunter.num_lasers, hunter.L_sensor)

            for obs in self.obstacles:
                r = obs.radius + hunter.safe_radius

                # Update lasers about the obstacle
                _current_lasers = update_lasers_to_obstacle(hunter.pos, obs.pos, r, hunter.L_sensor, hunter.num_lasers)
                hunter_current_lasers = np.minimum(_current_lasers, hunter_current_lasers)

            _current_lasers = update_lasers_to_boundary(hunter.pos, hunter.L_sensor, hunter.num_lasers, self.length,
                                                        self.width)
            hunter.lasers = np.minimum(_current_lasers, hunter_current_lasers)

    # Update the length of targets lasers
    def update_targets_lasers(self):
        for target in self.targets:
            # Initialize target's lasers
            target_current_lasers = np.full(target.num_lasers, target.L_sensor)
            for obs in self.obstacles:
                r = obs.radius + target.safe_radius

                # Update lasers about the obstacle
                _current_lasers = update_lasers_to_obstacle(target.pos, obs.pos, r, target.L_sensor, target.num_lasers)
                target_current_lasers = np.minimum(_current_lasers, target_current_lasers)

            _current_lasers = update_lasers_to_boundary(target.pos, target.L_sensor, target.num_lasers, self.length,
                                                        self.width)
            target.lasers = np.minimum(_current_lasers, target_current_lasers)

    def update_interference(self):
        for target in self.targets:
            # number of interference
            interference_num = 0
            interference_value = 0
            for hunter in self.hunters:
                # Calculate the distance of every target
                distance = np.linalg.norm(target.pos - hunter.pos)

                # if target is in the interference area
                if distance <= hunter.interference_radius:
                    interference_num += 1
                    interference_value += hunter.interference_strength

            if interference_num >= 3:
                target.interference += interference_value
            elif interference_num == 0:
                target.interference_decay()


    def render(self):

        plt.clf()

        # load UAV icon
        hunter_icon = mpimg.imread('icons/UAV.png')
        target_icon = mpimg.imread('icons/target.png')
        icon_size = 0.1
        # icon_height, icon_width, _ = uav_icon.shape

        # plot round-up-UAVs
        for hunter in self.hunters:
            pos = hunter.pos
            vel = hunter.vel
            self.hunters_history_positions[hunter.number].append(pos)
            trajectory = np.array(self.hunters_history_positions[hunter.number])

            # plot trajectory
            plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', alpha=0.3)

            # Calculate the angle of the velocity vector
            angle = np.arctan2(vel[1], vel[0])
            t = transforms.Affine2D().rotate(angle).translate(pos[0].tolist(), pos[1].tolist())

            # Adjust this size to your icon's aspect ratio
            plt.imshow(hunter_icon, transform=t + plt.gca().transData,
                       extent=(-icon_size / 2, icon_size / 2, -icon_size / 2, icon_size / 2))

        # plot target
        for target in self.targets:
            pos = target.pos
            vel = target.vel
            self.targets_history_positions[target.number].append(pos)
            trajectory = np.array(self.targets_history_positions[target.number])

            # plot trajectory
            plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', alpha=0.3)

            # Calculate the angle of the velocity vector
            angle = np.arctan2(vel[1], vel[0])
            t = transforms.Affine2D().rotate(angle).translate(pos[0].tolist(), pos[1].tolist())

            # Adjust this size to your icon's aspect ratio
            plt.imshow(target_icon, transform=t + plt.gca().transData,
                       extent=(-icon_size / 2, icon_size / 2, -icon_size / 2, icon_size / 2))

        for obstacle in self.obstacles:
            circle = plt.Circle(obstacle.pos.detach().cpu().numpy(), obstacle.radius, color='gray', alpha=0.5)
            plt.gca().add_patch(circle)
        plt.xlim(-0.1, self.length+0.1)
        plt.ylim(-0.1, self.width+0.1)
        plt.draw()
        plt.legend()

        # Save the current figure to a buffer
        canvas = agg.FigureCanvasAgg(plt.gcf())
        canvas.draw()
        buf = canvas.buffer_rgba()

        # Convert buffer to a NumPy array
        image = np.asarray(buf)
        return image

    def render_anime_2D(self):
        plt.clf()

        env_dir = os.path.dirname(__file__)
        uav_icon = mpimg.imread(os.path.join(env_dir, 'icons', 'UAV.png'))
        target_icon = mpimg.imread(os.path.join(env_dir, 'icons', 'target.png'))
        icon_size = 0.03
        colors = ['gold', 'lime', 'navy', 'peru']

        # Draw the length of the trajectory
        trajectory_length = 40

        for hunter in self.hunters:
            pos = hunter.pos
            vel = hunter.vel
            angle = np.arctan2(vel[1], vel[0])
            self.hunters_history_positions[hunter.number].append(pos)

            x, y = pos[0], pos[1]

            # safe radius
            r = hunter.safe_radius
            a = np.arange(x - r, x + r, 0.0001)
            b = np.sqrt(r ** 2 - (a - x) ** 2)
            plt.plot(a, y + b, color='g', linestyle='-', alpha=0.3, linewidth=0.3)
            plt.plot(a, y - b, color='g', linestyle='-', alpha=0.3, linewidth=0.3)

            # sensor radius
            r = hunter.L_sensor
            a = np.arange(x - r, x + r, 0.0001)
            b = np.sqrt(r ** 2 - (a - x) ** 2)
            plt.plot(a, y + b, color='y', linestyle='--', alpha=0.3, linewidth=0.3)
            plt.plot(a, y - b, color='y', linestyle='--', alpha=0.3, linewidth=0.3)

            # trajectory
            trajectory = np.array(self.hunters_history_positions[hunter.number])
            for i in range(len(trajectory) - trajectory_length, len(trajectory) - 1):
                color = colors[hunter.number]
                plt.plot(trajectory[i:i + 2, 0], trajectory[i:i + 2, 1], color=color, alpha=0.7, linewidth=0.7)

            transform = transforms.Affine2D().rotate(angle).translate(x.tolist(), y.tolist())
            plt.imshow(uav_icon, transform=transform + plt.gca().transData,
                       extent=(-icon_size / 2, icon_size / 2, -icon_size / 2, icon_size / 2))

        for target in self.targets:
            pos = target.pos
            vel = target.vel
            angle = np.arctan2(vel[1], vel[0])
            self.targets_history_positions[target.number].append(pos)

            x, y = pos[0], pos[1]
            for radius, color, style, width in [
                (self.d_capture, 'r', '--', 1),
                (self.d_escape, 'blue', '--', 1),
                (target.safe_radius, 'g', '-', 0.3),
                (target.L_sensor, 'y', '--', 1),
            ]:
                a = np.arange(x - radius, x + radius, 0.0001)
                b = np.sqrt(radius ** 2 - (a - x) ** 2)
                plt.plot(a, y + b, color=color, linestyle=style, alpha=0.3, linewidth=width)
                plt.plot(a, y - b, color=color, linestyle=style, alpha=0.3, linewidth=width)

            transform = transforms.Affine2D().rotate(angle).translate(x.tolist(), y.tolist())
            plt.imshow(target_icon, transform=transform + plt.gca().transData,
                       extent=(-icon_size / 2, icon_size / 2, -icon_size / 2, icon_size / 2))

            trajectory = np.array(self.targets_history_positions[target.number])
            plt.plot(trajectory[-trajectory_length:, 0], trajectory[-trajectory_length:, 1], 'r-', alpha=0.3, linewidth=1)

        # borders
        plt.plot([0, self.length], [0, 0], color='black')  # 下
        plt.plot([0, self.length], [self.length, self.length], color='black')  # 上
        plt.plot([0, 0], [0, self.width], color='black')  # 左
        plt.plot([self.length, self.length], [0, self.width], color='black')  # 右

        # obstacles
        for obstacle in self.obstacles:
            circle = plt.Circle(obstacle.pos, obstacle.radius, color='gray', alpha=0.5)
            plt.gca().add_patch(circle)

        plt.xlim(-0.1, self.length + 0.1)
        plt.ylim(-0.1, self.length + 0.1)
        plt.draw()


