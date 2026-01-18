import math
import numpy as np

eps = 1e-6
# 1 simulate lidar

def update_lasers_to_obstacle(pos, obs_pos, r, L, num_lasers):
    angles = np.linspace(0, 2 * np.pi * (1 - 1 / num_lasers), num_lasers)
    laser_lengths = check_lasers_intersection(pos, angles, obs_pos, r, L)  # shape: [num_lasers]
    laser_lengths = np.clip(laser_lengths, 0.0, L)
    return laser_lengths

def update_lasers_to_boundary(pos, L, num_lasers, length, width):
    angles = np.linspace(0, 2 * np.pi * (1 - 1 / num_lasers), num_lasers)
    wall_dists = check_wall_intersection(pos, angles, length, width, L)
    wall_dists = np.clip(wall_dists, 0.0, L)
    return wall_dists

def check_lasers_intersection(start_pos, angles, obs_pos, r, max_distance):

    cos = np.cos(angles)
    sin = np.sin(angles)

    end_pos = start_pos + max_distance * np.stack([cos, sin], axis=-1)  # [num_lasers, 2]

    dxdy = end_pos - start_pos
    fxfy = start_pos - obs_pos

    dx, dy = dxdy[:, 0], dxdy[:, 1]
    fx, fy = fxfy[0], fxfy[1]  # broadcast

    a = dx ** 2 + dy ** 2
    b = 2 * (fx * dx + fy * dy)
    c = (fx ** 2 + fy ** 2) - r ** 2

    discriminant = b ** 2 - 4 * a * c

    distances = np.full_like(angles, max_distance)

    valid_mask = discriminant >= 0
    discriminant_sqrt = np.sqrt(discriminant[valid_mask])

    a_valid = a[valid_mask]
    b_valid = b[valid_mask]

    t1 = (-b_valid - discriminant_sqrt) / (2 * a_valid)
    t2 = (-b_valid + discriminant_sqrt) / (2 * a_valid)

    t1_valid = (0 <= t1) & (t1 <= 1)
    t2_valid = (0 <= t2) & (t2 <= 1)

    distances_valid = np.full_like(t1, max_distance)

    distances_valid[t1_valid] = t1[t1_valid] * max_distance
    distances_valid[~t1_valid & t2_valid] = t2[~t1_valid & t2_valid] * max_distance

    distances[valid_mask] = distances_valid
    return distances  # shape [num_lasers]


def check_obs_intersection(start_pos, angle, obs_pos, r, max_distance):
    cos = np.cos(angle)
    sin = np.sin(angle)

    dir_vec = np.array([cos, sin])
    dxdy = max_distance * dir_vec
    fxfy = start_pos - obs_pos

    dx, dy = dxdy[0], dxdy[1]
    fx, fy = fxfy[0], fxfy[1]

    dist0 = np.linalg.norm(fxfy)
    if dist0 < r - eps:
        return -(r - dist0)

    a = dx ** 2 + dy ** 2
    b = 2 * (fx * dx + fy * dy)
    c = fx ** 2 + fy ** 2 - r ** 2

    discriminant = b ** 2 - 4 * a * c
    distance = 0

    if np.abs(a) < eps or discriminant < 0:
        return distance  # No intersection

    discriminant = np.maximum(discriminant, 0.0)
    sqrt_disc = np.sqrt(discriminant)
    t1 = (-b - sqrt_disc) / (2 * a)
    t2 = (-b + sqrt_disc) / (2 * a)

    # Find the nearest forward intersection point
    t_min = float('inf')
    for t in [t1, t2]:
        if 0.0 <= t <= 1.0:
            t_min = min(t_min, t)

    if t_min < float('inf'):
        distance = t_min * max_distance

    if np.isnan(distance):
        print("NaN detected in collision computation!")
        print(f"t1={t1}, t2={t2}")

    return distance

def compute_collision_positions(pos_1, pos_2, vel_1, vel_2, r_1, r_2, max_timestep, buffer=0.02):
    delta_p = pos_1 - pos_2
    delta_v = vel_1 - vel_2
    r_sum = r_1 + r_2 + buffer

    a = np.dot(delta_v, delta_v)
    b = 2 * np.dot(delta_p, delta_v)
    c = np.dot(delta_p, delta_p) - r_sum ** 2

    discriminant = b ** 2 - 4 * a * c

    if a < 1e-8 or discriminant < 0:
        t = 0.0
    else:
        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2 * a)
        t2 = (-b + sqrt_disc) / (2 * a)
        valid_ts = [t for t in [t1, t2] if 0 <= t <= max_timestep]
        t = min(valid_ts) if valid_ts else 0.0

    p1_collision = pos_1 + vel_1 * t
    p2_collision = pos_2 + vel_2 * t

    sep = p1_collision - p2_collision
    dist = np.linalg.norm(sep)
    r_exact = r_1 + r_2 + buffer

    if r_exact > dist > 1e-6:
        sep_unit = sep / dist
        mid_point = (p1_collision + p2_collision) / 2
        p1_collision = mid_point + sep_unit * (r_exact / 2)
        p2_collision = mid_point - sep_unit * (r_exact / 2)

    # === 异常检测 ===
    if np.isnan(p1_collision).any() or np.isnan(p2_collision).any():
        print("NaN detected in collision computation!")
        print(f"t={t}")
        print(f"pos_1={pos_1}, pos_2={pos_2}, vel_1={vel_1}, vel_2={vel_2}")

    return p1_collision, p2_collision

def check_wall_intersection(start_pos, angles, length, width, max_distance):
    x, y = start_pos[0], start_pos[1]
    cos = np.cos(angles)
    sin = np.sin(angles)
    distances = np.full_like(angles, fill_value=max_distance, dtype=np.float32)

    mask_up = sin > eps
    mask_down = sin < -eps
    mask_right = cos > eps
    mask_left = cos < -eps

    up_dist = (length - y) / (sin + eps)
    down_dist = y / (-sin + eps)
    right_dist = (width - x) / (cos + eps)
    left_dist = x / (-cos + eps)

    distances = np.minimum(distances, np.where(mask_up, up_dist, distances))
    distances = np.minimum(distances, np.where(mask_down, down_dist, distances))
    distances = np.minimum(distances, np.where(mask_right, right_dist, distances))
    distances = np.minimum(distances, np.where(mask_left, left_dist, distances))

    return distances

def select_nearest(ref_pos, p1, p2):
    dist1 = np.linalg.norm(ref_pos - p1)
    dist2 = np.linalg.norm(ref_pos - p2)
    return p1 if dist1 <= dist2 else p2

def calculate_polygon_area(coords: np.ndarray) -> float:
    coords = coords.astype(float)
    centroid = np.mean(coords, axis=0)

    angles = np.arctan2(coords[:, 1] - centroid[1], coords[:, 0] - centroid[0])
    sorted_indices = np.argsort(angles)
    sorted_coords = coords[sorted_indices]

    shifted_coords = np.roll(sorted_coords, shift=1, axis=0)
    area = np.sum((shifted_coords[:, 0] + sorted_coords[:, 0]) * (shifted_coords[:, 1] - sorted_coords[:, 1]))
    return np.abs(area) / 2

def calculate_all_triangle(coords: np.ndarray, target_pos: np.ndarray) -> float:
    coords = coords.astype(float)
    centroid = np.mean(coords, axis=0)

    angles = np.arctan2(coords[:, 1] - centroid[1], coords[:, 0] - centroid[0])
    sorted_indices = np.argsort(angles)
    sorted_coords = coords[sorted_indices]

    S_sum = 0
    for i in range(len(sorted_coords)):
        next_i = (i + 1) % len(sorted_coords)
        S_sum += calculate_polygon_area(np.array([(sorted_coords[i][0], sorted_coords[i][1]),
                                                  (sorted_coords[next_i][0], sorted_coords[next_i][1]),
                                                  (target_pos[0], target_pos[1])]))
    return S_sum


def calculate_angles(points: np.ndarray, center: np.ndarray) -> np.ndarray:

    vectors = points - center

    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    angles = np.remainder(angles, 2 * np.pi)

    sorted_angles = np.sort(angles)

    next_angles = np.roll(sorted_angles, shift=-1, axis=0)
    delta_angles = next_angles - sorted_angles
    delta_angles = np.remainder(delta_angles, 2 * np.pi)

    return delta_angles