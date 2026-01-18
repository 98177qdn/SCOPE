import numpy as np
import cvxpy as cp

def solve_joint_cbf(
        positions, velocities, u_raw_list,
        r_hunter, r_target,
        obs_positions, obs_velocities, obs_radii,
        target_positions, target_velocities,
        boundary_low, boundary_high,
        u_max,
        lambda1=3.0, lambda2=2.0, buffer=0.03,
        delta=0.0   # <-- 新增：tightening margin
):
    positions = np.asarray(positions, dtype=np.float64)
    velocities = np.asarray(velocities, dtype=np.float64)
    n, dim = positions.shape

    obs_positions = np.asarray(obs_positions, dtype=np.float64) if len(obs_positions) > 0 else np.zeros((0, dim))
    obs_velocities = np.asarray(obs_velocities, dtype=np.float64) if len(obs_velocities) > 0 else np.zeros((0, dim))
    obs_radii = np.asarray(obs_radii, dtype=np.float64) if len(obs_radii) > 0 else np.zeros((0,))

    target_positions = np.asarray(target_positions, dtype=np.float64) if len(target_positions) > 0 else np.zeros((0, dim))
    target_velocities = np.asarray(target_velocities, dtype=np.float64) if len(target_velocities) > 0 else np.zeros((0, dim))

    boundary_low = np.asarray(boundary_low, dtype=np.float64)
    boundary_high = np.asarray(boundary_high, dtype=np.float64)

    u_raw = np.concatenate([np.asarray(u, dtype=np.float64).reshape(-1) for u in u_raw_list])

    # hunter-hunter safe distance
    d_hh_safe = 2.0 * r_hunter + buffer

    alpha_hh = 3.0
    d2_hh_neighbor = (alpha_hh * d_hh_safe) ** 2

    # hunter-target safe distance
    d_ht_safe = r_hunter + r_target + buffer
    alpha_ht = 3.0
    d2_ht_neighbor = (alpha_ht * d_ht_safe) ** 2

    u = cp.Variable(n * dim)
    constraints = []
    objective = cp.Minimize(cp.sum_squares(u - u_raw))

    # ----------- safety constraints -----------
    for i in range(n):
        pi = positions[i]
        vi = velocities[i]
        ui = u[i * dim:(i + 1) * dim]

        # 1) hunter-hunter constraints
        for j in range(i + 1, n):
            pj = positions[j]
            vj = velocities[j]
            uj = u[j * dim:(j + 1) * dim]

            dij = pi - pj
            vij = vi - vj
            dist2 = float(np.dot(dij, dij))
            if dist2 < 1e-12:
                continue

            if dist2 > d2_hh_neighbor:
                continue

            h = dist2 - d_hh_safe ** 2
            dh = 2.0 * float(np.dot(dij, vij))
            vij2 = float(np.dot(vij, vij))

            ddh = -2.0 * vij2 + 2.0 * cp.sum(cp.multiply(dij, ui - uj))

            constraints.append(ddh + 2.0 * lambda1 * dh + lambda2 * h >= delta)

        # 2) hunter-target constraints
        for k in range(len(target_positions)):
            pt = target_positions[k]
            vt = target_velocities[k]

            dij = pi - pt
            vij = vi - vt
            dist2 = float(np.dot(dij, dij))
            if dist2 < 1e-12:
                continue
            if dist2 > d2_ht_neighbor:
                continue

            h = dist2 - d_ht_safe ** 2
            dh = 2.0 * float(np.dot(dij, vij))
            vij2 = float(np.dot(vij, vij))
            ddh = -2.0 * vij2 + 2.0 * cp.sum(cp.multiply(dij, ui))

            constraints.append(ddh + 2.0 * lambda1 * dh + lambda2 * h >= delta)

        # 3) hunter-obstacle constraints
        for k in range(len(obs_positions)):
            po = obs_positions[k]
            vo = obs_velocities[k]
            ro = float(obs_radii[k])

            dij = pi - po
            vij = vi - vo
            dist2 = float(np.dot(dij, dij))
            if dist2 < 1e-12:
                continue

            d_ho_safe = r_hunter + ro + buffer

            alpha_ho = 3.0
            d2_ho_neighbor = (alpha_ho * d_ho_safe) ** 2
            if dist2 > d2_ho_neighbor:
                continue

            h = dist2 - d_ho_safe ** 2
            dh = 2.0 * float(np.dot(dij, vij))
            vij2 = float(np.dot(vij, vij))
            ddh = -2.0 * vij2 + 2.0 * cp.sum(cp.multiply(dij, ui))

            constraints.append(ddh + 2.0 * lambda1 * dh + lambda2 * h >= delta)

        # 4) hunter-boundary constraints
        for d in range(dim):

            h_high = (boundary_high[d] - pi[d]) - buffer
            dh_high = -vi[d]
            ddh_high = -ui[d]
            constraints.append(ddh_high + 2.0 * lambda1 * dh_high + lambda2 * h_high >= delta)

            h_low = (pi[d] - boundary_low[d]) - buffer
            dh_low = vi[d]
            ddh_low = ui[d]
            constraints.append(ddh_low + 2.0 * lambda1 * dh_low + lambda2 * h_low >= delta)

    # ----------- QP solver -----------
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(
            solver=cp.OSQP,
            warm_start=True,
            eps_abs=1e-3,
            eps_rel=1e-3,
            max_iter=4000,
            verbose=False
        )
        if u.value is not None and not np.any(np.isnan(u.value)):
            safe_actions = [
                np.array(u.value[i * dim:(i + 1) * dim]).reshape(-1)
                for i in range(n)
            ]
            normalized_actions = []
            for a in safe_actions:
                norm_a = np.linalg.norm(a)
                if norm_a > u_max:
                    a = a / norm_a * u_max
                normalized_actions.append(a)
            return True, normalized_actions
        else:
            return False, u_raw_list
    except Exception:
        return False, u_raw_list


def check_future_cbf_feasibility(
    positions, velocities,
    u_current_list,
    r_hunter, r_target,
    obs_positions, obs_velocities, obs_radii,
    target_positions, target_velocities,
    boundary_low, boundary_high,
    u_max, dt,
    lambda1=3.0, lambda2=2.0,
    buffer=0.03,
    delta=0.0
):
    # rollout
    next_positions = []
    next_velocities = []

    for pos, vel, u in zip(positions, velocities, u_current_list):
        acc = np.clip(u, -u_max, u_max)
        v_next = vel + acc * dt
        p_next = pos + v_next * dt
        next_velocities.append(v_next)
        next_positions.append(p_next)

    next_positions = np.array(next_positions)
    next_velocities = np.array(next_velocities)

    # === the most conservative action ===
    u_next_list = []
    for v in next_velocities:
        speed = np.linalg.norm(v)
        if speed < 1e-6:
            u_brake = np.zeros_like(v)
        else:
            u_brake = -v / speed * v / dt
            if np.linalg.norm(u_brake) < 1e-6:
                u_brake = -v / speed * u_max
        u_next_list.append(u_brake)

    Flag, _ = solve_joint_cbf(
        next_positions, next_velocities, u_next_list,
        r_hunter, r_target,
        obs_positions, obs_velocities, obs_radii,
        target_positions, target_velocities,
        boundary_low, boundary_high,
        u_max=u_max, lambda1=lambda1, lambda2=lambda2,
        buffer=buffer,
        delta=delta
    )
    return Flag


def compute_cbf_violation_per_agent(
    positions, velocities, u_raw_list,
    r_hunter, r_target,
    obs_positions, obs_velocities, obs_radii,
    target_positions, target_velocities,
    boundary_low, boundary_high,
    lambda1=3.0, lambda2=2.0, buffer=0.03,
    delta=0.0
):
    n = len(positions)
    dim = len(positions[0])
    violations = np.zeros(n, dtype=np.float64)

    positions = np.asarray(positions, dtype=np.float64)
    velocities = np.asarray(velocities, dtype=np.float64)
    u_raw = [np.asarray(u, dtype=np.float64) for u in u_raw_list]

    obs_positions = np.asarray(obs_positions, dtype=np.float64) if len(obs_positions) > 0 else np.zeros((0, dim))
    obs_velocities = np.asarray(obs_velocities, dtype=np.float64) if len(obs_velocities) > 0 else np.zeros((0, dim))
    obs_radii = np.asarray(obs_radii, dtype=np.float64) if len(obs_radii) > 0 else np.zeros((0,))

    target_positions = np.asarray(target_positions, dtype=np.float64) if len(target_positions) > 0 else np.zeros((0, dim))
    target_velocities = np.asarray(target_velocities, dtype=np.float64) if len(target_velocities) > 0 else np.zeros((0, dim))

    boundary_low = np.asarray(boundary_low, dtype=np.float64)
    boundary_high = np.asarray(boundary_high, dtype=np.float64)

    for i in range(n):
        pi = positions[i]
        vi = velocities[i]
        ui = u_raw[i]

        g_min = np.inf

        # 1) hunter-hunter constraints
        for j in range(n):
            if i == j:
                continue
            pj = positions[j]
            vj = velocities[j]
            uj = u_raw[j]

            dij = pi - pj
            vij = vi - vj
            norm_dij = np.linalg.norm(dij)
            if norm_dij < 1e-6:
                continue

            h = norm_dij**2 - (2 * r_hunter + buffer)**2
            dh = 2.0 * np.dot(dij, vij)
            ddh = -2.0 * np.linalg.norm(vij)**2 + 2.0 * np.dot(dij, ui - uj)
            g = ddh + 2.0 * lambda1 * dh + lambda2 * h
            g_min = min(g_min, g)

        # 2) hunter-target constraints
        for k in range(len(target_positions)):
            pt = target_positions[k]
            vt = target_velocities[k]

            dij = pi - pt
            vij = vi - vt
            norm_dij = np.linalg.norm(dij)
            if norm_dij < 1e-6:
                continue

            h = norm_dij**2 - (r_hunter + r_target + buffer)**2
            dh = 2.0 * np.dot(dij, vij)
            ddh = -2.0 * np.linalg.norm(vij)**2 + 2.0 * np.dot(dij, ui)
            g = ddh + 2.0 * lambda1 * dh + lambda2 * h
            g_min = min(g_min, g)

        # 3) hunter-obstacle constraints
        for k in range(len(obs_positions)):
            po = obs_positions[k]
            vo = obs_velocities[k]
            ro = obs_radii[k]

            dij = pi - po
            vij = vi - vo
            norm_dij = np.linalg.norm(dij)
            if norm_dij < 1e-6:
                continue

            h = norm_dij**2 - (r_hunter + ro + buffer)**2
            dh = 2.0 * np.dot(dij, vij)
            ddh = -2.0 * np.linalg.norm(vij)**2 + 2.0 * np.dot(dij, ui)
            g = ddh + 2.0 * lambda1 * dh + lambda2 * h
            g_min = min(g_min, g)

        # 4) hunter-boundary constraints
        for d in range(dim):

            h_high = (boundary_high[d] - pi[d]) - buffer
            dh_high = -vi[d]
            ddh_high = -ui[d]
            g_high = ddh_high + 2.0 * lambda1 * dh_high + lambda2 * h_high
            g_min = min(g_min, g_high)

            h_low = (pi[d] - boundary_low[d]) - buffer
            dh_low = vi[d]
            ddh_low = ui[d]
            g_low = ddh_low + 2.0 * lambda1 * dh_low + lambda2 * h_low
            g_min = min(g_min, g_low)

        if g_min == np.inf:
            violations[i] = 0.0
        else:
            violations[i] = max(0.0, float(delta - g_min))

    return violations
