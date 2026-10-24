import math
import time
from collections import defaultdict

import highspy
import numpy as np
from scipy.sparse import coo_matrix


def conflict_sets(placements, separation):
    if separation <= 0:
        return
    groups = defaultdict(list)
    for s, p in enumerate(placements):
        groups[(p.x_u, p.y_u)].append(s)
    groups = list(groups.items())
    for a, (xy_a, ids_a) in enumerate(groups):
        heights_a = [placements[s].h_u for s in ids_a]
        compact_a = max(heights_a) - min(heights_a) < separation - 1e-9
        for b in range(a, len(groups)):
            xy_b, ids_b = groups[b]
            horizontal = math.dist(xy_a, xy_b)
            if horizontal >= separation - 1e-9:
                continue
            heights_b = [placements[s].h_u for s in ids_b]
            largest_dh = max(abs(max(heights_a) - min(heights_b)),
                             abs(max(heights_b) - min(heights_a)))
            compact_b = max(heights_b) - min(heights_b) < separation - 1e-9
            if compact_a and compact_b and math.hypot(horizontal, largest_dh) < separation - 1e-9:
                yield ids_a if a == b else ids_a + ids_b
            else:
                for s in ids_a:
                    for t in ids_b:
                        if a == b and s >= t:
                            continue
                        if math.hypot(horizontal, placements[s].h_u - placements[t].h_u) < separation - 1e-9:
                            yield [s, t]


def objective_weights(ues, system_prbs, total_power):
    demands = [max(1, round(ue.require_mbps * 1e6)) for ue in ues]
    divisor = math.gcd(*demands)
    units = np.array([value // divisor for value in demands], dtype=float)
    demand_weight = 2 * system_prbs + 2
    count_weight = demand_weight * (int(units.sum()) + 1)
    return units, count_weight, demand_weight


def build_model(placements, modes, ues, num_uavs, power_limit, per_uav_prbs,
                system_prbs, separation, settings, log_path=None):
    started = time.perf_counter()
    count = len(placements)
    modes = list(modes)
    variables = count + len(modes)
    units, count_weight, demand_weight = objective_weights(ues, system_prbs, num_uavs * power_limit)
    costs = np.zeros(variables)
    by_user, by_site, by_link = defaultdict(list), defaultdict(list), defaultdict(list)
    for i, mode in enumerate(modes, count):
        by_user[mode.k].append(i)
        by_site[mode.placement].append((i, mode))
        by_link[(mode.placement, mode.k)].append(i)
        costs[i] = (count_weight + demand_weight * units[mode.k] - 2 * mode.num_prbs
                    - mode.power_w / (num_uavs * power_limit))
    row_indices, col_indices, values, lower, upper = [], [], [], [], []

    def add(indices, coefficients, lo=-np.inf, hi=np.inf):
        r = len(lower)
        indices = list(indices)
        row_indices.extend([r] * len(indices))
        col_indices.extend(indices)
        values.extend(coefficients)
        lower.append(lo)
        upper.append(hi)

    add(range(count), [1.] * count, num_uavs, num_uavs)
    for indices in by_user.values():
        add(indices, [1.] * len(indices), hi=1.)
    for (s, _), indices in by_link.items():
        add([s, *indices], [-1., *([1.] * len(indices))], hi=0.)
    for s, entries in by_site.items():
        indices = [i for i, _ in entries]
        add([s, *indices], [-power_limit, *[m.power_w for _, m in entries]], hi=0.)
        add([s, *indices], [-per_uav_prbs, *[m.num_prbs for _, m in entries]], hi=0.)
    add(range(count, variables), [m.num_prbs for m in modes], hi=system_prbs)
    conflicts = 0
    for ids in conflict_sets(placements, separation):
        add(ids, [1.] * len(ids), hi=1.)
        conflicts += 1
    matrix = coo_matrix((values, (row_indices, col_indices)), shape=(len(lower), variables)).tocsc()
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = variables, len(lower)
    lp.col_cost_, lp.col_lower_, lp.col_upper_ = costs, np.zeros(variables), np.ones(variables)
    lp.row_lower_, lp.row_upper_ = np.array(lower), np.array(upper)
    lp.integrality_ = [highspy.HighsVarType.kInteger] * variables
    lp.sense_ = highspy.ObjSense.kMaximize
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = matrix.indptr.astype(np.int32)
    lp.a_matrix_.index_ = matrix.indices.astype(np.int32)
    lp.a_matrix_.value_ = matrix.data
    model = highspy.Highs()
    for option, value in [('mip_rel_gap', 0.), ('threads', settings.num_threads),
                          ('parallel', 'on' if settings.parallel_search else 'off'),
                          ('output_flag', settings.log_to_console)]:
        model.setOptionValue(option, value)
    if log_path is not None:
        model.setOptionValue('log_file', str(log_path))
    if model.passModel(lp) == highspy.HighsStatus.kError:
        return None
    initial_sites = []
    for s, p in enumerate(placements):
        if all(math.dist((p.x_u, p.y_u, p.h_u),
                         (placements[t].x_u, placements[t].y_u, placements[t].h_u))
               >= separation - 1e-9 for t in initial_sites):
            initial_sites.append(s)
        if len(initial_sites) == num_uavs:
            start = highspy.HighsSolution()
            start.col_value = np.array([1. if i in initial_sites else 0. for i in range(variables)])
            start.value_valid = True
            model.setSolution(start)
            break
    return model, modes, dict(variables=variables, constraints=len(lower), nonzeros=matrix.nnz,
                              conflict_constraints=conflicts, count_weight=count_weight,
                              demand_weight=demand_weight, demand_units=units.tolist(),
                              build_seconds=time.perf_counter() - started)


def extract(model, placements, modes, num_uavs, ues, power_limit, per_uav_prbs,
            system_prbs, separation, phys):
    import env

    solution = model.getSolution()
    if not solution.value_valid:
        return None
    values = np.asarray(solution.col_value)
    if np.max(np.abs(values - np.rint(values)), initial=0.) > 1e-5:
        return None
    selected = np.flatnonzero(values[:len(placements)] > .5).tolist()
    if len(selected) != num_uavs:
        return None
    selected_set = set(selected)
    chosen = [m for i, m in enumerate(modes, len(placements)) if values[i] > .5]
    by_site = {s: i for i, s in enumerate(selected)}
    power, prbs, served, records = [0.] * num_uavs, [0] * num_uavs, set(), []
    for m in chosen:
        if m.placement not in selected_set or m.k in served:
            return None
        p, ue = placements[m.placement], ues[m.k]
        rate, snr = env.data_rate_mbps(m.power_w, m.num_prbs, env.gain(p, ue, phys), phys)
        if (env.dis_2d(p.x_u, p.y_u, ue.x_m, ue.y_m) > p.r_u + 1e-8
                or rate + 1e-7 < ue.require_mbps
                or m.power_w > phys.allocation_max_power_w + 1e-8
                or m.num_prbs > phys.allocation_max_prbs):
            return None
        u = by_site[m.placement]
        served.add(m.k)
        power[u] += m.power_w
        prbs[u] += m.num_prbs
        records.append(dict(uav=u, ue_id=ue.id, placement=m.placement, power_w=m.power_w,
                            num_prbs=m.num_prbs, rate_mbps=rate, snr_linear=snr))
    genome = [[placements[s].x_u, placements[s].y_u, placements[s].h_u, placements[s].theta_u]
              for s in selected]
    distances = [math.dist(a[:3], b[:3]) for i, a in enumerate(genome) for b in genome[i + 1:]]
    violation = sum(max(0., p / power_limit - 1.) for p in power)
    violation += max(0., sum(prbs) / system_prbs - 1.)
    violation += sum(max(0., n / per_uav_prbs - 1.) for n in prbs)
    if separation > 0:
        violation += sum(max(0., (separation - distance) / separation) for distance in distances)
    if violation > 1e-7:
        return None
    return dict(feasible=True, violation=violation, served_ues=len(served),
                served_ue_ids=sorted(ues[k].id for k in served),
                admitted_demand_mbps=sum(ues[k].require_mbps for k in served),
                total_power_w=sum(power), total_prbs=sum(prbs), power_by_uav_w=power,
                prbs_by_uav=prbs, min_uav_distance_m=min(distances, default=None),
                selected_placement_indices=selected, genome=genome, modes=records)
