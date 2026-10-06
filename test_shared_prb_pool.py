"""Shared-pool regressions using tiny synthetic MILPs, not a full GA run."""
from types import SimpleNamespace
import unittest

import highspy
import numpy as np

import config
import env
from milp import ServiceMode, UAVMILP
from surrogate import JointSurrogateMixin, _resource_statistics


def synthetic_allocator(allocation):
    """Prescribed <=10-PRB link modes isolate resource constraints from radio."""
    uavs = [env.UAV(i, 500., 500., max_prbs=config.uav_max_prb, max_power_w=2.)
            for i in range(5)]
    links = []
    for u, count in enumerate(allocation):
        while count:
            n = min(count, 10)
            links.append((u, n))
            count -= n
    ues = [env.UE(i, 500., 500., require_mbps=5.) for i in range(len(links))]
    allocator = UAVMILP(uavs, ues, [env.solution(0, 500., 500., 275., 45.)],
        config.PhysConstant(), config.exact_solver_settings(),
        fixed_placement_indices=(0,)*5, orthogonal_prb_pool=True)
    for k, (u, n) in enumerate(links):
        mode = ServiceMode(u, k, 0, 0, n, .1, 5., 10.)
        allocator.service_modes[mode.key] = mode
        allocator.modes_by_uk[u, k].append(mode.key)
        allocator.modes_by_uks[u, k, 0].append(mode.key)
        allocator.distance_2d[k, 0] = 0.
        for j in range(5):
            allocator.big_m[j, k] = 1000.
    return allocator


class SharedPoolTests(unittest.TestCase):
    def test_config_has_no_equal_split(self):
        self.assertEqual(config.total_system_prbs, 135)
        self.assertEqual(config.uav_max_prb, 135)
        self.assertEqual(config.uav_max_power_w, 2.)
        self.assertEqual(config.allocation_max_prbs, 10)

    def test_one_uav_can_use_40(self):
        result = synthetic_allocator([40, 0, 0, 0, 0]).solve()
        self.assertTrue(result['solver_optimal'])
        self.assertEqual(result['num_ues_served'], 4)
        self.assertEqual(result['total_num_prbs'], 40)
        self.assertEqual(sum(x['num_prbs'] for x in result['uav_mode_choices'] if x['u'] == 0), 40)

    def test_shared_135_can_be_unevenly_allocated(self):
        allocation = [40, 30, 25, 20, 20]
        result = synthetic_allocator(allocation).solve()
        self.assertEqual(result['system_prb_budget'], 135)  # not 5 * 135!
        self.assertEqual(result['total_num_prbs'], 135)
        used = [sum(x['num_prbs'] for x in result['uav_mode_choices'] if x['u'] == u)
                for u in range(5)]
        self.assertEqual(used, allocation)
        self.assertTrue(all(p <= 2.+1e-9 for p in result['power_by_uav_w']))

    def test_over_135_is_rejected(self):
        allocator = synthetic_allocator([40, 30, 30, 20, 20])
        model = allocator.build_model()
        model.addConstr(model.qsum(allocator.vars['pi'].values()) == len(allocator.ues))
        model.run()
        self.assertEqual(model.getModelStatus(), highspy.HighsModelStatus.kInfeasible)

    def test_unforced_overload_admits_only_what_fits(self):
        result = synthetic_allocator([40, 30, 30, 20, 20]).solve()
        self.assertTrue(result['solver_optimal'])
        self.assertEqual(result['total_num_prbs'], 130)
        self.assertEqual(result['num_ues_served'], 13)

    def test_surrogate_uses_global_prb_pressure_and_violation(self):
        allocations = np.array([[40, 30, 25, 20, 20], [40, 30, 30, 20, 20]])
        y = np.c_[np.full((2, 5), .2), allocations/135., np.ones((2, 2))]
        p, n, z, cost, v = _resource_statistics(y, np.full(5, 2.), np.full(5, 135.), 135., np.ones(3)/3)
        np.testing.assert_allclose(n, [1., 140/135])
        np.testing.assert_allclose(z, n)
        np.testing.assert_allclose(v, [0., 5/135], atol=1e-12)

    def test_40_has_no_local_violation(self):
        y = np.r_[np.full(5, .2), np.array([40, 0, 0, 0, 0])/135., 1., 1.]
        *_, violation = _resource_statistics(y, np.full(5, 2.), np.full(5, 135.), 135., np.ones(3)/3)
        self.assertEqual(violation, 0.)

    def test_exact_pressure_matches_surrogate(self):
        fake = SimpleNamespace(_uav_power_budgets=np.full(5, 2.), prb_budget=135.,
            _exact_uav_loads=lambda _: (np.zeros(5), np.array([40,30,25,20,20]), np.ones(5)))
        p, n, z = JointSurrogateMixin._exact_bottleneck_pressure(fake, {})
        self.assertEqual((p, n, z), (.5, 1., 1.))

    def test_gradient_matches_shared_pool_cost(self):
        fake = SimpleNamespace(num_uavs=5, _resource_weights=np.array([.3,.4,.3]),
            _uav_power_budgets=np.full(5, 2.), power_budget_w=10.,
            _uav_prb_budgets=np.full(5, 135.), prb_budget=135.)
        for power in ([.2]*5, [.9,.2,.2,.2,.2]):
            y = np.r_[power, np.array([40,10,10,10,10])/135., 1., 1.]
            analytic = JointSurrogateMixin._resource_j_gradient(fake, y)
            def cost(value):
                return _resource_statistics(value, fake._uav_power_budgets,
                    fake._uav_prb_budgets, 135., fake._resource_weights)[3]
            numeric = np.array([(cost(y+np.eye(len(y))[i]*1e-6)-cost(y-np.eye(len(y))[i]*1e-6))/2e-6
                                for i in range(len(y))])
            np.testing.assert_allclose(analytic, numeric, atol=1e-8)


if __name__ == '__main__':
    unittest.main()
