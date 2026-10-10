"""Configuration and joint-flow regressions; no full radio simulation."""
import contextlib
import ast
import io
from dataclasses import fields, replace
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

import config
from env import UE, UAV
from ga import Bounds, CSAEA, GASettings
from optimizer import JointTopKCSAEA, build_ga
from surrogate import JointSurrogateMixin


def fake_oracle(genome):
    power = .2 + .2*(genome[:, 2]-250.)/50.
    choices = [
        {"u": u, "ue_id": u, "num_prbs": u+1, "power_w": float(power[u])}
        for u in range(2)
    ]
    return {
        "served_ue_ids": [0, 1], "total_power_w": float(power.sum()),
        "total_num_prbs": 3,
        "milp": {"solver_optimal": True, "power_by_uav_w": power.tolist(),
                 "uav_mode_choices": choices},
    }


class SettingsCleanupTests(unittest.TestCase):
    def test_project_has_no_explicit_value_error_raises(self):
        for path in Path(__file__).parent.glob('*.py'):
            if path.name.startswith('test_'):
                continue
            tree = ast.parse(path.read_text(encoding='utf-8-sig'))
            for node in ast.walk(tree):
                if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
                    self.assertFalse(isinstance(node.exc.func, ast.Name)
                                     and node.exc.func.id == 'ValueError', str(path))

    def test_cma_does_not_learn_from_unverified_samples(self):
        from cma import JointCMA
        from ga import Individual
        genome = np.array([[500., 500., 275., 45.]])
        lower = np.array([0., 0., 250., 30.])
        span = np.array([1000., 1000., 50., 30.])
        cma = JointCMA.from_genome(genome, lower, span)
        mean = cma.mean.copy()
        self.assertFalse(cma.update([Individual(genome), Individual(genome.copy())],
                                    lambda item: 0., lower, span))
        np.testing.assert_array_equal(cma.mean, mean)
        self.assertEqual(cma.updates, 0)

    def test_only_active_ga_parameters_are_exposed(self):
        removed = {
            "clusters", "light_violation_epsilon", "light_violation_fraction",
            "bottleneck_target", "idle_uav_redeployment_weight",
            "unserved_infill_variants", "surrogate_service_tie_epsilon",
            "omega_uncertainty", "omega_overlap", "jaccard_weight",
            "radio_similarity_weight", "topology_similarity_weight",
            "transition_risk_kappa", "use_war_elimination",
        }
        self.assertFalse(removed & {f.name for f in fields(GASettings)})
        self.assertEqual(GASettings().min_uav_separation_m, config.uav_min_separation_m)
        self.assertEqual(GASettings().random_seed, config.seed)

    def test_builder_honors_settings_instead_of_hidden_overrides(self):
        defaults = replace(
            GASettings(), sbx_uav_blocks=False, include_separation_constraint=False,
            min_uav_separation_m=17., children_per_pair=7,
        )
        with patch("optimizer.GASettings", return_value=defaults):
            model = build_ga(
                uavs=[UAV(0, 500., 500.), UAV(1, 500., 500.)],
                ues=[UE(0, 500., 500.), UE(1, 550., 500.)],
                phys=config.PhysConstant(), evaluator=fake_oracle,
                population=6, population_capacity=24, generations=6, seed=7,
                weight_method="fixed", cma_pre_samples=0,
            )
        expected = replace(
            defaults, population_size=6, population_capacity=24, generations=6,
            initial_milp_seeds=min(defaults.initial_milp_seeds, 6), random_seed=7,
        )
        self.assertEqual(model.settings, expected)

    def test_single_surrogate_and_resource_score_provider(self):
        for name in ("evaluate_surrogate", "_resource_fitness", "_exact_bottleneck_pressure"):
            self.assertNotIn(name, CSAEA.__dict__)
            self.assertIs(getattr(JointTopKCSAEA, name), getattr(JointSurrogateMixin, name))
        for name in ("_transition_prediction", "_transition_state", "_radio_topology_descriptor"):
            self.assertFalse(hasattr(JointTopKCSAEA, name))

    def test_joint_history_still_populates_without_legacy_transitions(self):
        model = JointTopKCSAEA(
            num_uavs=2, ues=[UE(0, 500., 500.), UE(1, 550., 500.)],
            bounds=Bounds((0., 0., 250., 30.), (1000., 1000., 300., 60.)),
            power_budget_w=4., prb_budget=10., milp_evaluator=fake_oracle,
            weight_method="fixed",
            settings=GASettings(
                population_size=6, population_capacity=24, generations=6,
                initial_milp_seeds=4, update_period=2, children_per_pair=4,
                sbx_uav_blocks=True, include_separation_constraint=True,
                min_uav_separation_m=250., random_seed=7,
            ),
            milp_budget=2, rank_samples=8, cma_pre_samples=2, min_residual_samples=2,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            result = model.run()
        self.assertFalse(hasattr(model, "transitions"))
        self.assertGreater(len(model.joint_records), 0)
        self.assertEqual(result["joint_transition_window_size"], len(model.joint_records))
        self.assertEqual(len(result["joint_topk_history"]), 6)
        self.assertEqual(len(result["exact_incumbent_history"]), 7)
        self.assertTrue(np.isfinite(result["best_genome"]).all())
        self.assertTrue(result["joint_topk_checks"])
        self.assertTrue(result["joint_cma_updates"])


if __name__ == "__main__":
    unittest.main()

