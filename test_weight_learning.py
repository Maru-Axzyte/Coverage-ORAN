"""Small weight-learning regressions; no physical MILP or full experiment runs."""
import contextlib
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from contextual_weight_audit import AuditedContextualWeights


CONTEXT = np.array([1., .8, .5, .5, .7])


def add_chain(learner, violations, start=0):
    for i, v in enumerate(violations, start):
        learner.observe((i,), np.full(3, .1+.02*i), 40, .8, v, CONTEXT)


def unchanged_fit(pairs, **kwargs):
    return SimpleNamespace(vartheta=kwargs['previous'].copy(), status='optimal', mean_pair_loss=0.)


class WeightLearningTests(unittest.TestCase):
    def setUp(self):
        self.learner = AuditedContextualWeights(np.ones(3))

    def test_nonzero_violation_does_not_block_learning(self):
        add_chain(self.learner, [.2013]*3)
        with patch('contextual_weight_audit.fit_history_contextual_weights', side_effect=unchanged_fit) as fit:
            result = self.learner.update(CONTEXT)
            self.assertEqual(result['unique_exact_observations'], 3)
            self.assertEqual(result['training_pairs'], 3)
            self.assertEqual(result['successful_updates'], 1)
            self.assertFalse(result['violation_filter'])
            fit.assert_called_once()

    def test_violation_does_not_change_resource_labels(self):
        other = AuditedContextualWeights(np.ones(3))
        add_chain(self.learner, [0., 0., 0., 0.])
        add_chain(other, [.2013, 1., 10., 100.])
        with patch('contextual_weight_audit.fit_history_contextual_weights', side_effect=unchanged_fit) as fit:
            first = self.learner.update(CONTEXT)
            second = other.update(CONTEXT)
            self.assertEqual(first['training_pairs'], 6)
            self.assertEqual(second['training_pairs'], 6)
            self.assertEqual(fit.call_count, 2)
        for a, b in zip(self.learner.pairs, other.pairs):
            np.testing.assert_array_equal(a.better, b.better)
            np.testing.assert_array_equal(a.worse, b.worse)
            self.assertEqual(a.margin, b.margin)

    def test_no_pairs_keeps_last_matrix_not_fake_learning(self):
        add_chain(self.learner, [.2]*2)
        with patch('contextual_weight_audit.fit_history_contextual_weights', side_effect=unchanged_fit) as fit:
            before = self.learner.matrix.copy()
            result = self.learner.update(CONTEXT)
            self.assertEqual(result['training_pairs'], 1)
            self.assertEqual(result['status'], 'insufficient_exact_pairs')
            fit.assert_not_called()
            np.testing.assert_array_equal(before, self.learner.matrix)

    def test_duplicate_observations_and_updates_do_not_multiply_evidence(self):
        add_chain(self.learner, [.2]*3)
        add_chain(self.learner, [.2]*3)
        with patch('contextual_weight_audit.fit_history_contextual_weights', side_effect=unchanged_fit) as fit:
            self.learner.update(CONTEXT)
            result = self.learner.update(CONTEXT)
            self.assertEqual(result['duplicate_observations_ignored'], 3)
            self.assertEqual(result['training_pairs'], 3)
            self.assertEqual(result['status'], 'no_new_pairs')
            self.assertEqual(fit.call_count, 1)

    def test_other_pair_conditions_unchanged(self):
        rows = [([.2, .4, .2], 40, .8), ([.4, .2, .2], 40, .8),
                ([.1, .1, .1], 39, .8), ([.1, .1, .1], 40, .7)]
        for i, (resource, served, service) in enumerate(rows):
            self.learner.observe((i,), resource, served, service, .2, CONTEXT)
        self.assertEqual(self.learner.update(CONTEXT)['training_pairs'], 0)

    def test_context_importance_for_valid_inputs(self):
        add_chain(self.learner, [.2]*3)
        pair = SimpleNamespace(context=CONTEXT.copy())
        np.testing.assert_allclose(self.learner.context_importance([pair], CONTEXT), [1.])
        opposite = CONTEXT.copy()
        opposite[1:] = 1.-opposite[1:]
        expected = 1.-np.mean((opposite[1:]-CONTEXT[1:])**2)
        np.testing.assert_allclose(self.learner.context_importance([pair], opposite), [expected])
        self.assertFalse(self.learner._paired_observations)

    def test_failed_fit_keeps_previous_weights_without_raising(self):
        for status, matrix in (
            ('infeasible', np.zeros((3, 5))),
            ('optimal', np.zeros((2, 5))),
            ('optimal', np.full((3, 5), np.nan)),
            ('optimal', np.zeros((3, 5))),
        ):
            with self.subTest(status=status, shape=matrix.shape):
                learner = AuditedContextualWeights(np.ones(3))
                add_chain(learner, [.2]*3)
                before = learner.matrix.copy()
                fake = SimpleNamespace(status=status, vartheta=matrix, mean_pair_loss=0.)
                with patch('contextual_weight_audit.fit_history_contextual_weights', return_value=fake):
                    result = learner.update(CONTEXT)
                np.testing.assert_array_equal(learner.matrix, before)
                self.assertEqual(result['successful_updates'], 0)
                self.assertEqual(result['status'], 'fit_failed: invalid_solver_result')

    def test_real_cvxpy_small_weight_fit(self):
        add_chain(self.learner, [.2013]*3)
        result = self.learner.update(CONTEXT)
        self.assertIn(result['status'], ('optimal', 'optimal_inaccurate'))
        weights = self.learner.matrix @ CONTEXT
        self.assertAlmostEqual(float(weights.sum()), 1., places=6)
        self.assertGreaterEqual(float(weights.min()), -1e-6)

    def test_joint_flow_learner_ignores_epsilon_but_comparator_keeps_v(self):
        from env import UE
        from ga import Bounds, GASettings, Individual
        from optimizer import JointTopKCSAEA

        def fake_oracle(genome):
            power = .2+.1*float(genome[0, 2]-250.)/50.
            choices = [{'u': u, 'ue_id': u, 'num_prbs': 1, 'power_w': power} for u in range(2)]
            return {'served_ue_ids': [0, 1], 'total_power_w': 2*power,
                    'total_num_prbs': 2, 'milp': {'solver_optimal': True,
                    'power_by_uav_w': [power]*2, 'uav_mode_choices': choices}}

        ga = JointTopKCSAEA(num_uavs=2, ues=[UE(0, 500, 500), UE(1, 550, 500)],
            bounds=Bounds((0., 0., 250., 30.), (1000., 1000., 300., 60.)),
            power_budget_w=4., prb_budget=10., milp_evaluator=fake_oracle,
            settings=GASettings(population_size=3, population_capacity=9,
                generations=2, initial_milp_seeds=3, update_period=1,
                include_separation_constraint=True, min_uav_separation_m=250.),
            milp_budget=1, rank_samples=2, cma_pre_samples=0, min_residual_samples=2)
        with contextlib.redirect_stdout(io.StringIO()), patch(
                'contextual_weight_audit.fit_history_contextual_weights', side_effect=unchanged_fit):
            result = ga.run()
        first = result['weight_learning_history'][0]
        self.assertFalse(first['violation_filter'])
        self.assertEqual(first['unique_exact_observations'], 3)
        pair_count = len(ga.weight_learner.pairs)
        ga.current_eps = 0.
        again = ga.weight_learner.update(CONTEXT)
        self.assertEqual(again['training_pairs'], pair_count)
        recorded = {tuple(r.genome.ravel()): r.violation for r in ga.archive}
        for item in ga.weight_learner.observations:
            self.assertEqual(item.violation, recorded[item.key])
        # Learning tolerance never relaxes the reported incumbent comparator.
        feasible = Individual(np.zeros((2, 4)))
        infeasible = Individual(np.zeros((2, 4)))
        feasible.exact_result = infeasible.exact_result = {}
        feasible.violation, infeasible.violation = 0., .2
        feasible.served_fraction, infeasible.served_fraction = .2, 1.
        ga.current_eps = .25
        self.assertTrue(ga._better_exact_incumbent(feasible, infeasible))


if __name__ == '__main__':
    unittest.main()
