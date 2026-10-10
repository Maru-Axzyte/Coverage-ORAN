"""Synthetic solver/evidence tests only; no radio MILP simulation."""
import contextlib
import io
import unittest
from unittest.mock import patch

import cvxpy as cp
import numpy as np

from contextual_weight_audit import AuditedContextualWeights
from preference_weights import (
    ContextualPreferencePair, ContextualPredictionPair, fit_history_contextual_weights,
)


CONTEXT = np.array([1., .8, .5, .5, .7])


class TwoStageWeightTests(unittest.TestCase):
    def fit(self, pairs, errors=()):
        prior = np.zeros((3, 5))
        prior[:, 0] = 1./3.
        return fit_history_contextual_weights(
            pairs, average=prior, previous=prior, pair_weights=np.ones(len(pairs)),
            prediction_pairs=errors,
        )

    def test_prediction_error_moves_weights_without_breaking_preferences(self):
        pair = ContextualPreferencePair(np.zeros(3), np.full(3, .2), CONTEXT, .01)
        error = ContextualPredictionPair(np.array([1., 0., 0.]), CONTEXT)
        result = self.fit([pair], [error])
        weights = result.vartheta @ CONTEXT
        self.assertLess(weights[0], 1./3.-.01)
        self.assertLess(result.prediction_mse_after, result.prediction_mse_before)
        self.assertLessEqual(result.mean_pair_loss, result.preference_loss_limit)
        self.assertAlmostEqual(sum(weights), 1., places=6)

    def test_nonzero_primary_optimum_is_protected(self):
        pairs = [ContextualPreferencePair(np.zeros(3), direction, CONTEXT, .8)
                 for direction in (np.array([1., 0., 0.]), np.array([0., 1., 0.]))]
        # Prediction-only learning would select all z and incur hinge .8;
        # the true minimum exact preference hinge is .3.
        errors = [ContextualPredictionPair(10.*p.worse, CONTEXT) for p in pairs]
        result = self.fit(pairs, errors)
        self.assertAlmostEqual(result.stage1_pair_loss, .3, places=6)
        self.assertLessEqual(result.mean_pair_loss, result.preference_loss_limit)
        self.assertLess((result.vartheta @ CONTEXT)[2], 1e-5)

    def test_missing_predictions_are_not_zero_error_evidence(self):
        pair = ContextualPreferencePair(np.zeros(3), np.full(3, .2), CONTEXT, .01)
        result = self.fit([pair])
        self.assertIsNone(result.prediction_mse_before)
        self.assertIsNone(result.prediction_mse_after)
        np.testing.assert_allclose(result.vartheta @ CONTEXT, np.full(3, 1./3.), atol=1e-5)

    def test_simplex_holds_at_every_context_corner(self):
        from itertools import product
        pair = ContextualPreferencePair(np.zeros(3), np.full(3, .2), CONTEXT, .01)
        result = self.fit([pair], [ContextualPredictionPair(np.array([.8, -.2, .1]), CONTEXT)])
        for features in product((0., 1.), repeat=4):
            weights = result.vartheta @ np.array([1., *features])
            self.assertGreaterEqual(min(weights), -1e-7)
            self.assertAlmostEqual(sum(weights), 1., places=6)

    def test_prediction_pairs_use_same_batch_context_and_service(self):
        learner = AuditedContextualWeights(np.ones(3))
        a, b = np.array([.2, .4, .3]), np.array([.4, .2, .3])
        pa, pb = a+np.array([.1, 0., 0.]), b+np.array([0., .2, 0.])
        learner.observe(('a',), a, 40, .8, 3., CONTEXT, pa, (1, 1))
        learner.observe(('b',), b, 40, .8, 5., CONTEXT, pb, (1, 1))
        learner.observe(('other_batch',), b, 40, .8, 0., CONTEXT, pb, (2, 2))
        learner.observe(('other_context',), b, 40, .8, 0., CONTEXT*.9, pb, (1, 1))
        learner.observe(('other_service',), b, 39, .7, 0., CONTEXT, pb, (1, 1))
        learner.observe(('seed',), b, 40, .8, 0., CONTEXT)
        pa[:] = 100.  # Stored forecast is a snapshot, not a mutable alias.
        pairs = learner._collect_prediction_pairs()
        self.assertEqual(len(pairs), 1)
        np.testing.assert_allclose(pairs[0].error_delta, [-.1, .2, 0.])
        self.assertEqual(learner._collect_prediction_pairs(), [])
        self.assertEqual(learner.duplicates, 0)
        learner.observe(('a',), a, 40, .8, 0., CONTEXT, pa, (3, 3))
        self.assertEqual(learner.duplicates, 1)
        self.assertEqual(learner._collect_prediction_pairs(), [])

    def test_new_calibration_evidence_triggers_fit_without_new_preferences(self):
        learner = AuditedContextualWeights(np.ones(3))
        for i in range(3):
            learner.observe((i,), np.full(3, .1+.02*i), 40, .8, .2, CONTEXT)
        learner.update(CONTEXT)
        for i in range(2):
            learner.observe(('new', i), np.full(3, .4), 30, .6, .2, CONTEXT,
                            np.array([.4+.2*i, .4, .4]), (1, 1))
        result = learner.update(CONTEXT)
        self.assertEqual(result['new_pairs'], 0)
        self.assertEqual(result['new_prediction_pairs'], 1)
        self.assertEqual(result['successful_updates'], 2)
        self.assertIsNotNone(result['prequential_prediction_mse'])
        self.assertLessEqual(result['mean_pair_loss'], result['preference_loss_limit'])

    def test_second_stage_solver_failure_is_transactional(self):
        learner = AuditedContextualWeights(np.ones(3))
        for i in range(3):
            learner.observe((i,), np.full(3, .1+.02*i), 40, .8, 0., CONTEXT)
        before = learner.matrix.copy()
        real_solve = cp.Problem.solve
        calls = []

        def fail_after_stage_one(problem, *args, **kwargs):
            calls.append(problem)
            if len(calls) > 1:
                raise cp.error.SolverError('synthetic stage-two failure')
            return real_solve(problem, *args, **kwargs)

        with patch.object(cp.Problem, 'solve', new=fail_after_stage_one):
            result = learner.update(CONTEXT)
        self.assertTrue(result['status'].startswith('fit_failed:'))
        self.assertEqual(learner.successful_updates, 0)
        np.testing.assert_array_equal(learner.matrix, before)
        np.testing.assert_array_equal(learner.average_matrix, before)

    def test_numerically_invalid_preference_cap_is_rejected(self):
        learner = AuditedContextualWeights(np.ones(3))
        for i in range(3):
            learner.observe((i,), np.array([.1+.02*i, .2, .2]), 40, .8, 0., CONTEXT)
        before = learner.matrix.copy()
        real_solve = cp.Problem.solve
        calls = []

        def corrupt_stage_two(problem, *args, **kwargs):
            result = real_solve(problem, *args, **kwargs)
            calls.append(problem)
            if len(calls) > 1:
                # Simplex-feasible, but violates the primary optimum. A solver
                # status alone must not permit committing this matrix.
                for variable in problem.variables():
                    if variable.name() == 'two_stage_contextual_weights':
                        bad = np.zeros((3, 5))
                        bad[2, 0] = 1.
                        variable.value = bad
            return result

        with patch.object(cp.Problem, 'solve', new=corrupt_stage_two):
            result = learner.update(CONTEXT)
        self.assertTrue(result['status'].startswith('fit_failed:'))
        self.assertEqual(learner.successful_updates, 0)
        np.testing.assert_array_equal(learner.matrix, before)

    def test_joint_capture_preserves_pre_milp_forecasts(self):
        from env import UE
        from ga import Bounds, GASettings, Individual
        from optimizer import JointTopKCSAEA

        def fake_oracle(genome):
            p = .2+.1*float(genome[0, 2]-250.)/50.
            choices = [{'u': u, 'ue_id': u, 'num_prbs': 1, 'power_w': p} for u in range(2)]
            return {'served_ue_ids': [0, 1], 'total_power_w': 2*p, 'total_num_prbs': 2,
                    'milp': {'solver_optimal': True, 'power_by_uav_w': [p, p],
                             'uav_mode_choices': choices}}

        ga = JointTopKCSAEA(
            num_uavs=2, ues=[UE(0, 500, 500), UE(1, 550, 500)],
            bounds=Bounds((0., 0., 250., 30.), (1000., 1000., 300., 60.)),
            power_budget_w=4., prb_budget=10., milp_evaluator=fake_oracle,
            settings=GASettings(population_size=3, population_capacity=9, generations=2),
            cma_pre_samples=0,
        )
        seed = Individual(np.array([[500., 500., 250., 40.], [550., 500., 250., 40.]]))
        ga.evaluate_exact(seed)
        ga._joint_commit()
        ga._joint_freeze()
        expected = []
        for height in (270., 290.):
            child = Individual(seed.genome.copy())
            child.genome[:, 2] = height
            ga.evaluate_surrogate(child)
            pred = ga.joint_predictions[ga._joint_key(child)]
            expected.append(np.array(ga._joint_stats(pred['mean'])[:3]))
            ga.evaluate_exact(child)
            ga.evaluate_exact(child)  # Cached exact read must not create evidence.
        ga._joint_commit()
        with contextlib.redirect_stdout(io.StringIO()):
            ga.update_contextual_model('test')
        observations = ga.weight_learner.observations
        self.assertEqual(len(observations), 3)
        self.assertIsNone(observations[0].predicted_resource)
        for observation, forecast in zip(observations[1:], expected):
            np.testing.assert_allclose(observation.predicted_resource, forecast)
            self.assertGreater(np.linalg.norm(observation.resource-forecast), 1e-6)
        self.assertEqual(len(ga.weight_learner.prediction_pairs), 1)


if __name__ == '__main__':
    unittest.main()
