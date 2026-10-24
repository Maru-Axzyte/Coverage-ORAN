import contextlib
import importlib.util
import io
import json
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest

spec = importlib.util.spec_from_file_location('extend_300', Path(__file__).with_name('run_all_300.py'))
extension = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extension)


class ExtensionTests(unittest.TestCase):
    def manifests(self):
        common = dict(ue_counts=[20, 30], base_map_seed=2026, algorithm_seed=2026,
                      runner_settings=dict(generations=300), source_sha256=dict(milp='same'))
        return dict(common, maps_per_case={'20': 1, '30': 1}), dict(common, maps_per_case=50)

    def test_reuses_maps_without_overwriting_and_respects_plan(self):
        old, new = self.manifests()
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory, \
                contextlib.redirect_stdout(io.StringIO()):
            source, target = Path(directory)/'source', Path(directory)/'target'
            extension.sweep.save_json(source/'manifest.json', old)
            row = dict(status='complete', num_ues=20, map_index=1,
                       map_seed=extension.sweep.scenario_seed(2026, 20, 1), algorithm_seed=2026,
                       served_ues=20, solver_optimal=True, final_violation=0.)
            result = source/'runs/ue_020_map_001.json'
            extension.sweep.save_json(result, row)
            before = result.read_bytes()
            extension.prepare_extension(source, target, new)
            self.assertEqual(before, result.read_bytes())
            self.assertEqual(before, (target/'runs/ue_020_map_001.json').read_bytes())
            extension.prepare_extension(source, target, new)
            summary = extension.read(target/'summary.json')
            self.assertEqual(summary[0]['required_maps'], 50)
            self.assertEqual(summary[0]['completed_maps'], 1)
            self.assertIsNone(summary[0]['mean_served_ues'])
            self.assertEqual(extension.read(source/'manifest.json'), old)

    def test_rejects_algorithm_change(self):
        old, new = self.manifests()
        new['runner_settings'] = dict(generations=400)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            source, target = Path(directory)/'source', Path(directory)/'target'
            extension.sweep.save_json(source/'manifest.json', old)
            with self.assertRaises(RuntimeError):
                extension.prepare_extension(source, target, new)
            self.assertFalse(target.exists())

    def test_checkpoint_allows_only_verified_experiment_alias(self):
        expected = dict(experiment='new', count=60, map_index=16, map_seed=99)
        payload = dict(version=1, identity=dict(expected, experiment='old'),
                       optimizer_state=dict(marker=7), cache={'cached': 1},
                       unique_milp_solves=680, milp_solve_seconds=12.,
                       elapsed_seconds=50., run_state=dict(generation=150))
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory)/'checkpoint.pkl'
            extension.sweep.atomic_bytes(path, pickle.dumps(payload))
            optimizer, evaluator, cache = SimpleNamespace(), SimpleNamespace(), {}
            state, elapsed = extension.restore_extended_checkpoint(path, expected,
                optimizer, cache, evaluator, 'old')
            self.assertEqual((state['generation'], elapsed, evaluator.unique_solves), (150, 50., 680))
            self.assertEqual(cache, {'cached': 1})
            with self.assertRaises(RuntimeError):
                extension.restore_extended_checkpoint(path, dict(expected, map_index=17),
                    optimizer, cache, evaluator, 'old')


if __name__ == '__main__':
    unittest.main()
