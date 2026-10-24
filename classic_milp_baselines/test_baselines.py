import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('classic_baselines', Path(__file__).with_name('main.py'))
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)


class BaselineTests(unittest.TestCase):
    def test_save_retries_temporary_windows_lock(self):
        original = Path.replace
        attempts = []
        def replace_after_lock(path, target):
            attempts.append(path)
            if len(attempts) < 3:
                raise PermissionError('simulated Windows file lock')
            return original(path, target)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            destination = Path(directory) / 'result.json'
            baseline.save(destination, dict(value=1))
            with patch.object(Path, 'replace', replace_after_lock), patch.object(baseline.time, 'sleep'):
                baseline.save(destination, dict(value=2))
            self.assertEqual(len(attempts), 3)
            self.assertEqual(baseline.read(destination), dict(value=2))

    def test_persistent_lock_preserves_old_and_pending_json(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            destination = Path(directory) / 'result.json'
            baseline.save(destination, dict(value=1))
            with patch.object(Path, 'replace', side_effect=PermissionError('locked')), \
                    patch.object(baseline.time, 'sleep'), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(PermissionError):
                    baseline.save(destination, dict(value=2))
            self.assertEqual(baseline.read(destination), dict(value=1))
            pending = list(Path(directory).glob('*.tmp'))
            self.assertEqual(len(pending), 1)
            self.assertEqual(baseline.read(pending[0]), dict(value=2))

    def test_runner_migration_does_not_allow_settings_changes(self):
        old = dict(runner=next(iter(baseline.IO_COMPATIBLE_RUNNERS)), population=80)
        new = dict(runner='patched', population=80)
        self.assertTrue(baseline.compatible_runner_update(old, new, True))
        self.assertFalse(baseline.compatible_runner_update(old, dict(new, population=40), True))
        self.assertFalse(baseline.compatible_runner_update(dict(old, runner='unknown'), new, True))

    def test_selection_counts_and_reproducibility(self):
        eligible = [(f'ue_{n:03d}_map_{i:03d}.json', dict(num_ues=n, map_index=i))
                    for n in baseline.COUNTS for i in range(1, 51)]
        first = baseline.select_maps(eligible, 20261015)
        self.assertEqual(first, baseline.select_maps(eligible, 20261015))
        self.assertEqual(len(first), 60)
        self.assertEqual(len({p for p, _ in first}), 60)
        for n, count in baseline.COUNTS.items():
            self.assertEqual(sum(row['num_ues'] == n for _, row in first), count)

    def test_only_default_generation_difference_allowed(self):
        import hashlib
        old = b'generations: int = 300\nother = 1\n'
        manifest = dict(ga_settings=dict(generations=300),
                        source_sha256={'ga.py': hashlib.sha256(old).hexdigest()})
        self.assertTrue(baseline.matches_historical('ga.py', old.replace(b'300', b'400'), manifest))
        self.assertFalse(baseline.matches_historical('ga.py', old.replace(b'other = 1', b'other = 2'), manifest))

    def test_real_scoring_both_methods_with_fake_milp(self):
        manifest = baseline.read(baseline.SOURCE / 'manifest.json')
        row = baseline.read(baseline.SOURCE / 'runs' / 'ue_020_map_001.json')
        row['unique_milp_solves'] = 9
        args = SimpleNamespace(population=4, report_every=4)
        for method in ('random', 'ga'):
            def evaluator(genome):
                evaluator.unique_solves += 1
                return dict(total_power_w=0., total_num_prbs=0, served_ue_ids=[],
                            evaluated_genome=genome.tolist(),
                            milp=dict(solver_optimal=True, power_by_uav_w=[0.] * 5,
                                      uav_mode_choices=[]))
            evaluator.unique_solves = 0
            evaluator.solve_seconds = 0.
            with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory, contextlib.redirect_stdout(io.StringIO()):
                path = Path(directory) / 'result.json'
                with patch('evaluation.make_exact_evaluator', return_value=(evaluator, {})):
                    result = baseline.run_map(method, row, manifest, args, path)
                self.assertEqual(result['status'], 'complete')
                self.assertEqual(result['unique_milp_solves'], 9)
                self.assertEqual(len(result['evaluations']), 9)
                self.assertEqual(result['environment'], manifest)
                self.assertEqual(baseline.read(path)['status'], 'complete')
                self.assertTrue(all(h['milp_calls'] <= 9 for h in result['history']))
                self.assertEqual(result['served_ues'], 0)


if __name__ == '__main__':
    unittest.main()
