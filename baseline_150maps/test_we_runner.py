import contextlib
from dataclasses import asdict, dataclass
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from common import read, save, token
import run_we


class MainWEStudyTests(unittest.TestCase):
    def test_import_only_selected_compatible_completed_maps(self):
        row = dict(status='complete', num_ues=20, map_index=37, map_seed=12, algorithm_seed=17,
                   ues=[dict(id=0, x_m=500., y_m=500., require_mbps=5.)], solver_optimal=True,
                   served_ues=1, served_ue_ids=[0], final_violation=0., total_power_w=.1,
                   total_prbs=1, elapsed_seconds=2.)
        name = 'ue_020_map_037.json'
        environment = dict(maps_per_case=50, algorithm_seed=17, test_physics='same')
        selection = dict(identity=dict(environment=environment), maps=[name], sources={})
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            root = Path(directory)
            source = root/'we_normalized_300maps/results_300maps'
            output = root/'study'
            save(source/'manifest.json', environment)
            save(source/'runs'/name, row)
            selection['sources'][name] = dict(result=str(source/'runs'/name), manifest=str(source/'manifest.json'))
            save(source/'runs/ue_020_map_001.json', dict(row, map_index=1))
            save(output/'inputs'/name, row)
            original = (source/'runs'/name).read_bytes()
            with patch.object(run_we, 'ROOT', root):
                report = run_we.prepare_we(output, selection)
                run_we.prepare_we(output, selection)
            self.assertEqual(len(report['reused']), 1)
            self.assertTrue((output/'we/runs'/name).exists())
            self.assertFalse((output/'we/runs/ue_020_map_001.json').exists())
            self.assertEqual((source/'runs'/name).read_bytes(), original)

    def test_different_environment_is_not_imported(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            root = Path(directory)
            source = root/'we_normalized_300maps/results_300maps'
            name = 'ue_020_map_037.json'
            save(source/'manifest.json', dict(maps_per_case=50, objective='old'))
            save(source/'runs'/name, dict(status='complete'))
            output = root/'study'
            save(output/'inputs'/name, {})
            selection = dict(identity=dict(environment=dict(maps_per_case=10, objective='new')), maps=[name],
                sources={name: dict(result=str(source/'runs'/name), manifest=str(source/'manifest.json'))})
            with patch.object(run_we, 'ROOT', root):
                report = run_we.prepare_we(output, selection)
            self.assertFalse((output/'we/runs'/name).exists())
            self.assertEqual(len(report['not_reused']), 1)

    def test_checkpoint_alias_accepts_same_map_only(self):
        from test_sweep_checkpoint import optimizer_and_evaluator
        from sweep_checkpoint import save_checkpoint
        model, evaluator, cache = optimizer_and_evaluator()
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            path = Path(directory)/'map.pkl'
            old = dict(experiment='source', count=20, map_index=37, map_seed=12)
            save_checkpoint(path, old, model, dict(generation=2), cache, evaluator, 3.)
            save(path.with_suffix('.origin.json'), dict(experiments=['source']))
            current = dict(old, experiment='target')
            state, elapsed = run_we.restore_checkpoint(path, current, model, {}, evaluator)
            self.assertEqual(state, dict(generation=2))
            self.assertEqual(elapsed, 3.)
            with self.assertRaises(RuntimeError):
                run_we.restore_checkpoint(path, dict(current, map_index=38), model, {}, evaluator)

    def test_pending_maps_are_left_for_real_we_execution(self):
        row = dict(num_ues=20, map_index=37, map_seed=12, algorithm_seed=17,
                   ues=[dict(id=0, x_m=500., y_m=500., require_mbps=5.)])
        name = 'ue_020_map_037.json'
        environment = dict(maps_per_case=50, algorithm_seed=17, test_physics='same')
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            root = Path(directory)
            source, output = root/'we_normalized_300maps/results_300maps', root/'study'
            save(source/'manifest.json', environment)
            save(output/'inputs'/name, row)
            selection = dict(identity=dict(environment=environment), maps=[name], sources={name:
                dict(kind='pending', result=str(source/'runs'/name), manifest=str(source/'manifest.json'))})
            with patch.object(run_we, 'ROOT', root):
                report = run_we.prepare_we(output, selection)
            self.assertEqual(report['pending'], [name])
            self.assertEqual(report['not_reused'], [])
            self.assertFalse((output/'we/runs'/name).exists())

    def test_pending_checkpoint_is_copied_and_restored_without_changing_source(self):
        from test_sweep_checkpoint import optimizer_and_evaluator
        from sweep_checkpoint import save_checkpoint
        from run_ue_count_sweep import scenario_seed
        model, evaluator, cache = optimizer_and_evaluator()
        environment = dict(maps_per_case=50, algorithm_seed=17, base_map_seed=2026)
        old_environment = dict(environment, maps_per_case=30)
        name = 'ue_070_map_001.json'
        seed = scenario_seed(2026, 70, 1)
        row = dict(num_ues=70, map_index=1, map_seed=seed, algorithm_seed=17, ues=[])
        selection = dict(identity=dict(environment=environment), maps=[name])
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            root = Path(directory)
            source, output = root/'we_normalized_300maps/results_300maps', root/'study'
            save(source/'manifest.json', environment)
            save(source/'extension.json', dict(source_manifest=old_environment,
                target_experiment=run_we.native_experiment(environment),
                source_experiment=run_we.native_experiment(old_environment)))
            checkpoint = source/'checkpoints/ue_070_map_001.pkl'
            old = dict(experiment=run_we.native_experiment(old_environment),
                       count=70, map_index=1, map_seed=seed)
            save_checkpoint(checkpoint, old, model, dict(generation=20), cache, evaluator, 4.)
            original = checkpoint.read_bytes()
            provenance = dict(checkpoints=[dict(path=str(checkpoint), manifest=str(source/'manifest.json'))])
            with patch.object(run_we, 'ROOT', root):
                copied = run_we.copy_native_checkpoint(output, selection, name, row, provenance)
                self.assertEqual(copied['map'], name)
                self.assertIsNone(run_we.copy_native_checkpoint(output, selection, name, row, provenance))
                current = dict(old, experiment=token(run_we.we_identity(selection)))
                state, elapsed = run_we.restore_checkpoint(output/'we/checkpoints'/checkpoint.name,
                                                          current, model, {}, evaluator)
            self.assertEqual(state, dict(generation=20))
            self.assertEqual(elapsed, 4.)
            self.assertEqual(checkpoint.read_bytes(), original)

    def test_worker_freezes_exact_selected_ues_and_uses_native_checkpoint(self):
        import config
        import env
        import run_ue_count_sweep as sweep
        name = 'ue_020_map_037.json'
        row = dict(num_ues=20, map_index=37, map_seed=12, algorithm_seed=17,
                   ues=[asdict(env.UE(0, 111., 222., 5.))])
        selection = dict(identity=dict(environment={}), maps=[name])
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            output = Path(directory)
            save(output/'inputs'/name, row)
            save(output/'we/manifest.json', run_we.we_identity(selection))
            @dataclass(frozen=True)
            class FakeScene:
                ues: tuple
            fake_scene = FakeScene(ues=(env.UE(0, 999., 999., 15.),))
            def fake_run(count, index, seed, defaults, log, **kwargs):
                _, scene = sweep.configured_scenario(seed=seed)
                self.assertEqual([asdict(ue) for ue in scene.ues], row['ues'])
                self.assertEqual(index, 37)
                self.assertEqual(defaults.seed, 17)
                self.assertEqual(kwargs['checkpoint_every'], 10)
                self.assertEqual(kwargs['checkpoint_identity']['map_index'], 37)
                return dict(row, status='complete', solver_optimal=True)
            with patch.object(run_we, 'validate_study', return_value=selection), \
                    patch.object(sweep, 'configured_scenario', lambda **_: (config.PhysConstant(), fake_scene)), \
                    patch.object(sweep, 'run_one', fake_run):
                self.assertEqual(run_we.run_one(output, name), 0)
            self.assertEqual(read(output/'we/runs'/name)['ues'], row['ues'])


if __name__ == '__main__':
    unittest.main()
