from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from common import COUNTS, save, token
import recorded_data as data


def fixture_row(count, index):
    return dict(status='complete', solver_optimal=True, num_ues=count, map_index=index,
                map_seed=count*1000+index, algorithm_seed=2026,
                ues=[dict(id=i, x_m=float(i), y_m=float(i+1), require_mbps=5.) for i in range(count)])


class RecordedDataTests(unittest.TestCase):
    def test_archive_path_is_rejected_cross_platform(self):
        self.assertTrue(data.forbidden_source('C:\\old\\GA_logs-20261013T002029Z-1-001\\runs'))
        self.assertTrue(data.forbidden_source('/old/GA_logs-20261013T002029Z-1-001/runs'))

    def test_only_recorded_rows_are_selected_and_duplicates_are_not_counted(self):
        manifest = dict(maps_per_case=50, milp_objective='current')
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            root = Path(directory)
            source = root/'we_normalized_300maps/results_300maps'
            duplicate = root/'we_normalized_300maps/results_85maps'
            save(source/'manifest.json', manifest)
            save(duplicate/'manifest.json', dict(manifest, maps_per_case={str(k): v for k, v in COUNTS.items()}))
            for count, required in COUNTS.items():
                for index in range(1, required+1):
                    name = f'ue_{count:03d}_map_{index:03d}.json'
                    save(source/'runs'/name, fixture_row(count, index))
            name = 'ue_020_map_001.json'
            save(duplicate/'runs'/name, fixture_row(20, 1))
            def no_generator(*args, **kwargs):
                raise AssertionError('A recorded-data study must never generate scenarios')
            original_open = Path.open
            def guarded_open(path, *args, **kwargs):
                self.assertFalse(data.forbidden_source(path), str(path))
                return original_open(path, *args, **kwargs)
            with patch.object(data, 'ROOT', root), patch('scenario.configured_scenario', no_generator), \
                    patch.object(Path, 'open', guarded_open):
                rows, provenance, report = data.selected_recorded_maps(manifest, 20261023)
            self.assertEqual(len(rows), 150)
            self.assertEqual(sum(g['available'] for g in report['groups']), 150)
            self.assertTrue(all(row['input_origin'] == data.DATA_POLICY for row in rows.values()))
            self.assertTrue(all(str(source) in p['result'] for p in provenance.values()))
            for row in rows.values():
                self.assertFalse({'served_ues', 'elapsed_seconds', 'unique_milp_solves'} & row.keys())

    def test_missing_maps_are_selected_from_same_plan_without_fake_results(self):
        manifest = dict(maps_per_case=50, milp_objective='current', base_map_seed=2026,
                        algorithm_seed=2026, ue_distribution='uniform')
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            root = Path(directory)
            source = root/'we_normalized_300maps/results_300maps'
            save(source/'manifest.json', manifest)
            save(source/'runs/ue_020_map_001.json', fixture_row(20, 1))
            from types import SimpleNamespace
            import env
            calls = []
            def generator(**kwargs):
                calls.append(kwargs)
                return None, SimpleNamespace(ues=tuple(env.UE(i, float(i), float(i), 5.)
                                                       for i in range(kwargs['num_ues'])))
            with patch.object(data, 'ROOT', root), patch('scenario.configured_scenario', generator):
                rows, sources, report = data.selected_recorded_maps(manifest, 20261023)
                again, _, _ = data.selected_recorded_maps(manifest, 20261023)
            self.assertEqual(len(rows), 150)
            self.assertEqual(rows, again)
            self.assertEqual(sources['ue_020_map_001.json']['kind'], 'completed')
            self.assertEqual(sum(p['kind'] == 'pending' for p in sources.values()), 149)
            self.assertEqual(sum(g['selected_new'] for g in report['groups']), 149)
            self.assertEqual(len(calls), 298)
            for row in rows.values():
                self.assertFalse({'served_ues', 'elapsed_seconds', 'unique_milp_solves'} & row.keys())

    def test_saved_selection_is_not_redrawn_after_more_source_maps_finish(self):
        from worker import load
        from types import SimpleNamespace
        import env
        launcher = load('resume_plan_test', 'baseline_150maps/main.py')
        manifest = dict(maps_per_case=50, milp_objective='current', base_map_seed=2026,
            algorithm_seed=2026, ue_distribution='uniform', ga_settings=dict(initial_milp_seeds=80),
            runner_settings=dict(population=400, generations=300, joint_milp_budget=4))
        def generator(**kwargs):
            return None, SimpleNamespace(ues=tuple(env.UE(i, float(i), float(i), 5.) for i in range(kwargs['num_ues'])))
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            root, output = Path(directory), Path(directory)/'output'
            source = root/'we_normalized_300maps/results_300maps'
            save(source/'manifest.json', manifest)
            save(source/'runs/ue_020_map_001.json', fixture_row(20, 1))
            with patch.object(data, 'ROOT', root), patch('scenario.configured_scenario', generator), \
                    patch('run_ue_count_sweep.experiment_manifest', return_value=manifest):
                first = launcher.prepare(output, 20261023)
                pending = next(n for n in first['maps'] if first['sources'][n]['kind'] == 'pending')
                from common import read
                saved_row = read(output/'inputs'/pending)
                save(source/'runs'/pending, dict(saved_row, status='complete', solver_optimal=True))
                with patch('scenario.configured_scenario', side_effect=AssertionError('Selection must not change')):
                    resumed = launcher.prepare(output, 20261023)
            self.assertEqual(token(first), token(resumed))

    def test_wrong_manifest_and_conflicting_ues_are_rejected(self):
        manifest = dict(maps_per_case=50, milp_objective='current')
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            root = Path(directory)
            a = root/'we_normalized_300maps/results_300maps'
            b = root/'we_normalized_300maps/results_85maps'
            save(a/'manifest.json', manifest)
            save(b/'manifest.json', dict(manifest, milp_objective='old'))
            save(a/'runs/ue_020_map_001.json', fixture_row(20, 1))
            changed = fixture_row(20, 1)
            changed['ues'][0]['x_m'] = 999.
            save(b/'runs/ue_020_map_001.json', changed)
            with patch.object(data, 'ROOT', root):
                rows, _, report = data.recorded_maps(manifest)
                self.assertEqual(len(rows), 1)
                self.assertEqual(len(report['rejected_manifests']), 1)
                save(b/'manifest.json', manifest)
                with self.assertRaisesRegex(RuntimeError, 'Conflicting recorded map data'):
                    data.recorded_maps(manifest)

    def test_budget_and_timing_never_use_archive_values(self):
        manifest = dict(ga_settings=dict(initial_milp_seeds=80),
                        runner_settings=dict(population=400, generations=300, joint_milp_budget=4))
        selection = dict(identity=dict(data_policy=data.DATA_POLICY, execution_policy=data.execution_policy(manifest)))
        self.assertEqual(data.allocation_input(selection, 'ga', dict(unique_milp_solves=9999))['unique_milp_solves'], 1280)
        from run_we import we_identity
        row = fixture_row(20, 1)
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            output = Path(directory)
            with self.assertRaises(RuntimeError):
                data.current_grid_budget(output, selection, row)
            save(output/'we/runs/ue_020_map_001.json', dict(row, elapsed_seconds=123.,
                 we_runner_identity=token(we_identity(selection))))
            self.assertEqual(data.current_grid_budget(output, selection, dict(row, elapsed_seconds=9999)), 123.)


if __name__ == '__main__':
    unittest.main()
