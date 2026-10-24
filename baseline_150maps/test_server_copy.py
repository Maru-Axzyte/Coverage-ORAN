import contextlib
import hashlib
import io
from pathlib import Path, PureWindowsPath
import tempfile
import unittest
from unittest.mock import patch

from common import RunLock, digest, read, save, token
import repair_server_copy as transfer


HERE = Path(__file__).resolve().parent


class ServerCopyTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        output = root/'baseline_150maps/results_normalized'
        source = root/'we_normalized_300maps/results_300maps'
        original = b'num_threads = 8\r\nparallel = True\r\n'
        (root/'config.py').write_bytes(original.replace(b'\r\n', b'\n'))
        runner = root/'baseline_150maps/run_we.py'
        runner.parent.mkdir(parents=True)
        runner.write_bytes(b'pass\n')
        environment = dict(maps_per_case=50, objective='fixed',
                           source_sha256={'config.py': hashlib.sha256(original).hexdigest()})
        identity = dict(data_policy='normalized-resume-plan-v4', environment=environment,
                        code={'config.py': digest(root/'config.py'),
                              'baseline_150maps/run_we.py': digest(runner)})
        name = 'ue_020_map_001.json'
        row = dict(num_ues=20, map_index=1, map_seed=42, algorithm_seed=2026, ues=[])
        old = PureWindowsPath('C:/old/project/we_normalized_300maps/results_300maps')
        selection = dict(identity=identity, maps=[name], inputs={name: token(row)}, sources={name:
            dict(kind='completed', result=str(old/'runs'/name), manifest=str(old/'manifest.json'),
                 checkpoints=[dict(path=str(old/'checkpoints'/'ue_020_map_001.pkl'),
                                   manifest=str(old/'manifest.json'))])})
        save(source/'manifest.json', environment)
        save(source/'runs'/name, dict(row, status='complete'))
        save(output/'selection.json', selection)
        save(output/'inputs'/name, row)
        save(output/'we/manifest.json', dict(study=token(identity), runner=digest(runner)))
        save(output/'we/runs'/name, dict(row, status='complete', study_identity=token(identity)))
        checkpoint = output/'we/checkpoints/ue_020_map_001.pkl'
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b'checkpoint-preserved')
        return root, output, original

    def test_exact_original_bytes_can_be_restored_in_both_directions(self):
        lf, crlf = b'a = 1\nb = 2\n', b'a = 1\r\nb = 2\r\n'
        self.assertEqual(transfer.matching_newlines(lf, hashlib.sha256(crlf).hexdigest()), crlf)
        self.assertEqual(transfer.matching_newlines(crlf, hashlib.sha256(lf).hexdigest()), lf)
        with self.assertRaises(RuntimeError):
            transfer.matching_newlines(b'a = 9\n', hashlib.sha256(lf).hexdigest())

    def test_check_only_does_not_modify_inputs_results_or_sources(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            root, output, _ = self.fixture(directory)
            before = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
            with contextlib.redirect_stdout(io.StringIO()):
                transfer.repair(root, output)
            self.assertEqual({p: p.read_bytes() for p in before}, before)
            self.assertEqual(list(output.glob('transfer_backup_*')), [])

    def test_apply_preserves_identity_results_and_checkpoint_and_is_repeatable(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            root, output, original = self.fixture(directory)
            previous_selection = (output/'selection.json').read_bytes()
            identity = read(output/'selection.json')['identity']
            protected = [p for p in root.rglob('*') if p.is_file()
                         and p not in (root/'config.py', output/'selection.json')]
            before = {p: p.read_bytes() for p in protected}
            with contextlib.redirect_stdout(io.StringIO()):
                backup = transfer.repair(root, output, apply=True)
                self.assertIsNone(transfer.repair(root, output, apply=True))
            self.assertEqual((root/'config.py').read_bytes(), original)
            self.assertEqual((backup/'config.py').read_bytes(), original.replace(b'\r\n', b'\n'))
            self.assertEqual((backup/'baseline_150maps/results_normalized/selection.json').read_bytes(), previous_selection)
            self.assertEqual(read(output/'selection.json')['identity'], identity)
            self.assertEqual({p: p.read_bytes() for p in before}, before)
            self.assertEqual(len(list(output.glob('transfer_backup_*'))), 1)
            repairs, _, paths = transfer.plan_repair(root, output)
            self.assertEqual((repairs, paths), ({}, 0))

    def test_real_code_change_stops_before_any_repair(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            root, output, _ = self.fixture(directory)
            changed = b'num_threads = 16\nparallel = True\n'
            (root/'config.py').write_bytes(changed)
            previous = (output/'selection.json').read_bytes()
            with self.assertRaisesRegex(RuntimeError, 'Not a newline-only difference'):
                transfer.repair(root, output, apply=True)
            self.assertEqual((root/'config.py').read_bytes(), changed)
            self.assertEqual((output/'selection.json').read_bytes(), previous)
            self.assertEqual(list(output.glob('transfer_backup_*')), [])

    def test_changed_frozen_map_stops_before_repair(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            root, output, original = self.fixture(directory)
            save(output/'inputs/ue_020_map_001.json', dict(ues=['changed']))
            with self.assertRaisesRegex(RuntimeError, 'Frozen input changed'):
                transfer.repair(root, output, apply=True)
            self.assertEqual((root/'config.py').read_bytes(), original.replace(b'\r\n', b'\n'))

    def test_changed_source_manifest_stops_before_repair(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            root, output, original = self.fixture(directory)
            path = root/'we_normalized_300maps/results_300maps/manifest.json'
            save(path, dict(read(path), objective='different'))
            with self.assertRaisesRegex(RuntimeError, 'Different source experiment'):
                transfer.repair(root, output, apply=True)
            self.assertEqual((root/'config.py').read_bytes(), original.replace(b'\r\n', b'\n'))

    def test_archive_and_traversal_paths_are_not_relocated(self):
        for value in ('C:/old/GA_logs/we_normalized_300maps/results/manifest.json',
                      'C:/old/we_normalized_300maps/../other/manifest.json',
                      'C:/old/other/results/manifest.json'):
            with self.assertRaises(RuntimeError):
                transfer.local_source(value, HERE)

    def test_link_resolving_to_retired_archive_is_rejected(self):
        original_resolve = Path.resolve
        base = HERE/'we_normalized_300maps'
        archive = HERE/'GA_logs_retired'

        def resolve(path, *args, **kwargs):
            if path == base:
                return archive
            if path == base/'results/manifest.json':
                return archive/'results/manifest.json'
            return original_resolve(path, *args, **kwargs)

        with patch.object(Path, 'resolve', resolve):
            with self.assertRaises(RuntimeError):
                transfer.local_source('C:/old/we_normalized_300maps/results/manifest.json', HERE)

    def test_active_study_is_not_modified(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            root, output, original = self.fixture(directory)
            with RunLock(output/'.run.lock'):
                with self.assertRaises(RuntimeError):
                    transfer.repair(root, output, apply=True)
            self.assertEqual((root/'config.py').read_bytes(), original.replace(b'\r\n', b'\n'))


if __name__ == '__main__':
    unittest.main()
