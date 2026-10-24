import argparse
import hashlib
import json
from pathlib import Path, PureWindowsPath
import shutil
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from common import RunLock, digest, read, token
from sweep_checkpoint import atomic_bytes


def matching_newlines(data, expected):
    normalized = data.replace(b'\r\n', b'\n')
    for candidate in (data, normalized, normalized.replace(b'\n', b'\r\n')):
        if hashlib.sha256(candidate).hexdigest() == expected:
            return candidate
    raise RuntimeError('Not a newline-only difference; restore the original file instead')


def local_source(value, root):
    parts = PureWindowsPath(value).parts
    anchors = [i for i, part in enumerate(parts) if part.casefold() == 'we_normalized_300maps']
    if len(anchors) != 1 or '..' in parts or any(p.casefold().startswith('ga_logs') for p in parts):
        raise RuntimeError(f'Unapproved source path: {value}')
    target = root.joinpath(*parts[anchors[0]:]).resolve()
    base = (root/'we_normalized_300maps').resolve()
    if (not base.is_relative_to(root.resolve()) or not target.is_relative_to(base)
            or any(p.casefold().startswith('ga_logs') for p in target.parts)):
        raise RuntimeError('Source path escapes the normalized directory')
    return target


def configuration(manifest):
    return {key: value for key, value in manifest.items() if key != 'maps_per_case'}


def plan_repair(root, output):
    root, output = root.resolve(), output.resolve()
    selection = read(output/'selection.json')
    if selection['identity'].get('data_policy') != 'normalized-resume-plan-v4':
        raise RuntimeError('Unsupported study; its identity must not be replaced')
    expected_sources = selection['identity']['environment']['source_sha256']
    actual_sources = {p.name for p in root.glob('*.py') if not p.name.startswith('test_')}
    if actual_sources != set(expected_sources):
        raise RuntimeError(f'Root Python files differ: missing={sorted(set(expected_sources)-actual_sources)}, '
                           f'extra={sorted(actual_sources-set(expected_sources))}')
    repairs = {}
    for name, expected in expected_sources.items():
        file = root/name
        if file.resolve().parent != root:
            raise RuntimeError(f'Unapproved source: {name}')
        data = file.read_bytes()
        try:
            candidate = matching_newlines(data, expected)
        except RuntimeError as error:
            raise RuntimeError(f'{name}: {error}') from error
        if candidate != data:
            repairs[file] = candidate
    for name, expected in selection['identity']['code'].items():
        file = (root/name).resolve()
        if not file.is_relative_to(root) or not file.is_file() or digest(file) != expected:
            raise RuntimeError(f'Implementation changed or missing: {name}')
    we_manifest = output/'we/manifest.json'
    if we_manifest.exists() and read(we_manifest) != dict(study=token(selection['identity']),
                                                       runner=digest(root/'baseline_150maps/run_we.py')):
        raise RuntimeError('WE runner or study identity changed')
    checked, relocated = set(), 0

    def relocate(entry, key):
        nonlocal relocated
        target = local_source(entry[key], root)
        if key == 'manifest' and target not in checked:
            if target.name != 'manifest.json' or not target.is_file():
                raise RuntimeError(f'Missing source manifest: {target}')
            if configuration(read(target)) != configuration(selection['identity']['environment']):
                raise RuntimeError(f'Different source experiment: {target}')
            checked.add(target)
        if str(target) != entry[key]:
            relocated += 1
            entry[key] = str(target)
        return target

    for name in selection['maps']:
        if PureWindowsPath(name).name != name or token(read(output/'inputs'/name)) != selection['inputs'][name]:
            raise RuntimeError(f'Frozen input changed: {name}')
        source = selection['sources'][name]
        result, manifest = relocate(source, 'result'), relocate(source, 'manifest')
        if result != manifest.parent/'runs'/name:
            raise RuntimeError(f'Source map/path mismatch: {name}')
        if source.get('kind') == 'completed' and not result.is_file():
            raise RuntimeError(f'Missing completed normalized source: {result}')
        for checkpoint in source.get('checkpoints', []):
            file, manifest = relocate(checkpoint, 'path'), relocate(checkpoint, 'manifest')
            if file != manifest.parent/'checkpoints'/(Path(name).stem+'.pkl'):
                raise RuntimeError(f'Checkpoint map/path mismatch: {name}')
    return repairs, selection, relocated


def repair(root, output, apply=False):
    root, output = root.resolve(), output.resolve()
    if not output.is_relative_to(root):
        raise RuntimeError('Output must be inside this project')
    with RunLock(output/'.run.lock'):
        repairs, selection, relocated = plan_repair(root, output)
        print(f'Exact original hashes verified. Newline repairs: {len(repairs)}; source paths: {relocated}')
        for file in repairs:
            print(f'  Newlines only: {file.name}')
        if not apply or not (repairs or relocated):
            print('CHECK ONLY; no experiment data changed.' if not apply else 'Already compatible; nothing changed.')
            return None
        if relocated:
            repairs[output/'selection.json'] = json.dumps(selection, indent=2, ensure_ascii=False).encode('utf-8')
        backup = Path(tempfile.mkdtemp(prefix='transfer_backup_', dir=output))
        for file in repairs:
            target = backup/file.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, target)
        for file, data in repairs.items():
            atomic_bytes(file, data)
        remaining, _, paths = plan_repair(root, output)
        if remaining or paths:
            raise RuntimeError(f'Repair incomplete; backups: {backup}')
        print(f'REPAIRED. Backups: {backup}')
        print('Study identity, map inputs, results and checkpoints were preserved.')
        return backup


def main():
    parser = argparse.ArgumentParser(description='Repair newline-only source differences and normalized source paths after moving a study')
    parser.add_argument('--output-dir', type=Path, default=HERE/'results_normalized')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    repair(ROOT, args.output_dir, args.apply)


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, OSError, ValueError, KeyError) as error:
        print(f'STOPPED: {error}', file=sys.stderr)
        sys.exit(1)
