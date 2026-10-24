"""Trusted local sweep checkpoints and atomic saves with Windows-lock retries."""
import os
from pathlib import Path
import pickle
import time
import uuid


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'{path.name}.{uuid.uuid4().hex}.tmp')
    with temporary.open('wb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    for attempt in range(8):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 7:
                print(f'Cannot replace locked file {path}; new data retained at {temporary}', flush=True)
                raise
            time.sleep(min(.1 * 2**attempt, 2.))


def save_checkpoint(path, identity, optimizer, run_state, cache, evaluator, elapsed):
    payload = dict(version=1, identity=identity, run_state=run_state,
        optimizer_state={k: v for k, v in vars(optimizer).items() if k != 'milp_evaluator'},
        cache=cache, unique_milp_solves=evaluator.unique_solves,
        milp_solve_seconds=evaluator.solve_seconds, elapsed_seconds=elapsed)
    atomic_bytes(path, pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))


def restore_checkpoint(path, identity, optimizer, cache, evaluator):
    # Only open this experiment's own checkpoints, never an untrusted pickle.
    with Path(path).open('rb') as stream:
        payload = pickle.load(stream)
    if payload.get('version') != 1 or payload.get('identity') != identity:
        raise RuntimeError(f'Checkpoint does not match this map/experiment: {path}')
    optimizer.__dict__.update(payload['optimizer_state'])
    cache.update(payload['cache'])
    evaluator.unique_solves = payload['unique_milp_solves']
    evaluator.solve_seconds = payload['milp_solve_seconds']
    return payload['run_state'], payload['elapsed_seconds']
