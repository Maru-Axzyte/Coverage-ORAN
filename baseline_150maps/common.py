import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
COUNTS = {20: 10, 30: 10, 40: 10, 50: 20, 60: 50, 70: 50}
METHODS = ('random', 'ga', 'pso', 'zhong', 'grid')
CORE = ('config.py', 'env.py', 'milp.py', 'evaluation.py', 'scenario.py',
        'ga.py', 'contextual_ga.py', 'contextual_weight_audit.py',
        'preference_weights.py', 'surrogate.py', 'joint_topk_we.py', 'cma.py', 'optimizer.py')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def token(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def save(path, value):
    from sweep_checkpoint import atomic_bytes
    atomic_bytes(path, json.dumps(value, indent=2, ensure_ascii=False).encode('utf-8'))


def map_identity(row):
    return {k: row[k] for k in ('num_ues', 'map_index', 'map_seed', 'algorithm_seed', 'ues')}


def select_maps(rows, completed, seed=20261023):
    rng = np.random.default_rng(seed)
    selected = []
    for count, requested in COUNTS.items():
        group = sorted((name, row) for name, row in rows.items() if row['num_ues'] == count)
        if len(group) < requested:
            raise RuntimeError(f'{count} UE: need {requested} saved maps, found {len(group)}')
        priorities = {name: int(completed.get(name, 0)) for name, _ in group}
        order = rng.permutation(len(group)).tolist()
        order.sort(key=lambda i: -priorities[group[i][0]])
        selected.extend(group[i][0] for i in order[:requested])
    return sorted(selected, key=lambda name: (rows[name]['num_ues'], rows[name]['map_index']))


class RunLock:
    def __init__(self, path):
        self.path = Path(path)

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open('a+b')
        self.stream.seek(0)
        self.stream.write(b'0')
        self.stream.flush()
        self.stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.stream.close()
            raise RuntimeError(f'This output already has an active runner: {self.path}')
        return self

    def __exit__(self, *_):
        if os.name == 'nt':
            import msvcrt
            self.stream.seek(0)
            msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(self.stream, fcntl.LOCK_UN)
        self.stream.close()


class ExactJournal:
    def __init__(self, path, identity):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)')
        self.db.execute('CREATE TABLE IF NOT EXISTS solves (key TEXT PRIMARY KEY, result TEXT, seconds REAL)')
        previous = self.db.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()
        current = token(identity)
        if previous and previous[0] != current:
            self.db.close()
            raise RuntimeError('Checkpoint belongs to different code, settings or map')
        self.db.execute("INSERT OR IGNORE INTO metadata VALUES ('identity', ?)", (current,))
        self.db.commit()
        self.replayed = 0
        self.new_solves = 0

    def wrap(self, base):
        seen = {}

        def evaluate(genome):
            key = json.dumps(np.round(np.asarray(genome), 7).ravel().tolist())
            if key in seen:
                return seen[key]
            stored = self.db.execute('SELECT result, seconds FROM solves WHERE key=?', (key,)).fetchone()
            if stored:
                result, seconds = json.loads(stored[0]), stored[1]
                self.replayed += 1
            else:
                started = time.perf_counter()
                result = base(genome)
                seconds = time.perf_counter() - started
                if result.get('milp', result).get('solver_optimal', False):
                    self.db.execute('INSERT INTO solves VALUES (?, ?, ?)',
                                    (key, json.dumps(result), seconds))
                    self.db.commit()
                    self.new_solves += 1
            seen[key] = result
            evaluate.unique_solves += 1
            evaluate.solve_seconds += seconds
            return result

        evaluate.unique_solves = 0
        evaluate.solve_seconds = 0.
        return evaluate

    def close(self):
        self.db.close()
