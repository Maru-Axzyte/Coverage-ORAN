import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from common import METHODS, RunLock, read
from recorded_data import OUTPUT, forbidden_source
from worker import load


def main(argv=None):
    parser = argparse.ArgumentParser(description='Sequential main WE + five baselines, 150 identical maps each')
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    parser.add_argument('--methods', nargs='+', choices=('we', *METHODS), default=['we', *METHODS])
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--summary-only', action='store_true')
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    if forbidden_source(output):
        raise RuntimeError('The retired archive cannot be used as input or output.')
    baseline = load('all_baseline_launcher', 'baseline_150maps/main.py')
    we = load('all_we_launcher', 'baseline_150maps/run_we.py')
    common_args = ['--output-dir', str(output)]
    if args.summary_only:
        with RunLock(output / '.run.lock'):
            selection = we.validate_study(output)
            we.summarize_all(output, selection)
        return 0
    preparation = [*common_args, '--prepare-only']
    code = baseline.main(preparation)
    if code:
        return code
    code = we.main([*common_args, '--prepare-only'])
    if code or args.prepare_only:
        return code
    for method in args.methods:
        print(f'=== Starting method {method}: 150 maps, sequential ===', flush=True)
        code = (we.main(common_args) if method == 'we' else
                baseline.main([*common_args, '--methods', method]))
        with RunLock(output / '.run.lock'):
            we.summarize_all(output, we.validate_study(output))
        if code:
            return code
    print(f'Finished. All six methods: {output / "all_methods_summary.csv"}', flush=True)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (RuntimeError, OSError) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
