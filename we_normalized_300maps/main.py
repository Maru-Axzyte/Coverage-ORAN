import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from run_ue_count_sweep import main


if __name__ == '__main__':
    main(['--output-dir', str(HERE / 'results_85maps'), '--no-show',
          '--maps-per-count', '5', '5', '5', '10', '30', '30', *sys.argv[1:]])
