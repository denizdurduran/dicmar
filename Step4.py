import sys             # SKIPPED

from pathlib import Path
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
import utilities.utils as u
import utilities.constants as ct

if __name__ == "__main__":
    ct.create_parser()
    u.run_optimization()