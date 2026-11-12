from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from continual_training import method_main


if __name__ == "__main__":
    method_main("adapter")