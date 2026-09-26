import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Run the offline ultrasound robot demo.")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    for relative in ("checkpoint/model.ckpt", "InputState/PreviousState.pkl"):
        if not (root / relative).is_file():
            parser.error(f"Missing {relative}; follow README.md to obtain the demo files.")
    (root / "Outputaction").mkdir(exist_ok=True)
    subprocess.run(
        [sys.executable, str(root / "testdemo.py"),
         "--checkpoint", "checkpoint/model.ckpt",
         "--classification_checkpoint", "", "--output_dir", "testoutput",
         "--device", args.device],
        cwd=root, check=True,
    )


if __name__ == "__main__":
    main()
