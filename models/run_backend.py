from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="RA-CFGCache multi-backend runner")
    parser.add_argument(
        "--backend",
        type=str,
        required=True,
        choices=["flux", "wan", "cogvideox"],
        help="Backend to run (flux, wan, or cogvideox).",
    )
    args, backend_args = parser.parse_known_args(argv)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    if args.backend == "flux":
        script = PROJECT_ROOT / "models" / "flux" / "src" / "sample.py"
        cmd = [sys.executable, str(script), *backend_args]
    elif args.backend == "wan":
        cmd = [sys.executable, "-m", "models.wan.sample", *backend_args]
    elif args.backend == "cogvideox":
        cmd = [sys.executable, "-m", "models.cogvideox.sample", *backend_args]
    else:
        raise ValueError(f"Unsupported backend: {args.backend}")

    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT), env=env)


if __name__ == "__main__":
    main()
