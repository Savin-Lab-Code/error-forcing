"""
Usage

python make_disbatch.py, need the disbatch installed

this will create disbatch_script, then you can use for example:

sbatch -J rnn_cpu -p ccn -n 200 -C rome --time=24:00:00 -o .../error_forcing/code/out.out -e .../error_forcing/code/err.err disBatch -p .../error_forcing/code/dislogs disbatch_script
"""

import itertools, shlex
from pathlib import Path
import sys, os, textwrap

ROOT = Path(__file__).resolve().parent
TRAIN = ROOT / "train_single.py"
DSCRIPT = ROOT / "disbatch_script"

GRID = {                      # ← your sweep dictionary
    "B":         [0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0],
    "batch_size":[128],
    "cell":      ["RNN"],
    "diff":      [1.0],
    "g":         [1.5],
    "gtf":       [False],
    "init_method":["neuro"],
    "lr":        [0.0003],
    "perturb":   [0],
    "seed":      [2447, 7175, 74, 1971, 7204, 2634, 6314, 5303, 9416, 7761, 7052, 9489, 7599, 4822, 2232, 3073, 2803, 557, 1561, 747],
    "task":      ["MDXOR"],
}

def flags(cfg: dict) -> str:
    parts = []
    for k, v in cfg.items():
        flag = f"--{k}"
        if isinstance(v, bool):
            if v: parts.append(flag)
        else:
            parts.extend([flag, shlex.quote(str(v))])
    return " ".join(parts)

def main():
    keys = list(GRID)
    lines = []
    for combo in itertools.product(*(GRID[k] for k in keys)):
        cfg = dict(zip(keys, combo))
        line = f"./run_one.sh {flags(cfg)}"
        lines.append(line)

    DSCRIPT.write_text("\n".join(lines))
    os.chmod(DSCRIPT, 0o755)
    print(f"Wrote {len(lines)} cmd lines to {DSCRIPT}")

if __name__ == "__main__":
    main()