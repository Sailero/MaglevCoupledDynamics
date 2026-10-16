"""Generate stationary two-point data from one filename under configs/."""
from pathlib import Path
import sys

from sim.dataset import generate_dataset
from sim.model import load_config


def main(argv=None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("Usage: python make_data.py <config_filename.yaml>", file=sys.stderr)
        return 2
    name = args[0]
    if Path(name).name != name or "/" in name or "\\" in name or Path(name).suffix.lower() not in {".yaml", ".yml"}:
        print("Please supply one YAML filename under configs/.", file=sys.stderr)
        return 2
    path = Path(__file__).resolve().parent / "configs" / name
    try:
        cfg = load_config(path)
        generate_dataset(cfg)
    except (OSError, ValueError, AssertionError) as exc:
        print(f"Generation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
