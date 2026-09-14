from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .benchmark import final, run_variant, sweep
from .config import VARIANTS, load_config
from .resources import doctor, prepare, prepare_local
from .selftest import run_self_tests


DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "default.toml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_benchmark.py",
        description="Standalone Qwen3-VL hallucination benchmark",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="TOML configuration")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("doctor", help="check the GPU, dependencies, and data")
    subparsers.add_parser("prepare", help="download and lock model and evaluation data")
    subparsers.add_parser(
        "prepare-local",
        help="lock the existing local Qwen2.5-VL checkpoint and AMBER dataset",
    )
    subparsers.add_parser("self-test", help="run method and cache invariant tests")

    run_parser = subparsers.add_parser("run", help="run one method variant")
    run_parser.add_argument("--variant", choices=VARIANTS, required=True)
    run_parser.add_argument("--benchmark", choices=("chair", "amber"), default="chair")
    run_parser.add_argument(
        "--split", choices=("smoke", "dev", "generation"), default="smoke"
    )

    subparsers.add_parser("sweep", help="run sequential coordinate search on CHAIR dev")
    subparsers.add_parser("final", help="run sealed CHAIR test and AMBER generation")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "doctor":
            result = doctor(config)
        elif args.command == "prepare":
            result = prepare(config)
        elif args.command == "prepare-local":
            result = prepare_local(config)
        elif args.command == "self-test":
            result = run_self_tests()
        elif args.command == "run":
            result = run_variant(
                config,
                variant=args.variant,
                benchmark=args.benchmark,
                split=args.split,
            )
        elif args.command == "sweep":
            result = sweep(config)
        elif args.command == "final":
            result = final(config)
        else:  # pragma: no cover
            raise AssertionError(args.command)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if isinstance(result, dict) and result.get("ok") is False:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
