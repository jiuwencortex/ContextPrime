# oracle_builder/cli.py
# Entry point: python -m jiuwenswarm.tools.oracle_builder build
from __future__ import annotations

import logging
import sys

from jiuwenswarm.thalamus.oracle_builder.cli_args_parser import make_parser
from jiuwenswarm.thalamus.oracle_builder.evolutionary.cmd_build import cmd_build
from jiuwenswarm.thalamus.oracle_builder.classifier.cmd_train_classifier import cmd_train_classifier



def main() -> None:
    parser = make_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    logging.basicConfig(level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
                        format="%(levelname)s %(message)s", stream=sys.stderr)

    if args.command == "evolve":
        cmd_build(args)
    elif args.command == "train-classifier":
        cmd_train_classifier(args)


if __name__ == "__main__":
    main()
