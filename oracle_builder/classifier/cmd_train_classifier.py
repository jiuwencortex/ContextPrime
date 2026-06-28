from __future__ import annotations

import sys
import argparse

from .component_classifier_trainer import ComponentClassifierTrainer


def cmd_train_classifier(args: argparse.Namespace) -> None:
    if not args.oracle_dir.exists():
        print(f"ERROR: --oracle-dir does not exist: {args.oracle_dir}", file=sys.stderr)
        sys.exit(1)

    log_dir = args.log_dir or (args.oracle_dir / "online_logs")
    trainer = ComponentClassifierTrainer(log_dir=log_dir, min_turns=args.min_turns, max_weeks=args.max_weeks, C=args.C)
    classifier_path = args.oracle_dir / "classifier.pkl"
    success = trainer.train_and_save(classifier_path)

    if success:
        print(f"Classifier saved to {classifier_path}")
    else:
        print(f"Not enough turns to train (need {args.min_turns}). Collect more interaction data and retry.",
              file=sys.stderr,)
        sys.exit(1)
