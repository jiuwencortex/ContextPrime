"""Phase R3a — Cross-path knowledge transfer.

Research goal: transfer the classifier's learned component co-inclusion patterns
(from Path B) back into the GA fitness function (Path A), so Path A improves
without requiring new real data.

Planned implementation
----------------------
- Extract co-inclusion signal from ``classifier_current.pkl`` weight matrix W
- Augment GA fitness: add a co-inclusion reward term for component pairs with
  high covariance under the classifier
- CLI flag: ``thalamus-oracle evolve --use-classifier-prior``

Prerequisite: Phase R1 complete (baselines established, evaluation harness ready).
"""
