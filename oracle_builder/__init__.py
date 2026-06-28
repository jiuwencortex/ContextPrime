"""Oracle builder: offline preparation of context selection artifacts.

Two paths:
  - evolutionary/  Evolutionary search over component_scoring matrices →
                   writes context_configs.json + query_clusterer.pkl
  - classifier/    Supervised training from interaction logs →
                   writes classifier.pkl

Quick start:
    # Run evolutionary search to build the oracle:
    python -m jiuwenswarm.thalamus.oracle_builder evolve \
        --oracle-dir ~/.jiuwenswarm/oracle

    # Train the classifier:
    python -m jiuwenswarm.thalamus.oracle_builder train-classifier \
        --oracle-dir ~/.jiuwenswarm/oracle

    # At query time (Phase 3 — cluster-based):
    from jiuwenswarm.thalamus.context_selectors import ClusterSelector
    selector = ClusterSelector.load(oracle_dir)
    config = selector.select(user_query, budget="medium")

    # At query time (Phase 4 — classifier, embedding required):
    from jiuwenswarm.thalamus.context_selectors import ClassifierSelector
    selector = ClassifierSelector.load(oracle_dir)
    config = selector.select(query_embedding)  # query_embedding: np.ndarray
"""
