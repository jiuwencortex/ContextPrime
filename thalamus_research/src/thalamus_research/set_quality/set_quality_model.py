"""Phase R4 — Set-level quality model.

Wraps a ``GradientBoostingRegressor`` that predicts *outcome_quality* from
the 14-dimensional feature vector produced by
:func:`~thalamus_research.set_quality.interaction_features.compute_feature_vector`.

Unlike the GA's linear sum of marginal scores, this model captures
non-linear pairwise interactions between components (C7 design choice).

Saved artefact layout (``set_quality_model/`` directory)::

    set_quality_model/
        model.pkl         GradientBoostingRegressor (sklearn joblib)
        meta.json         training metadata (n_records, rmse, r2, feature_dim)

Usage::

    from thalamus_research.set_quality.set_quality_model import SetQualityModel

    # Train
    model = SetQualityModel()
    model.fit(X_train, y_train)
    model.save("/oracle/set_quality_model")

    # Load and predict
    model = SetQualityModel.load("/oracle/set_quality_model")
    pred = model.predict(X)                # np.ndarray shape (n,)
    score = model.score_set(component_names, cluster_id, catalog, extractor)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_MODEL_FILE = "model.pkl"
_META_FILE = "meta.json"

# GBR hyper-parameters (conservative defaults for small datasets)
_GBR_PARAMS: dict = {
    "n_estimators": 200,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "min_samples_leaf": 5,
    "random_state": 42,
}


class SetQualityModel:
    """Gradient-boosting regressor for set-level outcome quality.

    Parameters
    ----------
    gbr_params:
        Keyword arguments forwarded to
        ``sklearn.ensemble.GradientBoostingRegressor``.  Defaults to
        :data:`_GBR_PARAMS`.
    """

    def __init__(self, gbr_params: dict | None = None) -> None:
        from sklearn.ensemble import GradientBoostingRegressor

        params = gbr_params or _GBR_PARAMS
        self._model = GradientBoostingRegressor(**params)
        self._fitted = False
        self._meta: dict = {}

    # ── training ──────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SetQualityModel":
        """Fit the model on *(X, y)* arrays from :class:`OutcomeDataset`.

        Parameters
        ----------
        X:
            Feature matrix of shape ``(n_records, n_features)``.
        y:
            Outcome quality labels of shape ``(n_records,)``.
        """
        if len(X) == 0:
            raise ValueError("Cannot train SetQualityModel on empty dataset.")

        self._model.fit(X, y)
        self._fitted = True

        # Compute in-sample metrics for metadata
        y_pred = self._model.predict(X)
        rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        self._meta = {
            "n_records": int(len(X)),
            "feature_dim": int(X.shape[1]),
            "in_sample_rmse": round(rmse, 4),
            "in_sample_r2": round(r2, 4),
        }
        logger.info(
            "SetQualityModel fitted: n=%d  RMSE=%.4f  R²=%.4f",
            len(X), rmse, r2,
        )
        return self

    # ── inference ─────────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted quality scores, clipped to [0, 1]."""
        self._require_fitted()
        raw = self._model.predict(X)
        return np.clip(raw, 0.0, 1.0)

    def score_set(
        self,
        component_names: list[str],
        cluster_id: int,
        catalog,
        extractor=None,
    ) -> float:
        """Predict quality for a single component set.

        Parameters
        ----------
        component_names:
            Flat list of component names in the candidate set.
        cluster_id:
            Query cluster ID for this context.
        catalog:
            :class:`~thalamus_research.baselines.component_catalog.ComponentCatalog`
        extractor:
            Optional :class:`CoInclusionExtractor`.

        Returns
        -------
        float in [0, 1].
        """
        from .interaction_features import compute_feature_vector

        fvec = compute_feature_vector(component_names, cluster_id, catalog, extractor)
        X = np.array([fvec], dtype=float)
        return float(self.predict(X)[0])

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, model_dir: str | Path) -> None:
        """Serialise model and metadata to *model_dir*."""
        import joblib

        self._require_fitted()
        out = Path(model_dir)
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, out / _MODEL_FILE)
        (out / _META_FILE).write_text(
            json.dumps(self._meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("SetQualityModel saved to %s", out)

    @classmethod
    def load(cls, model_dir: str | Path) -> "SetQualityModel":
        """Load a previously saved model.

        Raises
        ------
        FileNotFoundError
            If *model_dir* or ``model.pkl`` is absent.
        """
        import joblib

        path = Path(model_dir)
        pkl = path / _MODEL_FILE
        if not pkl.exists():
            raise FileNotFoundError(
                f"model.pkl not found in {path}. Train with: thalamus-research set-quality train"
            )
        obj = cls.__new__(cls)
        obj._model = joblib.load(pkl)
        obj._fitted = True
        meta_path = path / _META_FILE
        obj._meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        logger.info("SetQualityModel loaded from %s  meta=%s", path, obj._meta)
        return obj

    @property
    def meta(self) -> dict:
        """Training metadata (n_records, rmse, r2, feature_dim)."""
        return dict(self._meta)

    # ── internals ─────────────────────────────────────────────────────────────

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "SetQualityModel is not fitted.  Call fit() or load() first."
            )
