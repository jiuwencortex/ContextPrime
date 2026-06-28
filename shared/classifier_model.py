# shared/classifier_model.py
# ComponentInclusionClassifier: query_embedding → inclusion probability per component.
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


class ComponentInclusionClassifier:
    """A linear classifier mapping query embeddings to component inclusion probabilities.

    Architecture:
        Input:  query_embedding  shape (d_embed,)
        Weight: W                shape (n_components, d_embed)
        Bias:   b                shape (n_components,)
        Output: sigmoid(W @ x + b)  shape (n_components,)  ∈ (0, 1)

    At inference time threshold the output at ``threshold`` (default 0.5)
    to get a binary inclusion mask.

    Training is handled by ComponentClassifierTrainer (uses scikit-learn
    LogisticRegression per component, then extracts weights/biases).
    """

    def __init__(
        self,
        weights: np.ndarray,  # shape (n_components, d_embed)
        biases: np.ndarray,   # shape (n_components,)
        component_names: list[str],
    ):
        if weights.shape[0] != len(component_names):
            raise ValueError(
                f"weights rows ({weights.shape[0]}) must equal n_components ({len(component_names)})"
            )
        if biases.shape[0] != len(component_names):
            raise ValueError(
                f"biases length ({biases.shape[0]}) must equal n_components ({len(component_names)})"
            )
        self._W = weights.astype(np.float32)
        self._b = biases.astype(np.float32)
        self._names = list(component_names)

    # ── inference ─────────────────────────────────────────────────────────────

    def predict_proba(self, query_embedding: np.ndarray) -> np.ndarray:
        """Return inclusion probability for each component.

        Parameters
        ----------
        query_embedding : shape (d_embed,) — must match training embedding dim

        Returns
        -------
        proba : shape (n_components,)  values in (0, 1)
        """
        x = query_embedding.astype(np.float32)
        logits = self._W @ x + self._b
        return _sigmoid(logits)

    def predict(
        self,
        query_embedding: np.ndarray,
        threshold: float = 0.5,
    ) -> dict[str, list[str]]:
        """Return inclusion decision per component, split by type.

        Components whose names contain known type prefixes are grouped:
          - skill_*  or *_skill* → "skills"
          - mem_*                → "memory"
          - tool_* or others    → "tools"

        Returns
        -------
        dict with keys "skills", "memory", "tools" each containing a list of names,
        plus "probabilities" mapping name → float.
        """
        proba = self.predict_proba(query_embedding)
        included = [name for name, p in zip(self._names, proba) if p >= threshold]

        result: dict[str, list[str]] = {"skills": [], "memory": [], "tools": [], "probabilities": {}}
        for name, p in zip(self._names, proba):
            result["probabilities"][name] = float(p)

        for name in included:
            if "skill" in name.lower():
                result["skills"].append(name)
            elif "mem" in name.lower() or "::" in name:
                result["memory"].append(name)
            else:
                result["tools"].append(name)

        return result

    # ── introspection ─────────────────────────────────────────────────────────

    @property
    def n_components(self) -> int:
        return len(self._names)

    @property
    def component_names(self) -> list[str]:
        return list(self._names)

    def top_features(self, component_name: str, n: int = 10) -> list[tuple[int, float]]:
        """Return the top-n most influential embedding dimensions for a component.

        Returns list of (dim_index, weight) sorted by |weight| descending.
        """
        idx = self._names.index(component_name)
        weights = self._W[idx]
        top_idx = np.argsort(np.abs(weights))[::-1][:n]
        return [(int(i), float(weights[i])) for i in top_idx]

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "weights": self._W,
                    "biases": self._b,
                    "component_names": self._names,
                },
                f,
            )

    @classmethod
    def load(cls, path: Path) -> "ComponentInclusionClassifier":
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(
            weights=data["weights"],
            biases=data["biases"],
            component_names=data["component_names"],
        )


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x.clip(-500, 500)))
