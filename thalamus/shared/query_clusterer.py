# shared/query_clusterer.py
# Cluster example queries into K groups using TF-IDF + K-means (default)
# or sentence-transformer embeddings + K-means (optional, requires sentence-transformers).
# No LLM calls; uses pre-computed example_input texts from the scoring matrices.
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

_SUPPORTED_EMBEDDERS = ("tfidf", "sentence")


class QueryClusterer:
    """Embed and cluster query texts using TF-IDF + K-means (default) or
    sentence-transformer embeddings + K-means (optional).

    Two backends
    ~~~~~~~~~~~~
    ``tfidf`` (default)
        Fast, no extra dependencies.  Vocabulary captured from the training
        corpus; will not generalise to out-of-vocabulary paraphrases.

    ``sentence``
        Requires the ``sentence-transformers`` package.
        Uses a pre-trained model (default: ``all-MiniLM-L6-v2``) to produce
        dense semantic embeddings.  Paraphrase-robust and cross-lingual but
        slower and requires an internet connection on first use.

    Used in two modes:
      fit()      — during config build: fit on all example_input texts from matrices
      predict()  — at query time: assign a new user message to the nearest cluster
    """

    def __init__(
        self,
        n_clusters: int = 20,
        max_features: int = 2000,
        embedder: str = "tfidf",
        sentence_model: str = "all-MiniLM-L6-v2",
    ):
        if embedder not in _SUPPORTED_EMBEDDERS:
            raise ValueError(f"embedder must be one of {_SUPPORTED_EMBEDDERS}, got {embedder!r}")

        self._n_clusters = n_clusters
        self._max_features = max_features
        self._embedder = embedder
        self._sentence_model_name = sentence_model

        # TF-IDF backend (always initialised; ignored when embedder="sentence")
        self._vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self._kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        self._centroids: np.ndarray | None = None  # shape (n_clusters, n_features)
        self._embedding_dim: int | None = None      # set after fit() for sentence backend
        self._is_fitted = False

        # Sentence-transformer model — lazy-loaded at fit/predict time, never pickled
        self._sentence_encoder = None

    # ── fitting ───────────────────────────────────────────────────────────────

    def fit(self, texts: list[str]) -> None:
        """Fit on all example_input texts collected from scoring matrices."""
        if len(texts) < self._n_clusters:
            self._n_clusters = max(1, len(texts))
            self._kmeans = KMeans(
                n_clusters=self._n_clusters, n_init=10, random_state=42
            )

        X = self._embed_batch(texts, fit_vectorizer=True)
        self._kmeans.fit(X)
        self._centroids = self._kmeans.cluster_centers_
        self._is_fitted = True

    # ── inference ─────────────────────────────────────────────────────────────

    def transform(self, text: str) -> np.ndarray:
        """Embed a single text as a dense vector."""
        return self._embed_batch([text])[0]

    def transform_batch(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts into dense vectors."""
        return self._embed_batch(texts)

    def predict(self, text: str) -> int:
        """Return the cluster id closest to this text."""
        vec = self._embed_batch([text])
        return int(self._kmeans.predict(vec)[0])

    def predict_batch(self, texts: list[str]) -> np.ndarray:
        """Return cluster ids for a list of texts."""
        X = self._embed_batch(texts)
        return self._kmeans.predict(X)

    def centroid(self, cluster_id: int) -> np.ndarray:
        """Return the centroid vector for a cluster."""
        if self._centroids is None:
            raise RuntimeError("QueryClusterer.fit() must be called before centroid()")
        return self._centroids[cluster_id]

    @property
    def n_clusters(self) -> int:
        return self._n_clusters

    @property
    def n_features(self) -> int:
        if not self._is_fitted:
            raise RuntimeError("QueryClusterer.fit() must be called before n_features")
        if self._embedder == "sentence":
            if self._embedding_dim is None:
                raise RuntimeError("Embedding dim not set — internal error")
            return self._embedding_dim
        return len(self._vectorizer.vocabulary_)

    @property
    def embedder(self) -> str:
        """Return the active backend: "tfidf" or "sentence"."""
        return self._embedder

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Pickle the fitted model.

        The sentence-transformer model itself is NOT pickled — it is reloaded
        from the HuggingFace hub on first use after loading.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "n_clusters":     self._n_clusters,
                    "max_features":   self._max_features,
                    "embedder":       self._embedder,
                    "sentence_model": self._sentence_model_name,
                    "vectorizer":     self._vectorizer,
                    "kmeans":         self._kmeans,
                    "centroids":      self._centroids,
                    "embedding_dim":  self._embedding_dim,
                },
                f,
            )

    @classmethod
    def load(cls, path: Path) -> "QueryClusterer":
        """Load a previously saved clusterer."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(
            n_clusters=data["n_clusters"],
            max_features=data.get("max_features", 2000),
            embedder=data.get("embedder", "tfidf"),
            sentence_model=data.get("sentence_model", "all-MiniLM-L6-v2"),
        )
        obj._vectorizer    = data["vectorizer"]
        obj._kmeans        = data["kmeans"]
        obj._centroids     = data["centroids"]
        obj._embedding_dim = data.get("embedding_dim")
        obj._is_fitted     = True
        return obj

    # ── private helpers ───────────────────────────────────────────────────────

    def _embed_batch(self, texts: list[str], *, fit_vectorizer: bool = False) -> np.ndarray:
        """Embed a list of texts using the configured backend.

        Parameters
        ----------
        texts:
            Input texts to embed.
        fit_vectorizer:
            When True (only valid during fit()), fits the TF-IDF vectorizer on
            the texts before transforming.  Ignored for the sentence backend.
        """
        if self._embedder == "sentence":
            encoder = self._get_sentence_encoder()
            vecs = encoder.encode(
                texts,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            if self._embedding_dim is None:
                self._embedding_dim = int(vecs.shape[1])
            return vecs

        # TF-IDF backend
        if fit_vectorizer:
            X = self._vectorizer.fit_transform(texts)
        else:
            X = self._vectorizer.transform(texts)
        return X.toarray()

    def _get_sentence_encoder(self):
        """Lazy-load the sentence-transformer model."""
        if self._sentence_encoder is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore[import]
            except ImportError as exc:
                raise ImportError(
                    "The 'sentence-transformers' package is required when using "
                    "embedder='sentence'.  Install it with:\n"
                    "    pip install sentence-transformers"
                ) from exc
            self._sentence_encoder = SentenceTransformer(self._sentence_model_name)
        return self._sentence_encoder
