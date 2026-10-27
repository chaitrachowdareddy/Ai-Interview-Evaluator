"""Sentence-embedding semantic similarity with a process-wide model cache.

The model is loaded lazily on the first evaluation. Loading it once per backend
process avoids re-downloading/re-initializing it for every request, and the API
can still start when the model has not been downloaded yet.

Cosine similarity is a geometric similarity between two embeddings, not a
probability that an answer is correct. A high value means the texts are used in
similar contexts; it can be fooled by fluent but factually wrong answers, so it
is only one component of the combined score.
"""

from __future__ import annotations

import logging
import math
import threading
from typing import Any

from sklearn.metrics.pairwise import cosine_similarity

from config import SEMANTIC_MODEL_DEVICE, SEMANTIC_MODEL_NAME
from services.nlp_preprocessing import text_for_embedding

logger = logging.getLogger(__name__)


class SemanticModelError(Exception):
    """A safe, user-facing semantic-model failure with an HTTP status."""

    def __init__(self, message: str, status_code: int = 503) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


_model: Any | None = None
_model_lock = threading.Lock()


def get_semantic_model() -> Any:
    """Return the shared SentenceTransformer instance, loading it if needed."""
    global _model

    if _model is not None:
        return _model

    with _model_lock:
        if _model is not None:
            return _model

        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # noqa: BLE001 - dependency/import failures
            logger.exception("Sentence-transformers is unavailable.")
            raise SemanticModelError(
                "Semantic evaluation is not installed on the backend. Install "
                "the Phase 2 requirements and restart FastAPI."
            ) from exc

        try:
            logger.info(
                "Loading semantic model %s on %s (first use may download it).",
                SEMANTIC_MODEL_NAME,
                SEMANTIC_MODEL_DEVICE,
            )
            _model = SentenceTransformer(
                SEMANTIC_MODEL_NAME,
                device=SEMANTIC_MODEL_DEVICE,
            )
        except Exception as exc:  # noqa: BLE001 - network/cache/model errors
            logger.exception("Could not load semantic model.")
            raise SemanticModelError(
                "The semantic similarity model could not be loaded. The first "
                "use may require internet access to download "
                f"{SEMANTIC_MODEL_NAME}. Check the connection/model name and "
                "try again."
            ) from exc

    return _model


def model_is_loaded() -> bool:
    return _model is not None


def reset_semantic_model() -> None:
    """Drop the cached model (used by tests and troubleshooting)."""
    global _model

    with _model_lock:
        _model = None


def semantic_similarity(
    candidate_answer: str | None,
    reference_answer: str | None,
) -> float:
    """Return cosine similarity mapped to 0-100.

    Negative cosine values are clamped to 0 because a negative similarity is
    not evidence of relevant overlap. The result is rounded to two decimals.
    """
    candidate = text_for_embedding(candidate_answer)
    reference = text_for_embedding(reference_answer)

    if not candidate or not reference:
        return 0.0

    model = get_semantic_model()

    try:
        embeddings = model.encode(
            [reference, candidate],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        similarity = float(
            cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        )
    except Exception as exc:  # noqa: BLE001 - never leak model internals
        logger.exception("Semantic similarity calculation failed.")
        raise SemanticModelError(
            "The semantic model could not evaluate this answer. Please try "
            "again."
        ) from exc

    if not math.isfinite(similarity):
        raise SemanticModelError(
            "The semantic model returned an invalid similarity value. Please "
            "try again."
        )

    clamped = max(0.0, min(1.0, similarity))
    return round(clamped * 100.0, 2)
