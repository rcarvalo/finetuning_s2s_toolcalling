"""Comparing a predicted tool-call argument to the expected one, and saying why not.

Free functions, no state: this is the arithmetic layer under both
``ToolCallDiagnosis`` and ``toolcalling.calls_match``. It lives in its own module
so the diagnosis can use it without importing the scorer that will import the
diagnosis.

``diff_arguments`` returns the mismatches rather than a boolean. Matching is then
"no mismatch", and the reasons a call was rejected survive into the report — the
similarity that fell just under the threshold used to be computed and dropped on
the next line.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from lfm2_audio.core.lazy_component import LazyComponent

ArgMatch = str  # "exact" | "token_f1" | "semantic"

MISMATCH_REASONS: tuple[str, ...] = (
    "missing_key",
    "extra_key",
    "positional_argument",
    "below_threshold",
    "value_differs",
)

POSITIONAL_PREFIX = "_positional_"
"""``parse_tool_call_block`` names positional arguments this way. No expected
schema uses such a key, so a plain comparison reads as a wrong value when the
real defect is a format violation — hence its own mismatch reason."""


@dataclass(frozen=True, slots=True)
class ArgumentMismatch:
    """One argument that kept a predicted call from matching an expected one.

    Lives here rather than in its own file: it is the return type of
    ``diff_arguments`` and has no meaning without it.
    """

    call_index: int
    key: str
    reason: str
    expected: Any = None
    predicted: Any = None
    similarity: float | None = None
    threshold: float | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"call_index": self.call_index, "key": self.key, "reason": self.reason}
        if self.expected is not None:
            payload["expected"] = self.expected
        if self.predicted is not None:
            payload["predicted"] = self.predicted
        if self.similarity is not None:
            payload["similarity"] = round(self.similarity, 4)
            payload["threshold"] = self.threshold
        return payload


_EMBEDDER_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# sentence-transformers only serves the semantic arg-match: resolved by name so
# that `exact` and `token_f1` run without it.
_EMBEDDER = LazyComponent(
    module="sentence_transformers",
    class_name="SentenceTransformer",
    requires=("sentence_transformers",),
)


def normalize_value(value: Any) -> Any:  # noqa: ANN401 — arbitrary JSON argument value
    """Forgiving normalisation: accents/case/spacing for strings, recursive otherwise."""
    if isinstance(value, str):
        stripped = "".join(
            c for c in unicodedata.normalize("NFD", value.lower().strip()) if unicodedata.category(c) != "Mn"
        )
        return " ".join(stripped.split())
    if isinstance(value, dict):
        return {k: normalize_value(v) for k, v in sorted(value.items())}
    if isinstance(value, list):
        return [normalize_value(v) for v in value]
    if isinstance(value, float) and value == int(value):
        return int(value)
    return value


def _token_set(text: str) -> set[str]:
    norm = normalize_value(text)
    return set(norm.split()) if isinstance(norm, str) else set()


def token_f1(a: str, b: str) -> float:
    """Symmetric token F1 of the normalised strings — tolerant to order and light paraphrase."""
    ta, tb = _token_set(a), _token_set(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    if inter == 0:
        return 0.0
    precision, recall = inter / len(tb), inter / len(ta)
    return 2 * precision * recall / (precision + recall)


@lru_cache(maxsize=1)
def _embedder() -> Any:  # noqa: ANN401 — SentenceTransformer is untyped
    """Embedding model, built on first use and kept.

    ``lru_cache`` rather than a mutable global: same single load, no module-level
    mutable state.
    """
    return _EMBEDDER.build(model_name_or_path=_EMBEDDER_NAME)


def semantic_sim(a: str, b: str) -> float:
    """Cosine similarity of sentence embeddings (lazy, optional dependency)."""
    emb = _embedder().encode([a, b], normalize_embeddings=True)
    return float(emb[0] @ emb[1])


def _similarity(predicted: str, expected: str, *, arg_match: ArgMatch) -> float:
    return token_f1(predicted, expected) if arg_match == "token_f1" else semantic_sim(predicted, expected)


def diff_arguments(
    predicted: dict[str, Any],
    expected: dict[str, Any],
    *,
    arg_match: ArgMatch = "exact",
    threshold: float = 0.7,
    call_index: int = 0,
) -> list[ArgumentMismatch]:
    """Every reason ``predicted`` fails to satisfy ``expected``. Empty means match."""
    mismatches: list[ArgumentMismatch] = []

    for key in sorted(set(expected) - set(predicted)):
        mismatches.append(ArgumentMismatch(call_index, key, "missing_key", expected=expected[key]))
    for key in sorted(set(predicted) - set(expected)):
        # A positional argument lands here as `_positional_0`: reported as a
        # format violation rather than an unknown key, which is what it is.
        reason = "positional_argument" if key.startswith(POSITIONAL_PREFIX) else "extra_key"
        mismatches.append(ArgumentMismatch(call_index, key, reason, predicted=predicted[key]))

    for key in sorted(set(expected) & set(predicted)):
        expected_value, predicted_value = expected[key], predicted[key]
        if arg_match != "exact" and isinstance(predicted_value, str) and isinstance(expected_value, str):
            similarity = _similarity(predicted_value, expected_value, arg_match=arg_match)
            if similarity < threshold:
                mismatches.append(
                    ArgumentMismatch(
                        call_index,
                        key,
                        "below_threshold",
                        expected=expected_value,
                        predicted=predicted_value,
                        similarity=similarity,
                        threshold=threshold,
                    )
                )
        elif normalize_value(predicted_value) != normalize_value(expected_value):
            mismatches.append(
                ArgumentMismatch(
                    call_index,
                    key,
                    "value_differs",
                    expected=expected_value,
                    predicted=predicted_value,
                )
            )

    return mismatches
