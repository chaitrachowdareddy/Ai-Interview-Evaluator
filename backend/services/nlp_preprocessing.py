"""Reusable text normalization and tokenization for Phase 2 evaluation.

The original transcript is never changed. These helpers only produce derived
token lists for keyword matching and an independent usability check.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Negations are deliberately kept: removing them would make "does not preserve
# order" look equivalent to "preserves order". Scikit-learn's default English
# stop-word list includes many negations, so remove those from the set.
NEGATION_TERMS = frozenset(
    {
        "not",
        "no",
        "nor",
        "none",
        "never",
        "neither",
        "without",
        "cannot",
        "can't",
        "don't",
        "doesn't",
        "didn't",
        "isn't",
        "aren't",
        "wasn't",
        "weren't",
        "won't",
        "wouldn't",
        "shouldn't",
        "couldn't",
        "hasn't",
        "haven't",
        "hadn't",
        "mustn't",
        "needn't",
    }
)

STOP_WORDS = frozenset(ENGLISH_STOP_WORDS - NEGATION_TERMS)

# Conversational noise is only used for the "is this answer usable?" check. It
# is not removed from keyword matching, so a real technical term is never lost.
FILLER_WORDS = frozenset(
    {
        "ah",
        "actually",
        "anyway",
        "anyways",
        "basically",
        "blah",
        "eh",
        "er",
        "erm",
        "err",
        "gonna",
        "hmm",
        "honestly",
        "kinda",
        "like",
        "literally",
        "maybe",
        "mm",
        "mmm",
        "ok",
        "okay",
        "probably",
        "really",
        "right",
        "seriously",
        "sorta",
        "stuff",
        "thing",
        "things",
        "uh",
        "uhh",
        "uhm",
        "um",
        "umm",
        "very",
        "wanna",
        "whatever",
        "well",
        "yknow",
    }
)

# Keep technical punctuation inside tokens: C++, C#, .NET, Node.js, TCP/IP,
# object-oriented, 1NF, and contractions such as "don't".
_TOKEN_PATTERN = re.compile(
    r"(?<![a-z0-9])\.?[a-z0-9]"
    r"(?:[a-z0-9+#]|(?<=[a-z0-9])[._/'-](?=[a-z0-9]))*",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class AnswerUsability:
    """The result of checking whether a transcript can be scored at all."""

    usable: bool
    reason: str
    meaningful_token_count: int


def normalize_text(text: str | None) -> str:
    """Normalize Unicode and whitespace while preserving wording/punctuation."""
    if not text:
        return ""

    normalized = unicodedata.normalize("NFKC", text)
    normalized = (
        normalized.replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )
    return re.sub(r"\s+", " ", normalized).strip()


def tokenize(text: str | None) -> list[str]:
    """Return lowercase word/technical tokens from text, in order."""
    normalized = normalize_text(text).lower()
    if not normalized:
        return []

    tokens: list[str] = []
    for match in _TOKEN_PATTERN.finditer(normalized):
        # Keep a leading dot so ".NET" is not reduced to "net"; only trim
        # separators that cannot start a meaningful technical term.
        token = match.group(0).rstrip("._/'-").lstrip("_/'-")
        if token:
            tokens.append(token)
    return tokens


def is_stop_word(token: str) -> bool:
    return token in STOP_WORDS


def content_tokens(text: str | None) -> list[str]:
    """Tokens that carry meaning, excluding ordinary English stop words."""
    return [token for token in tokenize(text) if token not in STOP_WORDS]


def meaningful_tokens(text: str | None) -> list[str]:
    """Content tokens with obvious conversational filler removed."""
    return [
        token
        for token in tokenize(text)
        if token not in STOP_WORDS and token not in FILLER_WORDS
    ]


def assess_answer_usability(
    text: str | None,
    min_meaningful_tokens: int = 3,
) -> AnswerUsability:
    """Decide whether an answer is long enough to compare with a reference.

    Short/filler-only answers receive an explicit ``unevaluable`` status in the
    evaluation pipeline rather than relying on semantic similarity alone.
    """
    normalized = normalize_text(text)
    if not normalized:
        return AnswerUsability(
            usable=False,
            reason="The saved transcript is empty.",
            meaningful_token_count=0,
        )

    tokens = meaningful_tokens(normalized)
    if not tokens:
        return AnswerUsability(
            usable=False,
            reason=(
                "The saved transcript contains only filler or stop words, so "
                "there is no meaningful answer to evaluate."
            ),
            meaningful_token_count=0,
        )

    if len(tokens) < min_meaningful_tokens:
        return AnswerUsability(
            usable=False,
            reason=(
                "The saved transcript is too short for a reliable automated "
                f"comparison (found {len(tokens)} meaningful word(s); at least "
                f"{min_meaningful_tokens} are required)."
            ),
            meaningful_token_count=len(tokens),
        )

    return AnswerUsability(
        usable=True,
        reason="",
        meaningful_token_count=len(tokens),
    )


def text_for_embedding(text: str | None) -> str:
    """Prepare text for sentence embeddings without destroying meaning.

    Sentence-transformers models handle case and punctuation well, so only
    Unicode/whitespace normalization is applied. Negations and technical
    symbols are preserved.
    """
    return normalize_text(text)
