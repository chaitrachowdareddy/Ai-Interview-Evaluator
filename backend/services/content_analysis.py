"""Phase 3 content analysis: explainable lexical and structure statistics.

This module deliberately reuses the Phase 2 keyword coverage instead of
inventing a second coverage formula. If a valid Phase 2 evaluation exists, its
matched/missing lists and score are reused unchanged. Otherwise the exact same
``keyword_coverage`` function is called on the current transcript/reference, so
the two features never show conflicting coverage numbers.

The coverage number is **lexical**: it measures whether expected terminology
appears, not whether the answer is factually correct. Filler, repetition, and
sentence statistics describe the transcript only; they are not a personality or
competency judgement.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from config import EVALUATION_MIN_MEANINGFUL_TOKENS
from services.keyword_evaluation import keyword_coverage
from services.nlp_preprocessing import (
    FILLER_WORDS,
    STOP_WORDS,
    assess_answer_usability,
    normalize_text,
    tokenize,
)

STATUS_COMPLETED = "completed"
STATUS_UNEVALUABLE = "unevaluable"
STATUS_NOT_EVALUABLE = "not_evaluable"

COVERAGE_SOURCE_PHASE2 = "phase2_evaluation"
COVERAGE_SOURCE_KEYWORDS = "keyword_coverage"
COVERAGE_SOURCE_NONE = "none"

_REPEATED_TERM_LIMIT = 10
_SENTENCE_BOUNDARY = re.compile(r"[.!?]+(?=\s|$)")
_TERMINAL_PUNCTUATION = re.compile(r"[.!?]\s*$")


@dataclass(frozen=True)
class ExistingCoverage:
    """Coverage copied from a currently valid Phase 2 evaluation."""

    score: float
    matched: tuple[str, ...]
    missing: tuple[str, ...]


@dataclass(frozen=True)
class ContentAnalysisResult:
    status: str
    status_detail: str | None
    coverage_score: float | None
    coverage_source: str
    matched_concepts: tuple[str, ...]
    missing_concepts: tuple[str, ...]
    filler_word_count: int
    filler_rate_percent: float | None
    filler_words: tuple[tuple[str, int], ...]
    repeated_terms: tuple[tuple[str, int], ...]
    repetition_rate_percent: float | None
    sentence_count: int
    average_sentence_words: float | None
    structure_notes: tuple[str, ...]


def _round(value: float) -> float:
    return round(value, 2)


def _sentence_word_counts(text: str) -> list[int]:
    if not text:
        return []

    segments = [
        segment.strip()
        for segment in _SENTENCE_BOUNDARY.split(text)
        if segment.strip()
    ]
    if not segments and text.strip():
        segments = [text.strip()]

    counts: list[int] = []
    for segment in segments:
        tokens = tokenize(segment)
        if tokens:
            counts.append(len(tokens))
    return counts


def _filler_statistics(raw_tokens: list[str]) -> tuple[
    int,
    float | None,
    tuple[tuple[str, int], ...],
]:
    if not raw_tokens:
        return 0, None, ()

    counts = Counter(
        token for token in raw_tokens if token in FILLER_WORDS
    )
    total = sum(counts.values())
    rate = (total / len(raw_tokens)) * 100.0
    ordered = tuple(
        sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    )
    return total, _round(rate), ordered


def _repetition_statistics(raw_tokens: list[str]) -> tuple[
    tuple[tuple[str, int], ...],
    float | None,
]:
    meaningful = [token for token in raw_tokens if token not in STOP_WORDS]
    if not meaningful:
        return (), None

    counts = Counter(meaningful)
    repeated = [
        (term, count)
        for term, count in counts.items()
        if count > 1 and len(term) > 1
    ]
    repeated.sort(key=lambda item: (-item[1], item[0]))
    repeated_occurrences = sum(count - 1 for _term, count in repeated)
    rate = (repeated_occurrences / len(meaningful)) * 100.0
    return tuple(repeated[:_REPEATED_TERM_LIMIT]), _round(rate)


def _structure_notes(
    text: str,
    *,
    word_count: int,
    sentence_word_counts: list[int],
    average_sentence_words: float | None,
    filler_rate_percent: float | None,
    repetition_rate_percent: float | None,
    coverage_score: float | None,
    missing_concepts: tuple[str, ...],
) -> tuple[str, ...]:
    notes: list[str] = []

    if word_count == 0:
        return ()

    if word_count < 15:
        notes.append(
            f"The answer is short ({word_count} word(s)), so structure "
            "statistics are limited."
        )

    if (
        len(sentence_word_counts) == 1
        and word_count >= 35
        and not _TERMINAL_PUNCTUATION.search(text)
    ):
        notes.append(
            "No sentence-ending punctuation was detected in a long "
            "transcript; sentence segmentation is approximate."
        )

    if average_sentence_words is not None and average_sentence_words >= 35:
        notes.append(
            "Sentences are long on average "
            f"({average_sentence_words:.1f} words); shorter sentences may be "
            "easier to follow."
        )

    longest = max(sentence_word_counts) if sentence_word_counts else 0
    if longest >= 50:
        notes.append(
            f"The longest sentence has {longest} words and may be difficult "
            "to follow when spoken."
        )

    if filler_rate_percent is not None and filler_rate_percent >= 10:
        notes.append(
            f"Filler words make up {filler_rate_percent:.1f}% of the "
            "transcript."
        )

    if repetition_rate_percent is not None and repetition_rate_percent >= 25:
        notes.append(
            f"Repeated content words account for "
            f"{repetition_rate_percent:.1f}% of the meaningful words."
        )

    if (
        coverage_score is not None
        and coverage_score < 40
        and missing_concepts
    ):
        notes.append(
            f"Coverage of expected reference terminology is low "
            f"({coverage_score:.1f} / 100); "
            f"{len(missing_concepts)} expected concept(s) were not found."
        )

    return tuple(notes)


def analyze_content(
    transcript: str | None,
    reference_answer: str | None,
    existing_coverage: ExistingCoverage | None = None,
) -> ContentAnalysisResult:
    """Analyze one transcript without mutating the saved answer.

    ``existing_coverage`` should only be supplied when a Phase 2 evaluation is
    current (its transcript/reference hashes match). Otherwise coverage is
    recomputed with the same weighted keyword function.
    """

    text = normalize_text(transcript)
    reference = normalize_text(reference_answer)
    raw_tokens = tokenize(text)

    word_count = len(raw_tokens)
    sentence_word_counts = _sentence_word_counts(text)
    average_sentence_words = (
        _round(sum(sentence_word_counts) / len(sentence_word_counts))
        if sentence_word_counts
        else None
    )

    filler_count, filler_rate, filler_words = _filler_statistics(raw_tokens)
    repeated_terms, repetition_rate = _repetition_statistics(raw_tokens)

    if not reference:
        status = STATUS_NOT_EVALUABLE
        status_detail = (
            "This question has no reference answer, so reference coverage "
            "cannot be calculated. Transcript-only statistics are shown."
        )
        coverage_score = None
        coverage_source = COVERAGE_SOURCE_NONE
        matched: tuple[str, ...] = ()
        missing: tuple[str, ...] = ()
    else:
        usability = assess_answer_usability(
            text,
            min_meaningful_tokens=EVALUATION_MIN_MEANINGFUL_TOKENS,
        )
        if not usability.usable:
            status = STATUS_UNEVALUABLE
            status_detail = usability.reason
            coverage_score = None
            coverage_source = COVERAGE_SOURCE_NONE
            matched = ()
            missing = ()
        else:
            status = STATUS_COMPLETED
            status_detail = None
            if existing_coverage is not None:
                coverage_score = _round(
                    max(0.0, min(100.0, existing_coverage.score))
                )
                coverage_source = COVERAGE_SOURCE_PHASE2
                matched = existing_coverage.matched
                missing = existing_coverage.missing
            else:
                coverage = keyword_coverage(text, reference)
                coverage_score = coverage.score
                coverage_source = COVERAGE_SOURCE_KEYWORDS
                matched = coverage.matched
                missing = coverage.missing

    notes = _structure_notes(
        text,
        word_count=word_count,
        sentence_word_counts=sentence_word_counts,
        average_sentence_words=average_sentence_words,
        filler_rate_percent=filler_rate,
        repetition_rate_percent=repetition_rate,
        coverage_score=coverage_score,
        missing_concepts=missing,
    )

    return ContentAnalysisResult(
        status=status,
        status_detail=status_detail,
        coverage_score=coverage_score,
        coverage_source=coverage_source,
        matched_concepts=matched,
        missing_concepts=missing,
        filler_word_count=filler_count,
        filler_rate_percent=filler_rate,
        filler_words=filler_words,
        repeated_terms=repeated_terms,
        repetition_rate_percent=repetition_rate,
        sentence_count=len(sentence_word_counts),
        average_sentence_words=average_sentence_words,
        structure_notes=notes,
    )
