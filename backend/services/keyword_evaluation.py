"""Weighted keyword coverage for transparent, explainable scoring.

Method
------
1. Extract unique content words from the reference answer (stop words removed).
2. Give each expected term a small transparent weight:
   - 2.00 for terms containing technical punctuation (C++, C#, .NET, TCP/IP),
   - 1.75 for common technical acronyms (TCP, UDP, DBMS, BCNF, ...),
   - 1.30 for longer terms (10+ characters),
   - 1.15 for medium terms (7-9 characters),
   - 1.00 otherwise.
3. A candidate matches a term when the exact token, a simple singular/plural
   variant, or every part of a compound term (for example ``tcp/ip``) appears.
4. ``keyword_score = matched weight / total expected weight * 100``.

Each expected term is counted at most once even if it occurs many times in the
reference or candidate. The matched/missing lists are returned for the UI.

Limitations: this is a lexical metric. Synonyms, paraphrases, and correct ideas
expressed with different vocabulary are not recognized. It should be read as
coverage of expected terminology, not a measure of factual correctness.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from services.nlp_preprocessing import content_tokens

TECHNICAL_ACRONYMS = frozenset(
    {
        "acid",
        "api",
        "bcnf",
        "cpu",
        "crud",
        "css",
        "dbms",
        "dns",
        "ftp",
        "gpu",
        "html",
        "http",
        "https",
        "ip",
        "json",
        "jvm",
        "lan",
        "orm",
        "oop",
        "ram",
        "smtp",
        "sql",
        "ssh",
        "tcp",
        "tls",
        "udp",
        "uri",
        "url",
        "wan",
        "xml",
        "1nf",
        "2nf",
        "3nf",
    }
)

_COMPOUND_SEPARATOR = re.compile(r"[./'-]")
_TECHNICAL_PUNCTUATION = re.compile(r"[+#./'-]")


@dataclass(frozen=True)
class KeywordCoverage:
    """Weighted keyword coverage and the terms behind the number."""

    score: float
    matched: tuple[str, ...]
    missing: tuple[str, ...]
    total_weight: float
    matched_weight: float


def _term_weight(term: str) -> float:
    if _TECHNICAL_PUNCTUATION.search(term):
        return 2.0
    if term in TECHNICAL_ACRONYMS:
        return 1.75
    if len(term) >= 10:
        return 1.3
    if len(term) >= 7:
        return 1.15
    return 1.0


def _word_forms(term: str) -> set[str]:
    """Conservative singular/plural and -ing/-ed variants for matching.

    Technical terms and acronyms are left intact unless a simple suffix rule
    applies. This is deliberately lighter than full stemming so terms such as
    ``analysis`` or ``C++`` are not damaged.
    """
    forms = {term}
    if len(term) < 4:
        return forms

    if term.endswith("ies") and len(term) > 4:
        forms.add(term[:-3] + "y")
    elif term.endswith("es") and len(term) > 4:
        forms.add(term[:-2])
        forms.add(term[:-1])
    elif term.endswith("s") and not term.endswith("ss"):
        forms.add(term[:-1])

    if term.endswith("ing") and len(term) > 6:
        base = term[:-3]
        forms.add(base)
        forms.add(base + "e")
        if len(base) > 3 and base[-1] == base[-2]:
            forms.add(base[:-1])

    if term.endswith("ed") and len(term) > 5:
        base = term[:-2]
        forms.add(base)
        forms.add(base + "e")

    return forms


def extract_reference_terms(reference_answer: str | None) -> list[str]:
    """Return unique, meaningful reference terms in first-seen order."""
    terms: list[str] = []
    seen: set[str] = set()

    for token in content_tokens(reference_answer):
        if token in seen:
            continue
        seen.add(token)
        terms.append(token)

    return terms


def _candidate_forms(candidate_tokens: list[str]) -> set[str]:
    forms: set[str] = set()
    for token in candidate_tokens:
        forms.update(_word_forms(token))
    return forms


def term_is_present(term: str, candidate_token_forms: set[str]) -> bool:
    """Check a reference term against precomputed candidate token forms."""
    if candidate_token_forms.intersection(_word_forms(term)):
        return True

    parts = [
        part
        for part in _COMPOUND_SEPARATOR.split(term)
        if len(part) >= 2
    ]
    if len(parts) > 1:
        return all(
            candidate_token_forms.intersection(_word_forms(part))
            for part in parts
        )

    return False


def keyword_coverage(
    candidate_answer: str | None,
    reference_answer: str | None,
) -> KeywordCoverage:
    """Calculate weighted keyword coverage on a 0-100 scale."""
    terms = extract_reference_terms(reference_answer)
    if not terms:
        return KeywordCoverage(
            score=0.0,
            matched=(),
            missing=(),
            total_weight=0.0,
            matched_weight=0.0,
        )

    weights = {term: _term_weight(term) for term in terms}
    candidate_forms = _candidate_forms(content_tokens(candidate_answer))

    matched: list[str] = []
    missing: list[str] = []
    matched_weight = 0.0
    total_weight = 0.0

    for term in terms:
        weight = weights[term]
        total_weight += weight
        if term_is_present(term, candidate_forms):
            matched.append(term)
            matched_weight += weight
        else:
            missing.append(term)

    matched.sort(key=lambda term: (-weights[term], term))
    missing.sort(key=lambda term: (-weights[term], term))

    score = (
        round((matched_weight / total_weight) * 100.0, 2)
        if total_weight > 0
        else 0.0
    )

    return KeywordCoverage(
        score=max(0.0, min(100.0, score)),
        matched=tuple(matched),
        missing=tuple(missing),
        total_weight=round(total_weight, 2),
        matched_weight=round(matched_weight, 2),
    )
