"""Idempotent seed data for interview questions and reference answers.

Running this module (or starting the API repeatedly) never deletes data and
never creates duplicate questions. Existing reference answers are preserved so
locally customized answers are never overwritten.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from models import InterviewQuestion

SEED_QUESTIONS: list[dict[str, str]] = [
    {
        "question_text": "Explain inheritance in object-oriented programming.",
        "category": "Object-Oriented Programming",
        "reference_answer": (
            "Inheritance is an object-oriented programming mechanism in which a "
            "child or derived class acquires the properties and methods of a "
            "parent or base class. It enables code reuse and lets the child class "
            "extend or override inherited behavior. Inheritance models an is-a "
            "relationship and can be single, multiple, multilevel, hierarchical, "
            "or hybrid. Composition is often preferred when the relationship is "
            "not truly is-a."
        ),
    },
    {
        "question_text": "What is the difference between TCP and UDP?",
        "category": "Computer Networks",
        "reference_answer": (
            "TCP is a connection-oriented, reliable transport-layer protocol. It "
            "establishes a connection, uses acknowledgements, sequence numbers, "
            "and retransmission to deliver data in order, and provides flow "
            "control and congestion control. Its reliability and ordering add "
            "overhead. UDP is a connectionless, unreliable protocol that sends "
            "independent datagrams without acknowledgements or ordering "
            "guarantees. UDP has lower overhead and latency, so it suits "
            "streaming, online gaming, voice, and DNS. TCP suits web traffic, "
            "email, and file transfers that need reliable delivery."
        ),
    },
    {
        "question_text": "What is normalization in DBMS?",
        "category": "Database Management",
        "reference_answer": (
            "Normalization in a DBMS is the process of organizing tables and "
            "attributes to reduce data redundancy and prevent insert, update, and "
            "delete anomalies. It uses functional dependencies to decompose "
            "relations into smaller well-structured tables. First normal form "
            "requires atomic values; second normal form removes partial "
            "dependencies on a composite key; third normal form removes "
            "transitive dependencies; and BCNF requires every determinant to be "
            "a candidate key. Higher normal forms also exist. Denormalization is "
            "sometimes used deliberately to improve read performance."
        ),
    },
]


def seed_questions(db: Session) -> int:
    """Insert missing seed questions and backfill missing reference answers.

    Returns the number of questions inserted. Reference answers are only filled
    when the stored value is null/blank, so custom edits survive restarts.
    """
    existing = {
        question.question_text: question
        for question in db.scalars(select(InterviewQuestion)).all()
    }

    added = 0
    backfilled = 0

    for item in SEED_QUESTIONS:
        question = existing.get(item["question_text"])

        if question is None:
            db.add(InterviewQuestion(**item))
            added += 1
            continue

        if not (question.reference_answer or "").strip():
            question.reference_answer = item["reference_answer"]
            backfilled += 1

    if not added and not backfilled:
        return 0

    try:
        db.commit()
    except IntegrityError:
        # Another process may have inserted the same unique question between
        # our read and commit. Nothing was lost; re-reading is enough.
        db.rollback()
        return 0

    return added


def main() -> None:
    from database import SessionLocal, ensure_db_ready

    ensure_db_ready()
    with SessionLocal() as db:
        questions = list(
            db.scalars(
                select(InterviewQuestion).order_by(InterviewQuestion.id)
            ).all()
        )

    missing = sum(
        1 for question in questions if not (question.reference_answer or "").strip()
    )
    print(f"Database is ready. {len(questions)} interview question(s) available.")
    print(f"{len(questions) - missing} question(s) have a reference answer.")
    if missing:
        print(
            f"{missing} question(s) are not evaluable until a reference answer is "
            "added with SQL or the seed data."
        )


if __name__ == "__main__":
    main()
