-- Safe, rerunnable database setup for the AI Interview Evaluator.
-- Run this file in VS Code SQLTools (or the MySQL client) after starting MySQL.
--
-- This script only creates the database. Tables are created automatically by
-- SQLAlchemy when the FastAPI backend starts. Phase 2 adds the nullable
-- interview_questions.reference_answer column (via a safe additive migration)
-- and the answer_evaluations table. Seed questions are inserted without
-- deleting or duplicating existing records, and missing reference answers are
-- backfilled without overwriting custom edits.

CREATE DATABASE IF NOT EXISTS ai_interview_evaluator
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE ai_interview_evaluator;

SHOW TABLES;

-- After starting the backend, run this to verify the seeded questions:
-- SELECT id, question_text, category, reference_answer, created_at
-- FROM interview_questions;
--
-- After evaluating answers, verify saved results:
-- SELECT id, answer_id, interview_id, question_id, semantic_score,
--        keyword_score, combined_score, status, matched_keywords,
--        missing_keywords, methodology_version
-- FROM answer_evaluations ORDER BY id;
