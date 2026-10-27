# AI Interview Evaluator

Phase 1 + Phase 2 + Phase 3 + Phase 4 of a local technical-interview practice app.

Workflow: **React (Vite) → FastAPI → saved transcript and original audio → NLP + Sentence-Transformers + Librosa → MySQL → scores, measurements, and a clearly labelled prediction shown in React**.

Phase 1 records an answer in the browser, transcribes it with Groq Whisper, and
stores the transcript in MySQL. Phase 2 evaluates that **already-saved**
transcript against a reference answer and stores an explainable score. Phase 3
adds separate content statistics and objective voice measurements from the
saved recording. Phase 4 adds a versioned scoring pipeline: when a validated
model artifact is installed it returns an **ML prediction**; otherwise it
returns a transparent **rule-based baseline** and says so explicitly.

The automated score compares one answer with one reference answer. It is a
practice aid, not a measure of a candidate's complete knowledge, a hiring
decision, or an objective suitability rating. Voice features are reported as
acoustic measurements only; they are never interpreted as confidence,
personality, honesty, intelligence, or emotion. Phase 4 does not fabricate an
ML model: **no labelled training dataset is included in this repository and no
model artifact is committed or installed**, so a fresh checkout uses the
labelled baseline until an operator trains and installs a validated artifact.

## What is implemented

### Phase 1 (preserved)

- Interview questions are stored in MySQL and loaded through `GET /interview/questions`.
- The database is seeded with three technical questions without duplicating them on restart.
- A persisted interview session can be created for each practice run.
- The browser records an answer with `MediaRecorder` and negotiates a supported audio MIME type, or the user uploads a supported audio file.
- FastAPI validates the recording, sends it to Groq Whisper, and saves the transcript against the session/question.
- Saved transcripts are retrieved from MySQL and displayed in the React UI; the active session ID is kept in local storage so a page refresh restores the same session and transcripts.
- Invalid sessions/questions, empty or oversized recordings, unsupported audio formats, missing Groq configuration, Groq errors, and database errors return clear, safe responses.

### Phase 2

- Every seeded question has a reference answer; custom reference answers are never overwritten by seeding.
- Reusable NLP preprocessing preserves technical terms (`C++`, `.NET`, `TCP/IP`), acronyms, and negations.
- Weighted keyword coverage with explainable matched/missing keyword lists.
- Sentence-Transformers semantic similarity (`sentence-transformers/all-MiniLM-L6-v2`) with cosine similarity mapped to 0–100.
- Combined score = **70% semantic + 30% keyword** (weights configurable in one place).
- Evaluations are persisted in MySQL with foreign keys and one row per answer/scoring version.
- Duplicate requests reuse the saved evaluation; changed transcripts/references are detected as stale and can be re-evaluated in place.
- Interview summaries use only currently valid evaluated answers for the average.
- React UI adds evaluation cards, status badges, score breakdowns, keyword chips, errors, and a per-question summary.

### Phase 3

- The original recording is now persisted atomically in `backend/uploads/` with a content-addressed name, relative path, and SHA-256 so the saved audio can be measured later. A storage failure never discards a successful transcript.
- **Content analysis** reuses the Phase 2 weighted keyword coverage and matched/missing lists instead of inventing a second coverage formula. It adds filler-word counts/share, repeated content terms/share, sentence counts, average sentence length, and transparent structure notes.
- Content analysis is explicitly lexical, not factual correctness. A missing reference answer is persisted as `not_evaluable` while transcript-only statistics are still shown.
- **Voice analysis** uses Librosa + soundfile with a bundled FFmpeg fallback to decode WebM/M4A/MP3 containers. It measures duration, detected speech/silence via energy VAD, pauses, speaking rate from the saved transcript, RMS/peak/clipping, and pitch (F0) when enough voiced audio exists.
- Voice metrics are additive and separate from the Phase 2 score. Missing audio, exhausted audio storage, or an undecodable file is stored as an explicit `unavailable`/`failed`/`insufficient_speech` status; metrics are not fabricated.
- Two new tables (`content_analyses`, `voice_analyses`) store one row per answer and methodology version (`phase3-v1`). Changed transcripts, references, or audio are detected as stale and re-analysed in place.
- React UI adds separate **Content** and **Voice** cards, status badges, metric grids, quality notes, and saved-answer indicators.

### Phase 4

- **No labelled dataset ships with the repository, and no model is claimed.** Phase 4 inspects the project, defines a reproducible training path, and uses an explicitly labelled rule-based baseline until a validated artifact is installed.
- A versioned feature schema (`phase4-features-v1`) reuses Phase 2 semantic/keyword scores and current Phase 3 content/voice measurements. Semantic and keyword are required; optional features are represented as unavailable, never as zero.
- Source eligibility is strict: Phase 2 must be `evaluated` with matching hashes; content must be `completed` with matching transcript/reference hashes; voice must be `completed`, `speech_detected`, and match transcript/audio hashes. Stale or unevaluable sources are excluded.
- The **rule-based baseline** (`rule_based_baseline`) reuses the Phase 2 combined score as the answer component (weight 0.60), adds a transcript-style component (0.20), and a delivery component from voice (0.20). Available components are renormalized; missing components are excluded, not zero-filled. It is never labelled an ML prediction.
- **ML inference** (`ml_model`) is used only when a local artifact is present, was trained with this feature schema, is marked `validated` by the training pipeline, and is compatible with the installed scikit-learn version. Missing, invalid, unvalidated, incompatible, or failing artifacts produce an explicit `unavailable`/`failed` status — never a disguised heuristic.
- `backend/services/ml_training.py` is a reproducible, group-aware Random Forest pipeline (`GroupShuffleSplit`, median imputer fitted on the training split only, `RandomForestRegressor`). It validates the CSV, rejects unknown/duplicate/leaked columns, reports MAE/RMSE/R² against a mean-prediction baseline, and refuses to call an artifact validated below the configured sample/group/test thresholds. `--allow-small` writes an explicitly `unvalidated` artifact that inference refuses.
- Predictions are additive and persisted in `score_predictions`, one row per answer and methodology version (`phase4-v1`). Repeating a request reuses the row; changed sources update it in place. Staleness is computed on read from the stored source fingerprint.
- React UI adds a separate **Prediction** card with method/status labels, baseline components, model details, unavailable/stale/failed states, warnings, limitations, saved-answer prediction status, an ML-model status pill, and prediction statistics in the interview summary.

LLM-generated feedback, candidate ranking, authentication, deployment, and
invented ML accuracy claims are intentionally **not** part of Phase 4. The
Random Forest is implemented and tested, but a real model is only produced from
an operator-supplied labelled dataset.

## Project layout

```text
AI interview evalutor/
├── backend/
│   ├── config.py                      # env settings, evaluation weights/model
│   ├── database.py                    # engine/session, safe additive migration
│   ├── models.py                      # questions, sessions, answers, evaluations
│   ├── schemas.py                     # Pydantic request/response models
│   ├── seed.py                        # idempotent questions + reference answers
│   ├── main.py                        # FastAPI app, health, CORS, error handling
│   ├── routes/interview.py            # Phase 1 endpoints
│   ├── routes/evaluation.py           # Phase 2 evaluation/summary endpoints
│   ├── routes/analysis.py             # Phase 3 content/voice endpoints
│   ├── routes/prediction.py           # Phase 4 prediction/schema endpoints
│   ├── services/
│   │   ├── transcription.py           # Groq Whisper + audio validation
│   │   ├── nlp_preprocessing.py       # normalization, tokenization, usability
│   │   ├── keyword_evaluation.py      # weighted keyword coverage
│   │   ├── semantic_similarity.py     # cached model + cosine similarity
│   │   ├── scoring.py                 # combined-score formula
│   │   ├── evaluation.py              # evaluation pipeline/statuses/hashes
│   │   ├── audio_storage.py           # atomic recording storage + SHA-256
│   │   ├── content_analysis.py        # coverage/filler/repetition/structure
│   │   ├── voice_analysis.py          # Librosa measurements + FFmpeg decode
│   │   ├── ml_features.py             # versioned feature schema + eligibility
│   │   ├── ml_scoring.py              # artifact status, inference, baseline
│   │   └── ml_training.py             # reproducible group-aware training CLI
│   ├── ml_artifacts/                  # local model artifacts (ignored by git)
│   ├── tests/                         # isolated API/service tests
│   ├── requirements.txt
│   ├── requirements-dev.txt
│   ├── .env.example                   # placeholders only
│   ├── uploads/                       # local recordings (ignored by git)
│   └── venv/                          # existing local virtualenv
├── frontend/
│   ├── src/App.jsx                    # interview workflow + analysis state
│   ├── src/EvaluationViews.jsx        # evaluation cards + interview summary
│   ├── src/AnalysisViews.jsx          # Phase 3 content/voice cards
│   ├── src/PredictionViews.jsx        # Phase 4 prediction card
│   ├── src/analysisStatus.js          # Phase 3 status labels/tones
│   ├── src/predictionStatus.js        # Phase 4 method/status labels
│   ├── src/App.css
│   ├── src/api.js                     # API base URL + request helpers
│   ├── src/audio.js                   # MediaRecorder helpers
│   └── .env.example                   # optional VITE_API_BASE_URL
└── setup.sql                          # safe CREATE DATABASE only
```

## Evaluation method

### 1. Preprocessing

`backend/services/nlp_preprocessing.py` normalizes Unicode and whitespace,
tokenizes with a regex that keeps technical punctuation (`C++`, `C#`, `.NET`,
`Node.js`, `TCP/IP`, `1NF`) and keeps negations (`not`, `never`, `without`, …)
out of the stop-word set. The **stored transcript is never modified**;
preprocessing only produces derived token lists.

An answer is marked `unevaluable` when it is empty, contains only filler/stop
words, or has fewer than `EVALUATION_MIN_MEANINGFUL_TOKENS` (default 3)
meaningful tokens. It then receives zero scores and an explanatory message
instead of a misleading semantic-only score.

### 2. Weighted keyword coverage

`backend/services/keyword_evaluation.py` extracts unique content terms from the
reference answer and gives each a small transparent weight:

| Term type | Weight |
| --- | --- |
| Contains technical punctuation (`c++`, `tcp/ip`) | 2.00 |
| Common technical acronym (`tcp`, `udp`, `dbms`, `bcnf`, …) | 1.75 |
| 10+ characters | 1.30 |
| 7–9 characters | 1.15 |
| Other content terms | 1.00 |

`keyword_score = matched weight / total expected weight × 100`.

Each expected term counts at most once, even if it appears repeatedly.
Conservative singular/plural and `-ing`/`-ed` variants are recognized, and
compound terms such as `tcp/ip` match when all parts are present. Matched and
missing terms are stored and displayed.

**Limitation:** this is a lexical metric. Correct paraphrases and synonyms that
avoid the expected vocabulary are not recognized. It measures coverage of
expected terminology, not factual correctness.

### 3. Semantic similarity

`backend/services/semantic_similarity.py` lazily loads
`sentence-transformers/all-MiniLM-L6-v2` once per backend process, embeds the
reference and candidate, and computes cosine similarity with Scikit-learn.
Cosine values are clamped to `[0, 1]` and multiplied by 100 (negative values
become 0). The model is loaded through a lock so concurrent requests do not
initialize it repeatedly.

The first evaluation needs internet access to download the model (about 90 MB).
After that it is read from the Hugging Face cache. Model/import failures return
a safe `503`; the code never substitutes fake scores.

**Limitation:** cosine similarity is a geometric similarity between embeddings,
not a probability that an answer is correct. Fluent but factually wrong answers
can still score highly, which is why semantic similarity is only one component.

### 4. Combined score

`backend/services/scoring.py` implements:

```text
combined_score = 0.70 * semantic_score + 0.30 * keyword_score
```

Both components and the result are clamped to 0–100 and rounded to two decimal
places. Weights live in `config.py` (overridable through `backend/.env`) and are
normalized if they do not add up to 1. The methodology version is stored with
each evaluation as `phase2-v1`.

## Phase 3 method

Phase 3 is additive: it reads the same saved transcript and, for the first time,
the saved original recording. It never modifies the Phase 2 evaluation.

### Content statistics

`backend/services/content_analysis.py` produces:

- **Reference coverage** — copied from the current Phase 2 evaluation when its
  transcript/reference hashes match. Otherwise the exact same weighted
  `keyword_coverage` function is called, so the two features never show
  conflicting coverage numbers. This remains a lexical metric.
- **Filler words** — counts raw tokens found in the shared `FILLER_WORDS` set
  (for example `um`, `uh`, `basically`) and their share of the transcript.
- **Repetition** — content words repeated more than once, with the share of all
  meaningful tokens occupied by extra occurrences.
- **Sentence statistics** — approximate sentence count and average words per
  sentence, based on `.!?` boundaries. Structure notes are emitted only for
  observable conditions (short answer, very long sentences, high filler share,
  no terminal punctuation, low lexical coverage).

A missing reference answer produces `not_evaluable`; a filler-only or too-short
transcript produces `unevaluable`. Transcript statistics are still stored and
shown in both cases. Content analysis does not claim factual correctness.

### Voice measurements

`backend/services/voice_analysis.py`:

1. Reads the stored file with `soundfile`; if the container/codec is unsupported,
   decodes mono 16 kHz audio with the FFmpeg binary bundled by
   `imageio-ffmpeg`. Analysis is capped at `VOICE_MAX_ANALYSIS_SECONDS` (300 by
   default) and the response says when only part of a long file was analysed.
2. Measures duration, RMS energy (linear and dBFS), level variation, peak
   amplitude, and the percentage of samples at/near full scale.
3. Uses energy VAD (`librosa.effects.split`) to separate detected speech from
   silence. Digitally silent audio is treated as silence rather than one long
   speech interval.
4. Counts pauses as silent gaps of at least `VOICE_MIN_PAUSE_SECONDS` between
   detected speech segments. Leading/trailing silence is counted as silence, not
   as a pause.
5. Computes speaking rate as **saved transcript words ÷ detected speech time**
   (words per minute, pauses excluded). It is not a second transcription and is
   unavailable for empty transcripts or less than one second of speech.
6. Estimates pitch with `librosa.pyin` on up to `VOICE_PITCH_MAX_SECONDS` of
   detected speech. Pitch values are stored only when enough voiced frames exist
   and voiced audio is a reasonable share of the analysed speech; otherwise the
   fields stay `null` with `pitch_reliable: false`.

Voice statuses are `completed`, `insufficient_speech`, `unavailable`, and
`failed`. Missing/old audio is `unavailable`, an undecodable file is `failed`,
and neither invents metric values. Missing Python/FFmpeg dependencies return
`503` without writing a misleading row.

These measurements are descriptive. They are **not** converted into confidence,
personality, honesty, intelligence, or emotion scores.

## Phase 4 method

Phase 4 is additive: it reads the saved Phase 2 evaluation and current Phase 3
analyses, and writes a separate prediction row. It never rewrites a Phase 2
score or a Phase 3 analysis.

### Dataset availability (read this first)

No labelled dataset is included in this repository. There is no CSV/TSV/JSONL/
parquet/XLSX/ARFF file, and the documentation does not point at a licensed
source of human interview scores. Consequently this checkout does **not** train
a real model, does **not** ship a model artifact, and does **not** claim any
real-world accuracy or validation metric. The automated tests use synthetic,
temporary data only to verify the pipeline mechanics. Until you provide a
licensed, representative labelled dataset, predictions use the labelled
rule-based baseline.

### Feature schema (`phase4-features-v1`)

The exact order is stable and recorded inside every artifact:

| # | Feature | Required | Range | Source / meaning |
| --- | --- | --- | --- | --- |
| 1 | `semantic_score` | yes | 0–100 | Phase 2 semantic similarity |
| 2 | `keyword_score` | yes | 0–100 | Phase 2 weighted keyword coverage |
| 3 | `filler_rate_percent` | no | 0–100 | Phase 3 content: filler share |
| 4 | `repetition_rate_percent` | no | 0–100 | Phase 3 content: repeated content words |
| 5 | `average_sentence_words` | no | 1–200 | Phase 3 content: mean sentence length |
| 6 | `speaking_rate_wpm` | no | 0–400 | Phase 3 voice: detected speech rate |
| 7 | `pauses_per_minute` | no | 0–200 | Phase 3 voice: pause frequency |
| 8 | `speech_ratio_percent` | no | 0–100 | Phase 3 voice: detected speech share |
| 9 | `rms_variation_percent` | no | 0–500 | Phase 3 voice: loudness variation |
| 10 | `has_content_features` | no | 0/1 | Content availability indicator |
| 11 | `has_voice_features` | no | 0/1 | Voice availability indicator |

Pitch is deliberately **excluded**: it is a speaker characteristic with known
demographic variation and is not a defensible scoring input. Content coverage is
not duplicated; `keyword_score` is the single coverage feature. Identity-like
columns (`sample_id`, `candidate_id`, `interview_id`, `answer_id`) are accepted
by the training CSV but ignored, so they cannot leak into the model.

### Eligibility, missing data, and staleness

A source is usable only when it is current:

- Phase 2: status `evaluated`, and transcript/reference hashes match.
- Content: status `completed`, and transcript/reference hashes match.
- Voice: status `completed`, `speech_detected` is true, and transcript/audio
  hashes match.

Stale, unevaluable, not-evaluable, missing, or failed sources are excluded.
Required semantic/keyword values missing means no score is produced
(`unavailable`). Optional features may be absent; the baseline excludes their
component and renormalizes the remaining weights. The training pipeline uses a
median imputer fitted **only on the training split**, so an unavailable optional
feature is imputed in ML mode rather than silently treated as zero. Missingness
is stored in `missing_features` and shown in the UI.

Every prediction stores a source fingerprint. On read, the API compares it with
the current sources and versions and sets `is_stale` without rewriting data.
Only a new POST refreshes the same `(answer_id, methodology_version)` row.

### Rule-based baseline (`rule_based_baseline`)

The baseline is transparent and fixed. It is **not** trained and must never be
described as an ML prediction:

```text
answer component        = current Phase 2 combined_score   (weight 0.60)
content-style component = filler / repetition / sentence length (weight 0.20)
delivery component      = speaking rate / pauses / speech ratio /
                          loudness variation                (weight 0.20)

baseline = sum(available component_score * weight)
           / sum(available component_weight)
```

Component weights are renormalized when content or voice is unavailable. The
answer component is reused unchanged from Phase 2, not recomputed. Because the
weights and target ranges are design choices, every baseline response includes
that limitation and the UI labels it prominently.

### Reproducible ML training

`backend/services/ml_training.py` implements the supported training path:

1. Read and validate a labelled CSV with `group_id`, `target_score`, and all 11
   feature columns. Reject missing/unknown columns, non-numeric or out-of-range
   values, duplicate `sample_id`s, and exact duplicate rows.
2. Split **by `group_id`** with `GroupShuffleSplit`, then assert that no group
   appears on both sides. This prevents the same candidate/interview leaking from
   train to test.
3. Fit `SimpleImputer(strategy="median")` on the training split only, then
   `RandomForestRegressor`.
4. Evaluate with MAE, RMSE, and R² on the held-out split, plus a
   mean-prediction `DummyRegressor` baseline.
5. Mark the artifact `validated` only when the dataset meets
   `ML_MIN_TRAINING_SAMPLES` / `ML_MIN_TRAINING_GROUPS`, the held-out set is
   large enough, the target has variation, held-out MAE beats the mean
   baseline, and R² is not negative. Otherwise it is `unvalidated`.
6. Write the joblib bundle atomically, plus a `.metadata.json` sidecar with the
   dataset hash, split counts, metrics, warnings, and limitations.

```powershell
Set-Location "...\backend"
.\venv\Scripts\python.exe -m services.ml_training --data labeled_scores.csv

# Experiment only: writes an unvalidated artifact inference will refuse.
.\venv\Scripts\python.exe -m services.ml_training --data labeled_scores.csv --allow-small
```

The default artifact path is `backend/ml_artifacts/interview_score.joblib`
(override with `ML_MODEL_PATH`). The artifact bundle contains the fitted
pipeline, exact feature order, feature-schema and model versions, training time,
scikit-learn version, validation status, metrics, limitations, and target
bounds. `--force` is required to replace an existing artifact deliberately.

### Inference selection

`backend/services/ml_scoring.py` uses the ML artifact only when **all** of these
hold: the file exists and loads, the bundle has every required key, the feature
names/order match, the feature schema matches, `validation_status` is
`validated`, the scikit-learn major/minor version is compatible, the target
bounds match, and the estimator can predict. Otherwise the response records an
explicit state (`unavailable`, `invalid`, `incompatible`, `unvalidated`, or
`failed`). A model failure is reported; it is **never** replaced by a baseline
score and then called an ML prediction. ML outputs are clamped to 0–100 and
store the model version.

### Phase 4 limitations

- No real labelled data means no real model, no real metrics, and no claim that
  an ML prediction is better than the baseline.
- The baseline weights and target ranges are hand-designed, not learned.
- Phase 2 is lexical/semantic agreement with one reference answer, not factual
  truth.
- Voice features are acoustic measurements, not confidence or personality.
- Any future model is only as representative as its labelled data; subgroup
  fairness and calibration must be reviewed before any high-stakes use.
- These scores are practice aids, not hiring decisions.

## Database schema and migration

### Changes

- `interview_questions.reference_answer` — nullable `TEXT`. Existing question IDs
  and rows are preserved. A question without a reference answer is reported as
  `not_evaluable`; no score is invented.
- New `answer_evaluations` table:
  - `id` primary key
  - `answer_id` → `interview_answers.id` (`ON DELETE CASCADE`, indexed)
  - `interview_id` → `interview_sessions.id` (`ON DELETE CASCADE`, indexed)
  - `question_id` → `interview_questions.id` (indexed)
  - `semantic_score`, `keyword_score`, `combined_score` (`FLOAT`)
  - `matched_keywords`, `missing_keywords` (`JSON`)
  - `status`, `status_detail`, `methodology_version`
  - `transcript_hash`, `reference_hash` (SHA-256, stale detection)
  - `created_at`, `updated_at`
  - `UNIQUE (answer_id, methodology_version)` for idempotency
- `interview_answers.audio_path` — nullable `VARCHAR(500)`, relative to
  `AUDIO_STORAGE_DIR`; stores only the portable path, not an absolute OS path.
- `interview_answers.audio_sha256` — nullable `VARCHAR(64)` content fingerprint.
- New `content_analyses` table (unique `answer_id + methodology_version`):
  coverage score/source, matched/missing concepts, filler and repetition JSON
  counts/rates, sentence statistics, structure notes, status/detail,
  transcript/reference hashes, timestamps.
- New `voice_analyses` table (unique `answer_id + methodology_version`):
  status/detail, duration/sample metadata, speech/silence/pause statistics,
  estimated speaking rate, energy/clipping statistics, pitch statistics and a
  `pitch_reliable` flag, quality notes, audio/transcript hashes, timestamps.
- New `score_predictions` table (unique `answer_id + methodology_version`):
  status/detail, method (`ml_model` / `rule_based_baseline` / `none`), score,
  model/feature-schema/methodology versions, validated feature values,
  `missing_features`, baseline/model components, warnings, limitations, source
  versions, and a SHA-256 `source_fingerprint` used for read-time staleness.
  It is additive to Phase 2/3 and never changes their rows.
- Both analysis tables use foreign keys to answers, sessions, and questions with
  `ON DELETE CASCADE` on answer/session ownership; `score_predictions` follows
  the same ownership and uniqueness convention.
- Recording files are stored under `backend/uploads/` (ignored by git). Names are
  content-addressed (`interview-{id}/question-{id}-{sha256-prefix}.{ext}`), and
  writes use a temporary file plus atomic replace.

### Migration behavior

The project has no migration framework. `database.init_db()`:

1. Calls `Base.metadata.create_all()` to create missing tables only, including the
   Phase 2 evaluation table, the Phase 3 analysis tables, and the Phase 4
   prediction table.
2. Inspects `interview_questions`; if `reference_answer` is absent, runs
   `ALTER TABLE interview_questions ADD COLUMN reference_answer TEXT NULL`.
3. Inspects `interview_answers`; if `audio_path`/`audio_sha256` are absent, adds
   them with safe `ALTER TABLE ... ADD COLUMN` statements. This is additive and
   rerunnable, and existing answers remain valid with null audio fields.
4. Runs the idempotent seed/backfill.

It **never drops tables or deletes rows**, and it does not rewrite existing
transcripts, evaluations, or answers. The checks make startup safe to rerun.
Future non-additive column/constraint changes still require a manual migration.

`seed.py` inserts missing questions and fills reference answers only when the
stored value is null/blank. Custom answers survive restarts. To edit one
manually:

```sql
UPDATE interview_questions
SET reference_answer = 'Your curated reference answer.'
WHERE id = 1;
```

Run the CLI check from `backend`:

```powershell
.\venv\Scripts\python.exe seed.py
```

## API endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | API status (reports phase `4`) |
| `GET` | `/health` | API + MySQL + Groq + voice-analysis + ML-model status |
| `GET` | `/interview/questions` | Questions, including `has_reference_answer` |
| `POST` | `/interview/start` | Create a persisted interview session |
| `POST` | `/interview/transcribe` | Multipart audio + `interview_id` + `question_id` |
| `GET` | `/interview/{interview_id}/answers` | Session metadata and saved transcripts (`has_audio`) |
| `POST` | `/evaluation/{answer_id}` | Evaluate a saved transcript (new or reused) |
| `GET` | `/evaluation/{answer_id}` | Retrieve the saved evaluation |
| `GET` | `/interview/{interview_id}/evaluations` | Evaluations for one session |
| `GET` | `/interview/{interview_id}/summary` | Per-question results + average |
| `POST` | `/analysis/{answer_id}/content` | Analyse/reuse content statistics |
| `GET` | `/analysis/{answer_id}/content` | Retrieve saved content analysis |
| `POST` | `/analysis/{answer_id}/voice` | Measure/reuse saved recording or persist explicit status |
| `GET` | `/analysis/{answer_id}/voice` | Retrieve saved voice analysis |
| `GET` | `/interview/{interview_id}/analyses` | Both analysis types for one session |
| `GET` | `/prediction/schema` | Feature schema, model status, baseline methodology |
| `POST` | `/prediction/{answer_id}` | Create/refresh one answer prediction in place |
| `GET` | `/prediction/{answer_id}` | Retrieve the saved prediction (`is_stale` on read) |
| `GET` | `/interview/{interview_id}/predictions` | Predictions + model status for one session |

The evaluation endpoints operate only on saved transcripts, use the FastAPI
database dependency, and derive session/question ownership from the saved
answer — the client cannot attach an evaluation to a different session. Phase 3
endpoints follow the same rule and refuse to analyse an answer whose session is
missing. Phase 4 prediction endpoints likewise derive ownership from the saved
answer and never accept a client-supplied score, method, or model version.
Authentication is not implemented yet, so these routes do not restrict access by
user.

### Status codes

- `POST /evaluation/{id}`: `201` new evaluation, `200` reused or recomputed,
  `404` unknown answer/question, `409` missing reference answer, `503` semantic
  model/package unavailable, `422` invalid path values.
- `GET /evaluation/{id}`: `200`, `404` if no evaluation exists.
- `GET /interview/{id}/evaluations`: `200`, `404` unknown session.
- `GET /interview/{id}/summary`: `200`, `404` unknown session.
- `POST /analysis/{id}/content`: `201` new analysis, `200` reused or refreshed,
  `404` unknown answer/question or missing session link, `422` invalid IDs. A
  missing reference or unusable transcript is stored as an explicit status, not
  an HTTP error.
- `GET /analysis/{id}/content`: `200`, `404` if no analysis exists.
- `POST /analysis/{id}/voice`: `201` new analysis, `200` reused or refreshed,
  `404` unknown answer, `503` voice dependencies unavailable. Missing audio or an
  undecodable file is stored as `unavailable`/`failed` with `201`/`200` instead
  of fabricated values.
- `GET /analysis/{id}/voice`: `200`, `404` if no analysis exists.
- `GET /interview/{id}/analyses`: `200`, `404` unknown session.
- `POST /prediction/{id}`: `201` new prediction, `200` reused or refreshed,
  `404` unknown answer/question, `409` answer is not linked to a valid session.
  Missing/stale sources are stored as explicit `unavailable`/`failed` states, not
  fabricated scores.
- `GET /prediction/{id}`: `200`, `404` if no prediction exists.
- `GET /interview/{id}/predictions`: `200`, `404` unknown session.
- `GET /prediction/schema`: `200`; reports the installed artifact state without
  exposing local paths or secrets.

All three analysis types expose `is_stale`, computed on read from the current
transcript/reference/audio hashes. Repeating a POST for unchanged inputs returns
the saved row with `reused: true`; changed inputs update the same row in place.

### Example successful response

Scores depend on the transcript; this illustrates the response shape:

```json
{
  "evaluation": {
    "id": 12,
    "answer_id": 34,
    "interview_id": 5,
    "question_id": 2,
    "semantic_score": 78.42,
    "keyword_score": 64.29,
    "combined_score": 74.18,
    "matched_keywords": ["tcp", "reliable", "connection"],
    "missing_keywords": ["udp", "datagrams", "overhead"],
    "status": "evaluated",
    "status_detail": null,
    "methodology_version": "phase2-v1",
    "is_stale": false,
    "created_at": "2026-10-26T10:00:00Z",
    "updated_at": "2026-10-26T10:00:00Z"
  },
  "reused": false
}
```

### Example summary statuses

- `evaluated` — a valid score exists.
- `unevaluable` — a transcript exists but is empty/filler/too short; zero scores
  and an explanation are stored.
- `unevaluated` — a transcript exists but has not been evaluated.
- `not_evaluable` — the question has no reference answer.
- `stale` — transcript or reference changed after evaluation; re-evaluate.
- `unanswered` — no transcript saved for the question.

`average_score` is the mean of `combined_score` for **valid evaluated answers
only**. Unanswered, unevaluable, not-evaluable, and stale items are excluded. It
is `null` when there are no valid evaluations; sessions with no answers return
zeros/counts rather than dividing by zero.

### Example Phase 3 response shapes

Content analysis (`POST /analysis/{id}/content`):

```json
{
  "content_analysis": {
    "id": 7,
    "answer_id": 34,
    "coverage_score": 64.29,
    "coverage_source": "phase2_evaluation",
    "matched_concepts": ["tcp", "reliable"],
    "missing_concepts": ["udp", "datagrams"],
    "filler_word_count": 3,
    "filler_rate_percent": 4.11,
    "filler_words": [{"word": "um", "count": 2}],
    "repeated_terms": [{"term": "tcp", "count": 3}],
    "repetition_rate_percent": 2.74,
    "sentence_count": 4,
    "average_sentence_words": 18.25,
    "structure_notes": [],
    "status": "completed",
    "status_detail": null,
    "methodology_version": "phase3-v1",
    "is_stale": false
  },
  "reused": false
}
```

Voice analysis (`POST /analysis/{id}/voice`):

```json
{
  "voice_analysis": {
    "id": 9,
    "answer_id": 34,
    "status": "completed",
    "status_detail": null,
    "duration_seconds": 42.18,
    "speech_duration_seconds": 34.42,
    "silence_duration_seconds": 7.76,
    "speech_ratio_percent": 81.6,
    "pause_count": 5,
    "longest_pause_seconds": 1.42,
    "speaking_rate_wpm": 142.4,
    "rms_mean_dbfs": -22.7,
    "clipping_percent": 0.0,
    "pitch_reliable": true,
    "pitch_median_hz": 132.5,
    "pitch_range_hz": 48.1,
    "quality_notes": [],
    "methodology_version": "phase3-v1",
    "is_stale": false
  },
  "reused": false
}
```

Content statuses: `completed`, `unevaluable`, `not_evaluable`. Voice statuses:
`completed`, `insufficient_speech`, `unavailable`, `failed`. `null` metrics mean
"not available"; a numeric zero (for example `pause_count: 0`) is a real
measurement. `is_stale` is a read-time flag and does not delete or rewrite data.

### Example Phase 4 response shape

Baseline response (`POST /prediction/{id}`; values depend on the saved sources):

```json
{
  "prediction": {
    "id": 4,
    "answer_id": 34,
    "interview_id": 5,
    "question_id": 2,
    "status": "completed",
    "method": "rule_based_baseline",
    "method_label": "Rule-based baseline",
    "is_ml_prediction": false,
    "score": 71.42,
    "status_detail": "ML inference is unavailable, so this score is the rule-based baseline.",
    "model_version": null,
    "feature_schema_version": "phase4-features-v1",
    "methodology_version": "phase4-v1",
    "features": {
      "semantic_score": 78.42,
      "keyword_score": 64.29,
      "speaking_rate_wpm": null,
      "has_content_features": 1.0,
      "has_voice_features": 0.0
    },
    "missing_features": ["speaking_rate_wpm", "pauses_per_minute"],
    "components": {
      "components": [
        {"name": "answer", "score": 74.18, "weight": 0.6},
        {"name": "content_style", "score": 66.5, "weight": 0.2}
      ],
      "weights_renormalized": true,
      "effective_weight_total": 0.8
    },
    "warnings": ["Voice analysis is missing for this answer."],
    "limitations": ["This is a transparent rule-based baseline, not a machine-learning prediction."],
    "source_versions": {"evaluation_id": 12, "content_id": 7, "voice_id": null},
    "is_stale": false
  },
  "model_status": {
    "available": false,
    "state": "unavailable",
    "reason": "No ML model artifact is installed. Predictions use the clearly labelled rule-based baseline."
  },
  "reused": false
}
```

Prediction statuses are `completed`, `unavailable`, and `failed`; methods are
`ml_model`, `rule_based_baseline`, and `none`. `is_ml_prediction` is true only
for a validated model result. The interview summary additionally reports
`predictions_completed`, `predictions_ml`, `predictions_baseline`,
`predictions_stale`, `predictions_unavailable`, `average_predicted_score`,
`prediction_method`, `prediction_methodology_version`, and `ml_model_available`.
Only current, non-stale completed predictions contribute to the average; stale
or unavailable predictions are not counted as zero.

## 1. Configure the backend environment

The existing `backend/.env` is empty. Open it in VS Code and add the values below,
or copy the template first:

```powershell
Set-Location "C:\Users\DHANVINI\OneDrive\Desktop\AI interview evalutor\backend"
Copy-Item .env.example .env   # only safe while .env is still empty
notepad .env
```

Required values (replace the placeholders locally; never commit them):

```dotenv
DATABASE_URL=mysql+pymysql://YOUR_MYSQL_USER:YOUR_MYSQL_PASSWORD@127.0.0.1:3306/ai_interview_evaluator
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=whisper-large-v3-turbo
GROQ_TIMEOUT_SECONDS=60
MAX_AUDIO_MB=25
CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173

# Optional Phase 2 settings (defaults shown)
EVALUATION_SEMANTIC_WEIGHT=0.70
EVALUATION_KEYWORD_WEIGHT=0.30
EVALUATION_MIN_MEANINGFUL_TOKENS=3
SEMANTIC_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
SEMANTIC_MODEL_DEVICE=cpu

# Optional Phase 3 settings (defaults shown)
AUDIO_STORAGE_DIR=uploads
VOICE_MAX_ANALYSIS_SECONDS=300
VOICE_PITCH_MAX_SECONDS=30
VOICE_VAD_TOP_DB=35
VOICE_MIN_PAUSE_SECONDS=0.3

# Optional Phase 4 settings (defaults shown)
ML_MODEL_PATH=ml_artifacts/interview_score.joblib
ML_MIN_TRAINING_SAMPLES=60
ML_MIN_TRAINING_GROUPS=15
ML_TEST_FRACTION=0.25
ML_RANDOM_SEED=42
```

- `ML_MODEL_PATH` resolves relative to `backend/`. Keep model artifacts out of
  source control; no artifact is provided.
- The training thresholds only gate the `validated` label. `--allow-small` writes
  an `unvalidated` artifact that inference refuses.

- URL-encode special characters in the MySQL password. For example, `p@ss` becomes `p%40ss`.
- Create a Groq key at <https://console.groq.com/keys>.
- `whisper-large-v3-turbo` and `whisper-large-v3` are currently supported Groq Whisper models.
- The frontend defaults to `http://127.0.0.1:8000`; a `frontend/.env` file is optional.
- The first semantic evaluation needs internet access for the model download.
- `AUDIO_STORAGE_DIR` is resolved relative to `backend/` unless an absolute path
  is supplied. Keep it outside source control if you use a custom location.

## 2. Install dependencies

Backend (the existing virtualenv is used; no new Python version is required):

```powershell
Set-Location "C:\Users\DHANVINI\OneDrive\Desktop\AI interview evalutor\backend"
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# Optional, only needed to run the tests:
python -m pip install -r requirements-dev.txt
```

Phase 2 adds `scikit-learn==1.9.1` and `sentence-transformers==6.2.0`.
Phase 3 adds `librosa==1.0.0`, `soundfile==0.14.0`, and
`imageio-ffmpeg==0.6.0`. Phase 4 uses the existing scikit-learn install and
pins its `joblib` dependency explicitly (`joblib==1.6.0`); **no new package
needs to be downloaded for Phase 4**. Together with `sentence-transformers`
these bring compatible `numpy`, `scipy`, and audio-decoding dependencies; no
external AI API is used for evaluation, voice analysis, or prediction.

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

You can also skip activation and call the interpreter directly:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

Frontend (no new packages were required for Phases 2 or 3):

```powershell
Set-Location "C:\Users\DHANVINI\OneDrive\Desktop\AI interview evalutor\frontend"
npm install
```

## 3. Set up MySQL with VS Code SQLTools

1. Start the MySQL service (PowerShell as Administrator if needed):

   ```powershell
   Start-Service MySQL80
   Get-Service MySQL80
   ```

2. In VS Code, install these extensions:
   - **SQLTools** (`mtxr.sqltools`)
   - **SQLTools MySQL/MariaDB Driver** (`mtxr.sqltools-driver-mysql`)

3. Create a SQLTools connection:
   - Host: `127.0.0.1`
   - Port: `3306`
   - User: your MySQL user (often `root`)
   - Password: your local MySQL password
   - Database: leave empty for the first run, then use `ai_interview_evaluator`

4. Open `setup.sql`, connect with SQLTools, and run the whole file. It only runs
   `CREATE DATABASE IF NOT EXISTS`; it does **not** drop tables or delete data.

5. Start the backend (below). SQLAlchemy creates missing tables, adds the
   additive `reference_answer` and `audio_path`/`audio_sha256` columns when
   needed, and seeds questions/reference answers automatically. Then verify in
   SQLTools:

   ```sql
   USE ai_interview_evaluator;
   SHOW TABLES;
   SELECT id, question_text, category, reference_answer, created_at
   FROM interview_questions
   ORDER BY id;
   SELECT id, status, started_at FROM interview_sessions;
   SELECT id, interview_id, question_id, transcript, audio_path, audio_sha256,
          created_at
   FROM interview_answers
   ORDER BY id;
   SELECT id, answer_id, interview_id, question_id, semantic_score,
          keyword_score, combined_score, status, methodology_version,
          created_at, updated_at
   FROM answer_evaluations
   ORDER BY id;
   SELECT id, answer_id, coverage_score, coverage_source, filler_word_count,
          filler_rate_percent, repetition_rate_percent, status,
          methodology_version, created_at, updated_at
   FROM content_analyses
   ORDER BY id;
   SELECT id, answer_id, status, duration_seconds, speech_duration_seconds,
          pause_count, speaking_rate_wpm, pitch_reliable, pitch_median_hz,
          methodology_version, created_at, updated_at
   FROM voice_analyses
   ORDER BY id;
   SELECT id, answer_id, status, method, score, model_version,
          feature_schema_version, methodology_version, source_fingerprint,
          created_at, updated_at
   FROM score_predictions
   ORDER BY id;
   ```

   `audio_path` is relative to `AUDIO_STORAGE_DIR`; the matching file lives in
   `backend/uploads/` unless that setting was changed.

6. Re-check seeding safely at any time:

   ```powershell
   Set-Location "...\backend"
   .\venv\Scripts\python.exe seed.py
   ```

   It creates missing tables/questions, backfills only missing reference
   answers, and reports how many questions are evaluable.

The backend needs permission to create tables, alter the questions table, and
read/write rows. If you use a restricted user, grant those permissions on
`ai_interview_evaluator.*`.

## 4. Run the backend

Open **PowerShell terminal 1**:

```powershell
Set-Location "C:\Users\DHANVINI\OneDrive\Desktop\AI interview evalutor\backend"
.\venv\Scripts\Activate.ps1
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Or without activation:

```powershell
Set-Location "C:\Users\DHANVINI\OneDrive\Desktop\AI interview evalutor\backend"
.\venv\Scripts\python.exe -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Useful URLs:

- <http://127.0.0.1:8000/>
- <http://127.0.0.1:8000/health>
- <http://127.0.0.1:8000/docs>
- <http://127.0.0.1:8000/interview/questions>

The backend intentionally stays up if MySQL is temporarily unavailable; `/health`
and data endpoints then return a safe `503` message. When the database check
fails, `/health` also includes `database_error_category` (for example
`authentication_failed`, `unknown_database`, or `connection_failed`),
`database_hint`, `database_url_source`, and `env_file_found`. It never includes
the password or raw driver text.

## 5. Run the frontend

Open **PowerShell terminal 2**:

```powershell
Set-Location "C:\Users\DHANVINI\OneDrive\Desktop\AI interview evalutor\frontend"
npm run dev
```

Open the URL Vite prints (normally <http://localhost:5173>). The default API URL
is `http://127.0.0.1:8000`. If Vite chooses a different port, add its exact origin
to `CORS_ORIGINS` in `backend/.env` and restart FastAPI.

## 6. Tests and checks

Run from `backend`:

```powershell
.\venv\Scripts\python.exe -m compileall -q main.py config.py database.py models.py schemas.py seed.py routes services tests
.\venv\Scripts\python.exe -m pytest tests -q
```

Run the opt-in real Sentence-Transformers integration test (downloads/loads the
model on first use):

```powershell
$env:RUN_SEMANTIC_MODEL_INTEGRATION = "1"
.\venv\Scripts\python.exe -m pytest tests/test_phase2.py::test_real_semantic_model_integration -q -s
Remove-Item Env:\RUN_SEMANTIC_MODEL_INTEGRATION
```

Run from `frontend`:

```powershell
npm run lint
npm run build
```

The default pytest suite uses an isolated in-memory SQLite database, a mocked
transcription function, a mocked semantic score, a temporary audio-storage
directory, and synthetic WAV files. It does not require MySQL, a Groq key, a
microphone, a labelled dataset, or a model download. It covers preprocessing,
keyword weighting, scoring boundaries, unusable/missing-reference answers,
persistence, idempotency, stale detection, invalid IDs, model-failure handling,
and summary calculations, plus Phase 3 audio storage/hashing, content
statistics, voice measurements, missing/corrupt audio, FFmpeg fallback, silence
handling, and Phase 2 score preservation. Phase 4 tests cover the versioned
feature schema, artifact loading/version mismatch/invalid artifacts,
missing/stale sources, unavailable voice (never zero-filled), inference bounds,
model failure, duplicate/rescoring behavior, stale predictions, API errors,
baseline labeling, group-split leakage, small-data refusal, and validated
training metrics on synthetic temporary data. The real-model test is separate
and clearly marked skipped unless the environment variable is set.

## 7. Troubleshooting

- **`Could not load interview questions` / HTTP 503**: check `DATABASE_URL`, the
  MySQL service, and SQLTools connectivity. The frontend shows the backend's safe
  message. The backend log intentionally does not print secrets or driver details.
- **`Access denied for user`**: `/health` shows
  `"database_error_category": "authentication_failed"`. `backend/.env` is empty
  or has the wrong MySQL credentials; set `DATABASE_URL` with a URL-encoded
  password. `"database_url_source": "default"` means no `.env` or process
  environment value was loaded.
- **`"database_error_category": "unknown_database"`**: run `setup.sql` (it only
  creates the database) or point `DATABASE_URL` at an existing database.
- **`"database_error_category": "connection_failed"`**: start the MySQL service
  and confirm the host and port in `DATABASE_URL`.
- **`Groq key missing` badge / HTTP 503 on submit**: add `GROQ_API_KEY` to
  `backend/.env` and restart FastAPI.
- **First evaluation is slow / HTTP 503 "model could not be loaded"**: the first
  use downloads `all-MiniLM-L6-v2` (about 90 MB) and needs internet access. Check
  the connection/proxy, then retry. Later evaluations reuse the in-process model.
- **`Unevaluable` status**: the transcript was empty, filler-only, or too short.
  Re-record a complete answer; no zero-quality score is fabricated.
- **`Not evaluable` status**: the question has no reference answer. Add one in
  `interview_questions.reference_answer` (or update the seed data).
- **`Needs re-evaluation` / stale badge**: the transcript or reference changed
  after evaluation. Click **Re-evaluate** to refresh the same evaluation row.
- **`Voice analysis unavailable` badge / HTTP 503 on voice analysis**: the audio
  stack is missing. Install the Phase 3 requirements into the active venv
  (`python -m pip install -r requirements.txt`), confirm `librosa`, `soundfile`,
  and `imageio-ffmpeg` import, then restart FastAPI. `/health` reports
  `voice_analysis_available`.
- **`Audio unavailable` voice status**: this answer has no stored recording.
  Answers transcribed before Phase 3, or recordings whose upload failed, only
  have the transcript. Record or upload the answer again to measure voice.
  Metrics stay `null` rather than being invented.
- **`Analysis failed` voice status**: the stored file could not be decoded, or
  its SHA-256 no longer matches. Check the file in `backend/uploads/`, then
  re-record the answer. The backend log records the reason category without
  printing audio contents.
- **FFmpeg fallback messages**: `soundfile` cannot read some M4A/WebM/MP3
  containers. The bundled FFmpeg binary from `imageio-ffmpeg` is used
  automatically; sample-rate/channel fields are then reported as unknown instead
  of guessed. No separate system FFmpeg install is required.
- **`backend/uploads/` is missing or not writable**: it is created on demand
  relative to `AUDIO_STORAGE_DIR`. Grant the backend user write access, or point
  `AUDIO_STORAGE_DIR` at another local directory. A storage failure does not
  discard a successful transcript, but voice analysis will report
  `unavailable`.
- **`No reference answer` content status**: transcript statistics are still
  stored and shown; coverage is `null`. Add a reference answer to enable
  coverage.
- **`Transcript not evaluable` content status**: the transcript is empty,
  filler-only, or too short for meaningful statistics. Re-record a complete
  answer; no coverage score is fabricated.
- **`Baseline mode` / `ML model unavailable`**: this is expected on a fresh
  checkout. No labelled dataset or model artifact is included. Predictions use
  the clearly labelled rule-based baseline; a score is not an ML prediction.
- **`unvalidated` model artifact**: training did not meet the thresholds or did
  not beat the mean baseline, or `--allow-small` was used. Inference deliberately
  refuses it. Train with a larger, representative labelled dataset or keep using
  the baseline.
- **`incompatible` model artifact**: the feature schema/order or scikit-learn
  major/minor version differs from this server. Retrain with the installed
  scikit-learn version and this feature schema; do not hand-edit the bundle.
- **`invalid` model artifact**: the joblib file is corrupt or missing required
  metadata. Delete it and retrain; the API reports the state without crashing.
- **Training refused**: the CSV is missing columns, contains unknown/duplicate
  rows, has out-of-range values, or has fewer than
  `ML_MIN_TRAINING_SAMPLES`/`ML_MIN_TRAINING_GROUPS`. Read the error; use
  `--allow-small` only for an explicitly unvalidated experiment.
- **`Prediction unavailable`**: the Phase 2 evaluation is missing, unevaluable,
  or stale, or the required semantic/keyword features are unavailable. Re-evaluate
  the answer, then predict again.
- **`Prediction stale`**: the transcript, reference answer, Phase 2 evaluation, or
  Phase 3 analysis changed after the prediction. Click **Re-predict score** to
  refresh the same row; the old score is not silently reused.
- **`Prediction failed`**: a validated model raised an error or returned a
  non-finite value. The API does not hide this by substituting a heuristic; check
  the backend log and retrain/reinstall a valid artifact.
- **Microphone permission denied**: allow the microphone for the site in the
  browser, enable microphone access in Windows Privacy settings, and use
  `http://localhost:5173` or HTTPS. Plain HTTP on a LAN IP is usually blocked.
- **CORS error**: start Vite on one of the origins in `CORS_ORIGINS`, or add the
  actual origin and restart FastAPI.
- **Recording too large**: keep recordings/uploads under `MAX_AUDIO_MB` (25 MB by
  default, which is Groq's limit).
- **No speech detected**: the Groq response was empty; re-record and speak clearly.

## 8. Browser checklist

1. MySQL is running and `ai_interview_evaluator` exists.
2. `backend/.env` has the MySQL credentials and Groq key.
3. FastAPI starts and `/health` shows `database: connected` and
   `transcription_configured: true`.
4. React opens and shows the seeded questions with categories.
5. The status badges show API, MySQL, Groq, voice analysis, and ML model status
   (baseline mode on a fresh checkout); the footer says Phase 4.
6. Click **Start interview** and confirm a session number appears.
7. Move between questions; navigation is locked while recording.
8. Allow microphone permission, click **Start recording**, speak, then **Stop recording**. Alternatively, click **Upload audio file** and choose a supported file.
9. Review the audio, click **Submit & transcribe**, and wait for the loading state.
10. The transcript appears under **Saved transcript**. Refresh the browser and confirm the same session and transcript are restored from MySQL.
11. An **Automated evaluation** card appears under the transcript; click
    **Evaluate answer** and confirm the loading state prevents double clicks.
12. Confirm semantic, keyword, and combined scores (each 0–100, two decimals),
    plus matched/missing keyword chips.
13. Confirm a short/filler answer shows **Unevaluable** with an explanation and
    no misleading score.
14. Confirm a question without a reference answer shows **Not evaluable** and
    offers no evaluation button.
15. Click **Evaluate again** and confirm the saved result is reused/refreshed
    rather than duplicated.
16. Change the reference answer in MySQL, refresh, and confirm the evaluation is
    marked **Needs re-evaluation**; click **Re-evaluate**.
17. In **Saved answers**, confirm evaluate/re-evaluate actions, status badges, and
    saved scores distinguish not evaluated / evaluated / unevaluable.
18. Check **Interview summary** for question count, answers saved, evaluated,
    unevaluable, average score, and per-question scores/keywords.
19. Confirm the average excludes unanswered/unevaluable/stale answers and is
    shown as `—` when no valid evaluations exist.
20. Click **Refresh** in both sections and confirm results persist from MySQL.
21. Check `answer_evaluations` in SQLTools to confirm the same scores and keyword
    lists are stored.
22. Confirm a **Phase 3 · Content** card appears under the evaluation card and a
    **Phase 3 · Voice** card appears below it. Neither replaces or changes the
    Phase 2 score.
23. Click **Analyse content** and confirm coverage, filler-word share, repeated
    terms, average sentence length, chip lists, and structure notes appear.
    Coverage should equal the Phase 2 keyword score when that evaluation exists.
24. Click **Analyse voice** on a newly recorded answer and confirm duration,
    detected speech, pause count, speaking rate, energy, clipping, and pitch
    appear. Pitch shows **Not reliable** instead of a guessed value when voiced
    audio is insufficient.
25. Confirm the voice card says the measurements are acoustic only and makes no
    claim about confidence, personality, honesty, intelligence, or emotion.
26. Click **Analyse again** on both cards and confirm each result is reused
    (same saved row) rather than duplicated.
27. Change the transcript or reference answer, refresh, and confirm the affected
    card shows **Needs re-analysis**; re-analyse and confirm it updates in place.
28. Open an answer with no stored recording (for example one transcribed before
    Phase 3) and confirm **Audio unavailable** with `null` metrics, not zeros.
29. Upload a corrupt audio file, attach it, and confirm voice analysis reports
    **Analysis failed** instead of crashing or inventing measurements.
30. In **Saved answers**, confirm each entry shows Content and Voice status text.
    Then check `content_analyses` and `voice_analyses` in SQLTools and confirm
    `backend/uploads/` contains the matching content-addressed recordings.
31. Confirm a **Phase 4 · Prediction** card appears below the voice card. With no
    validated model artifact installed it should say predictions use the
    rule-based baseline and must **not** call the result an ML prediction.
32. Click **Predict score**. Confirm the card shows a 0–100 score, the method
    label, available baseline components/weights, missing features (excluded, not
    zero), warnings, limitations, and the saved methodology version.
33. Confirm an answer without a Phase 2 evaluation shows **Prediction
    unavailable** and no score, rather than a fabricated value.
34. Confirm voice-dependent components are absent when voice analysis is missing
    or `insufficient_speech`, and the remaining weights are renormalized.
35. Change the transcript and refresh: the prediction should be marked stale in
    the card and in **Saved answers**. Click **Re-predict score** and confirm the
    same prediction row is updated, not duplicated.
36. Check **Interview summary** for predicted/ML/baseline/stale/unavailable
    counts and the average predicted score. Unavailable and stale predictions
    must not be counted as zero. If no validated model is installed, the summary
    must label completed scores as the rule-based baseline.
