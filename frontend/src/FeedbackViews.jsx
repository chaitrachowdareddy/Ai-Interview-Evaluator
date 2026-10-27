import {
  feedbackBandMeta,
  feedbackMethodLabel,
  feedbackStatusMeta,
} from './feedbackStatus'

function formatScore(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return '—'
  }
  return Number(value).toFixed(2)
}

function formatWhen(value) {
  if (!value) return ''
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return ''
  return date.toLocaleString()
}

function StatusBadge({ status, isStale = false }) {
  const meta = feedbackStatusMeta(status, { isStale })
  return <span className={`badge badge-${meta.tone}`}>{meta.label}</span>
}

function BandBadge({ band }) {
  const meta = feedbackBandMeta(band)
  if (!meta) return null
  return <span className={`badge badge-${meta.tone}`}>{meta.label}</span>
}

function EvidenceLabels({ feedback }) {
  return (
    <div className="feedback-evidence" aria-label="Evidence sources">
      <span className="badge badge-neutral">Content evaluation</span>
      <span
        className={`badge ${
          feedback.content_available ? 'badge-ok' : 'badge-warn'
        }`}
      >
        {feedback.content_available
          ? 'Content analysis included'
          : 'Content analysis unavailable'}
      </span>
      <span
        className={`badge ${
          feedback.voice_available ? 'badge-ok' : 'badge-neutral'
        }`}
      >
        {feedback.voice_available
          ? 'Voice measurements included'
          : 'Voice unavailable'}
      </span>
      <span
        className={`badge ${
          feedback.prediction_is_ml
            ? 'badge-ok'
            : feedback.prediction_method === 'rule_based_baseline'
              ? 'badge-warn'
              : 'badge-neutral'
        }`}
      >
        {feedbackMethodLabel(
          feedback.prediction_method,
          feedback.prediction_is_ml,
        )}
      </span>
    </div>
  )
}

function FeedbackList({ title, items, tone = 'neutral' }) {
  if (!items || items.length === 0) return null
  return (
    <div className={`feedback-list feedback-list-${tone}`}>
      <h4>{title}</h4>
      <ul>
        {items.map((item, index) => (
          <li key={`${title}-${index}`}>{item}</li>
        ))}
      </ul>
    </div>
  )
}

export function FeedbackScoreGrid({ feedback }) {
  const cells = [
    ['Semantic', feedback.semantic_score],
    ['Keyword', feedback.keyword_score],
    ['Combined', feedback.combined_score],
    ['Content coverage', feedback.content_coverage_score],
    ['Predicted', feedback.prediction_score],
  ]

  return (
    <div className="score-grid feedback-score-grid">
      {cells.map(([label, value]) => (
        <div className="score-cell" key={label}>
          <span>{label}</span>
          <strong>{formatScore(value)}</strong>
        </div>
      ))}
    </div>
  )
}

export function AnswerFeedbackCard({
  answer,
  feedback,
  loading = false,
  error = '',
  disabled = false,
  onGenerate,
}) {
  if (!answer) return null

  const stale = feedback?.is_stale === true
  const hasFeedback = Boolean(feedback)

  return (
    <section className="feedback-block" aria-label="Answer feedback">
      <div className="evaluation-head">
        <div>
          <span className="section-kicker">Personalised feedback</span>
          <h3>Question feedback</h3>
        </div>
        <div className="feedback-head-actions">
          {feedback ? (
            <StatusBadge status={feedback.status} isStale={stale} />
          ) : (
            <span className="badge">Not generated</span>
          )}
          {feedback ? <BandBadge band={feedback.band} /> : null}
          <button
            type="button"
            className="button button-secondary button-small"
            onClick={() => onGenerate(answer)}
            disabled={loading || disabled}
          >
            {loading
              ? 'Generating…'
              : hasFeedback
                ? 'Refresh feedback'
                : 'Generate feedback'}
          </button>
        </div>
      </div>

      {error ? (
        <div className="alert alert-error" role="alert">
          {error}
        </div>
      ) : null}

      {loading && !hasFeedback ? (
        <div className="evaluation-loading" role="status">
          <span className="spinner" aria-hidden="true" />
          Generating evidence-based feedback from saved results…
        </div>
      ) : null}

      {!hasFeedback && !loading && !error ? (
        <p className="muted">
          No feedback has been saved for this answer. Generate it after the
          answer has a current Phase 2 evaluation. Missing voice or prediction
          evidence is reported as unavailable, never as zero.
        </p>
      ) : null}

      {hasFeedback ? (
        <>
          {stale ? (
            <div className="alert alert-warning" role="status">
              The transcript, reference answer, evaluation, analysis, or
              prediction changed after this feedback was saved. Refresh it
              before relying on these results.
            </div>
          ) : null}

          <p className="feedback-summary">{feedback.summary}</p>

          <EvidenceLabels feedback={feedback} />

          {feedback.combined_score !== null ||
          feedback.prediction_score !== null ? (
            <FeedbackScoreGrid feedback={feedback} />
          ) : null}

          {feedback.status_detail ? (
            <div className="feedback-status-detail">
              <strong>Why:</strong> {feedback.status_detail}
            </div>
          ) : null}

          <FeedbackList
            title="Strengths"
            items={feedback.strengths}
            tone="positive"
          />
          <FeedbackList
            title="Improvements"
            items={feedback.improvements}
            tone="warning"
          />
          <FeedbackList
            title="Actionable suggestions"
            items={feedback.actionable_suggestions}
            tone="action"
          />
          <FeedbackList
            title="Missing concepts"
            items={feedback.missing_concepts}
            tone="warning"
          />
          <FeedbackList
            title="Objective voice observations"
            items={feedback.voice_observations}
            tone="neutral"
          />

          {feedback.prediction_note ? (
            <p className="feedback-prediction-note">
              <span
                className={`badge ${
                  feedback.prediction_is_ml ? 'badge-ok' : 'badge-warn'
                }`}
              >
                {feedbackMethodLabel(
                  feedback.prediction_method,
                  feedback.prediction_is_ml,
                )}
              </span>{' '}
              {feedback.prediction_note}
            </p>
          ) : null}

          {feedback.notices?.length ? (
            <details className="feedback-details">
              <summary>Evidence notices and exclusions</summary>
              <ul>
                {feedback.notices.map((item, index) => (
                  <li key={`notice-${index}`}>{item}</li>
                ))}
              </ul>
            </details>
          ) : null}

          {feedback.limitations?.length ? (
            <details className="feedback-details">
              <summary>Limitations</summary>
              <ul>
                {feedback.limitations.map((item, index) => (
                  <li key={`limitation-${index}`}>{item}</li>
                ))}
              </ul>
            </details>
          ) : null}

          <p className="muted small">
            Saved {formatWhen(feedback.updated_at) || 'recently'} · methodology{' '}
            <code>{feedback.methodology_version}</code>
          </p>
        </>
      ) : null}
    </section>
  )
}

export function SavedAnswerFeedback({
  feedback,
  loading = false,
  disabled = false,
  onGenerate,
  answer,
}) {
  return (
    <div className="saved-feedback-state">
      <span className="muted small">
        Feedback:{' '}
        {feedback ? (
          <StatusBadge status={feedback.status} isStale={feedback.is_stale} />
        ) : (
          'not generated'
        )}
      </span>
      {feedback?.combined_score !== null &&
      feedback?.combined_score !== undefined ? (
        <span className="muted small">
          Combined {formatScore(feedback.combined_score)}
        </span>
      ) : null}
      <button
        type="button"
        className="button button-secondary button-small"
        onClick={() => onGenerate(answer)}
        disabled={loading || disabled}
      >
        {loading ? 'Generating…' : feedback ? 'Refresh' : 'Generate feedback'}
      </button>
    </div>
  )
}

function BreakdownItem({ item }) {
  const stale = item.status === 'stale'
  return (
    <article className="results-question">
      <div className="results-question-head">
        <span className="badge category-badge">{item.category}</span>
        <span className="muted small">Question {item.question_number}</span>
        <StatusBadge status={item.status} isStale={stale} />
        <BandBadge band={item.band} />
      </div>
      <h4>{item.question_text}</h4>

      <div className="results-question-labels">
        <span className="badge badge-neutral">
          Combined: {formatScore(item.combined_score)}
        </span>
        <span
          className={`badge ${
            item.voice_available ? 'badge-ok' : 'badge-neutral'
          }`}
        >
          {item.voice_available ? 'Voice measured' : 'Voice unavailable'}
        </span>
        <span
          className={`badge ${
            item.prediction_is_ml
              ? 'badge-ok'
              : item.prediction_method === 'rule_based_baseline'
                ? 'badge-warn'
                : 'badge-neutral'
          }`}
        >
          {feedbackMethodLabel(
            item.prediction_method,
            item.prediction_is_ml,
          )}
        </span>
      </div>

      {item.summary ? <p>{item.summary}</p> : null}
      {item.status_detail ? (
        <p className="muted small">{item.status_detail}</p>
      ) : null}

      <FeedbackList title="Strengths" items={item.strengths} tone="positive" />
      <FeedbackList
        title="Improvements"
        items={item.improvements}
        tone="warning"
      />
      {item.missing_concepts?.length ? (
        <p className="muted small">
          Missing concepts: {item.missing_concepts.join(', ')}
        </p>
      ) : null}
    </article>
  )
}

export function InterviewResultsDashboard({
  feedback,
  loading = false,
  generating = false,
  error = '',
  canGenerate = false,
  onGenerate,
}) {
  const stale = feedback?.is_stale === true
  const partial = feedback?.status === 'partial'

  return (
    <section className="card results-dashboard" aria-label="Interview results">
      <div className="panel-heading">
        <div>
          <span className="section-kicker">Results dashboard</span>
          <h2>Interview feedback</h2>
        </div>
        <div className="results-head-actions">
          {feedback ? (
            <StatusBadge status={feedback.status} isStale={stale} />
          ) : null}
          <button
            type="button"
            className="button button-primary"
            onClick={onGenerate}
            disabled={!canGenerate || generating}
          >
            {generating
              ? 'Generating feedback…'
              : feedback
                ? 'Refresh interview feedback'
                : 'Generate interview feedback'}
          </button>
        </div>
      </div>

      {error ? (
        <div className="alert alert-error" role="alert">
          {error}
        </div>
      ) : null}

      {loading && !feedback ? (
        <div className="evaluation-loading" role="status">
          <span className="spinner" aria-hidden="true" />
          Loading saved feedback and question-wise results…
        </div>
      ) : null}

      {!canGenerate && !feedback ? (
        <p className="muted">
          Start an interview and save at least one answer to generate feedback.
        </p>
      ) : null}

      {canGenerate && !feedback && !loading && !error ? (
        <div className="results-empty">
          <h3>No interview feedback saved yet</h3>
          <p className="muted">
            Generate feedback to combine current per-answer evidence into an
            interview-level summary. Unanswered questions and unavailable
            sources are excluded from averages, never scored as zero.
          </p>
        </div>
      ) : null}

      {feedback ? (
        <>
          {stale ? (
            <div className="alert alert-warning" role="status">
              A saved answer, reference answer, evaluation, analysis, or
              prediction changed after this interview feedback was generated.
              Refresh it before relying on the summary.
            </div>
          ) : null}

          {partial ? (
            <div className="alert alert-warning" role="status">
              Partial results: some questions are unanswered or some saved
              answers have unavailable, failed, or stale feedback. Those
              questions are excluded from averages, not scored as zero.
            </div>
          ) : null}

          <p className="results-summary">{feedback.summary}</p>

          <div className="results-metrics">
            <div className="results-metric">
              <span>Average combined score</span>
              <strong>{formatScore(feedback.average_score)}</strong>
            </div>
            <div className="results-metric">
              <span>Completed feedback</span>
              <strong>
                {feedback.feedback_completed}/{feedback.questions_total}
              </strong>
            </div>
            <div className="results-metric">
              <span>Answered / unanswered</span>
              <strong>
                {feedback.questions_answered} / {feedback.questions_unanswered}
              </strong>
            </div>
            <div className="results-metric">
              <span>Voice measurements</span>
              <strong>{feedback.voice_available_count}</strong>
            </div>
            <div className="results-metric">
              <span>ML / baseline predictions</span>
              <strong>
                {feedback.ml_predictions} / {feedback.baseline_predictions}
              </strong>
            </div>
            <div className="results-metric">
              <span>Average predicted score</span>
              <strong>{formatScore(feedback.average_predicted_score)}</strong>
            </div>
          </div>

          <div className="results-columns">
            <FeedbackList
              title="Overall strengths"
              items={feedback.overall_strengths}
              tone="positive"
            />
            <FeedbackList
              title="Overall improvements"
              items={feedback.overall_improvements}
              tone="warning"
            />
          </div>
          <FeedbackList
            title="Recommended next actions"
            items={feedback.recommendations}
            tone="action"
          />

          <div className="results-breakdown">
            <h3>Question-wise breakdown</h3>
            <div className="results-question-list">
              {feedback.question_breakdown.map((item) => (
                <BreakdownItem key={item.question_id} item={item} />
              ))}
            </div>
          </div>

          {feedback.notices?.length ? (
            <details className="feedback-details" open>
              <summary>Evidence notices and exclusions</summary>
              <ul>
                {feedback.notices.map((item, index) => (
                  <li key={`interview-notice-${index}`}>{item}</li>
                ))}
              </ul>
            </details>
          ) : null}

          {feedback.limitations?.length ? (
            <details className="feedback-details">
              <summary>Limitations</summary>
              <ul>
                {feedback.limitations.map((item, index) => (
                  <li key={`interview-limitation-${index}`}>{item}</li>
                ))}
              </ul>
            </details>
          ) : null}

          <p className="muted small">
            Generated {formatWhen(feedback.updated_at) || 'recently'} ·
            methodology <code>{feedback.methodology_version}</code>
          </p>
        </>
      ) : null}
    </section>
  )
}
