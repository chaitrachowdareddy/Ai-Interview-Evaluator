import { predictionStatusMeta } from './predictionStatus'

const STATUS_META = {
  evaluated: { label: 'Evaluated', tone: 'ok' },
  unevaluable: { label: 'Unevaluable', tone: 'warn' },
  unevaluated: { label: 'Not evaluated', tone: 'neutral' },
  not_evaluable: { label: 'Not evaluable', tone: 'error' },
  stale: { label: 'Needs re-evaluation', tone: 'warn' },
  unanswered: { label: 'Unanswered', tone: 'neutral' },
}

function statusMeta(status) {
  return STATUS_META[status] || { label: status || 'Unknown', tone: 'neutral' }
}

export function EvaluationStatusBadge({ status }) {
  const meta = statusMeta(status)
  return (
    <span className={`badge badge-${meta.tone}`}>
      {meta.label}
    </span>
  )
}

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
  return date.toLocaleString([], {
    dateStyle: 'medium',
    timeStyle: 'short',
  })
}

function Score({ label, value }) {
  return (
    <div className="score-cell">
      <span>{label}</span>
      <strong>{formatScore(value)}</strong>
      <span className="score-scale">/ 100</span>
    </div>
  )
}

function KeywordChips({ title, items, tone }) {
  return (
    <div className="keyword-group">
      <span className="keyword-title">{title}</span>
      {items?.length ? (
        <ul className="keyword-list">
          {items.map((keyword) => (
            <li className={`keyword-chip ${tone}`} key={keyword}>
              {keyword}
            </li>
          ))}
        </ul>
      ) : (
        <span className="muted small">None</span>
      )}
    </div>
  )
}

export function EvaluationCard({
  answer,
  evaluation,
  evaluating,
  disabled,
  evaluable = true,
  onEvaluate,
}) {
  if (!answer) return null

  const isStale = evaluation?.is_stale === true
  const isUnevaluable = evaluation?.status === 'unevaluable'
  const status = isStale ? 'stale' : evaluation?.status

  return (
    <section className="evaluation-block" aria-busy={evaluating}>
      <div className="evaluation-head">
        <div>
          <span className="section-kicker">Automated evaluation</span>
          <h3>Answer comparison</h3>
        </div>
        {evaluation ? (
          <EvaluationStatusBadge status={status} />
        ) : (
          <span className="badge">Not evaluated</span>
        )}
      </div>

      {evaluating ? (
        <div className="evaluation-loading" role="status">
          <span className="spinner" aria-hidden="true" />
          <span>Comparing this transcript with the reference answer…</span>
        </div>
      ) : null}

      {!evaluation && !evaluating ? (
        <>
          <p className="muted">
            Compare this saved transcript with the question&apos;s reference
            answer. The score combines semantic similarity (70%) and weighted
            keyword coverage (30%).
          </p>
          {evaluable ? (
            <button
              type="button"
              className="button button-primary"
              onClick={() => onEvaluate(answer)}
              disabled={disabled}
            >
              Evaluate answer
            </button>
          ) : (
            <div className="alert alert-warning">
              This question has no reference answer, so this transcript cannot
              be evaluated.
            </div>
          )}
        </>
      ) : null}

      {evaluation && !evaluating ? (
        <>
          {isStale ? (
            <div className="alert alert-warning">
              The transcript or reference answer changed after this evaluation.
              The saved scores are outdated; re-evaluate to refresh them.
            </div>
          ) : null}

          {isUnevaluable ? (
            <div className="alert alert-warning">
              This answer was not scored. {evaluation.status_detail}
            </div>
          ) : (
            <div className="score-grid">
              <Score label="Semantic similarity" value={evaluation.semantic_score} />
              <Score label="Keyword coverage" value={evaluation.keyword_score} />
              <Score label="Combined score" value={evaluation.combined_score} />
            </div>
          )}

          <KeywordChips
            title="Matched keywords"
            items={evaluation.matched_keywords}
            tone="matched"
          />
          <KeywordChips
            title="Missing keywords"
            items={evaluation.missing_keywords}
            tone="missing"
          />

          <div className="evaluation-footer">
            <span className="muted small">
              Saved {formatWhen(evaluation.updated_at)} · methodology{' '}
              {evaluation.methodology_version}
            </span>
            <button
              type="button"
              className="button button-secondary button-small"
              onClick={() => onEvaluate(answer)}
              disabled={disabled || !evaluable}
            >
              {isStale ? 'Re-evaluate answer' : 'Evaluate again'}
            </button>
          </div>
        </>
      ) : null}

      <p className="fine-print">
        This is an automated comparison against one reference answer, not a
        hiring decision or proof of knowledge. Semantic similarity is not a
        probability of correctness.
      </p>
    </section>
  )
}

function SummaryStat({ label, value }) {
  return (
    <div className="summary-stat">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  )
}

function summaryPredictionLabel(method) {
  if (method === 'mixed') return 'Mixed methods'
  if (method === 'ml_model') return 'ML prediction'
  if (method === 'rule_based_baseline') return 'Rule-based baseline'
  return 'No predictions'
}

export function InterviewSummary({
  summary,
  loading,
  error,
  canRefresh = true,
  onRefresh,
}) {
  return (
    <section className="card summary-section">
      <div className="panel-heading">
        <div>
          <span className="section-kicker">Phase 2–4 results</span>
          <h2>Interview summary</h2>
        </div>
        <button
          type="button"
          className="button button-secondary button-small"
          onClick={onRefresh}
          disabled={!canRefresh || loading}
        >
          {loading ? 'Refreshing…' : 'Refresh'}
        </button>
      </div>

      {error ? (
        <div className="alert alert-error" role="alert">
          {error}
        </div>
      ) : null}

      {!summary ? (
        loading ? (
          <div className="evaluation-loading" role="status">
            <span className="spinner" aria-hidden="true" />
            <span>Loading saved evaluations and summary…</span>
          </div>
        ) : (
          <p className="muted">
            Start an interview session and evaluate answers to see a summary.
          </p>
        )
      ) : (
        <>
          <div className="summary-stats">
            <SummaryStat label="Questions" value={summary.total_questions} />
            <SummaryStat label="Answers saved" value={summary.answers_submitted} />
            <SummaryStat label="Evaluated" value={summary.answers_evaluated} />
            <SummaryStat label="Unevaluable" value={summary.answers_unevaluable} />
            <SummaryStat
              label="Average score"
              value={
                summary.average_score == null
                  ? '—'
                  : `${formatScore(summary.average_score)} / 100`
              }
            />
          </div>

          <p className="fine-print">
            The average uses only currently valid evaluated answers. Unanswered
            or unevaluable questions are not counted as zero.
          </p>

          {summary.prediction_methodology_version ? (
            <div className="prediction-summary-block">
              <div className="prediction-summary-head">
                <div>
                  <span className="section-kicker">Phase 4 · Predictions</span>
                  <h3>Score predictions</h3>
                </div>
                <span
                  className={`badge badge-${
                    summary.prediction_method === 'ml_model'
                      ? 'ok'
                      : summary.prediction_method
                        ? 'warn'
                        : 'neutral'
                  }`}
                >
                  {summaryPredictionLabel(summary.prediction_method)}
                </span>
              </div>

              <div className="summary-stats">
                <SummaryStat
                  label="Predicted"
                  value={summary.predictions_completed}
                />
                <SummaryStat
                  label="ML predictions"
                  value={summary.predictions_ml}
                />
                <SummaryStat
                  label="Baseline"
                  value={summary.predictions_baseline}
                />
                <SummaryStat
                  label="Stale"
                  value={summary.predictions_stale}
                />
                <SummaryStat
                  label="Unavailable"
                  value={summary.predictions_unavailable}
                />
                <SummaryStat
                  label="Average prediction"
                  value={
                    summary.average_predicted_score == null
                      ? '—'
                      : `${formatScore(summary.average_predicted_score)} / 100`
                  }
                />
              </div>

              <p className="fine-print">
                Predictions are additive and never replace Phase 2 scores or
                Phase 3 analyses. Only current, non-stale completed predictions
                are averaged; unavailable or stale predictions are not counted
                as zero.{' '}
                {summary.ml_model_available
                  ? 'A validated ML model artifact is installed.'
                  : 'No validated ML model artifact is installed, so completed scores use the clearly labelled rule-based baseline, not an ML prediction.'}
              </p>
            </div>
          ) : null}

          <div className="summary-list">
            {summary.questions.map((item, index) => {
              const tone = statusMeta(item.status).tone
              const showScores =
                item.status === 'evaluated' ||
                item.status === 'stale' ||
                item.status === 'unevaluable'

              const predictionMeta = item.prediction_status
                ? predictionStatusMeta(item.prediction_status, {
                    isStale: item.prediction_is_stale,
                    method: item.prediction_method,
                  })
                : null

              return (
                <article
                  className={`summary-question ${tone}`}
                  key={item.question_id}
                >
                  <div className="summary-question-head">
                    <div>
                      <span className="section-kicker">
                        Question {index + 1} · {item.category}
                      </span>
                      <h3>{item.question_text}</h3>
                    </div>
                    <div className="summary-question-result">
                      <EvaluationStatusBadge status={item.status} />
                      <strong>
                        {item.status === 'unevaluable'
                          ? 'Not scored'
                          : item.combined_score === null ||
                              item.combined_score === undefined
                            ? '—'
                            : `${formatScore(item.combined_score)} / 100`}
                      </strong>
                    </div>
                  </div>

                  {showScores ? (
                    <div className="summary-metrics">
                      <span>
                        Semantic{' '}
                        <strong>
                          {item.status === 'unevaluable'
                            ? 'Not scored'
                            : formatScore(item.semantic_score)}
                        </strong>
                      </span>
                      <span>
                        Keywords{' '}
                        <strong>
                          {item.status === 'unevaluable'
                            ? 'Not scored'
                            : formatScore(item.keyword_score)}
                        </strong>
                      </span>
                    </div>
                  ) : null}

                  {item.matched_keywords?.length ||
                  item.missing_keywords?.length ? (
                    <div className="summary-keywords">
                      <KeywordChips
                        title="Matched"
                        items={item.matched_keywords}
                        tone="matched"
                      />
                      <KeywordChips
                        title="Missing"
                        items={item.missing_keywords}
                        tone="missing"
                      />
                    </div>
                  ) : null}

                  {item.status_detail ? (
                    <p className="summary-detail">{item.status_detail}</p>
                  ) : null}

                  {predictionMeta ? (
                    <div className="summary-prediction">
                      <span className={`badge badge-${predictionMeta.tone}`}>
                        {predictionMeta.label}
                      </span>
                      <span className="muted small">
                        {item.predicted_score === null ||
                        item.predicted_score === undefined
                          ? 'No predicted score'
                          : `${formatScore(item.predicted_score)} / 100`}
                        {item.prediction_is_stale ? ' · needs refresh' : ''}
                      </span>
                    </div>
                  ) : null}
                </article>
              )
            })}
          </div>
        </>
      )}
    </section>
  )
}
