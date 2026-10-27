import {
  predictionMethodLabel,
  predictionStatusMeta,
} from './predictionStatus'

const FEATURE_LABELS = {
  semantic_score: 'Semantic similarity',
  keyword_score: 'Keyword coverage',
  filler_rate_percent: 'Filler-word share',
  repetition_rate_percent: 'Repeated content words',
  average_sentence_words: 'Average sentence length',
  speaking_rate_wpm: 'Speaking rate',
  pauses_per_minute: 'Pauses per minute',
  speech_ratio_percent: 'Speech ratio',
  rms_variation_percent: 'Loudness variation',
  has_content_features: 'Content features available',
  has_voice_features: 'Voice features available',
}

const COMPONENT_LABELS = {
  answer: 'Answer comparison (Phase 2)',
  content_style: 'Transcript style (Phase 3)',
  delivery: 'Delivery (Phase 3)',
}

const COMPONENT_SOURCES = {
  phase2_combined_score: 'Phase 2 combined score, reused unchanged',
  phase3_content_analysis: 'Phase 3 content analysis',
  phase3_voice_analysis: 'Phase 3 voice analysis',
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

function formatWeight(value) {
  const number = Number(value)
  if (!Number.isFinite(number)) return '—'
  return `${Math.round(number * 100)}%`
}

export function PredictionStatusBadge({ status, method, isStale = false }) {
  const meta = predictionStatusMeta(status, { isStale, method })
  return <span className={`badge badge-${meta.tone}`}>{meta.label}</span>
}

function ModelStatusNotice({ modelStatus }) {
  if (!modelStatus) return null
  const available = modelStatus.available === true
  return (
    <div className={`model-status-notice ${available ? 'ready' : 'baseline'}`}>
      <strong>
        {available
          ? 'Validated ML model installed'
          : 'ML model unavailable — baseline mode'}
      </strong>
      <span>{modelStatus.reason}</span>
      {available ? (
        <small>
          Version {modelStatus.model_version || 'unknown'} · schema{' '}
          {modelStatus.feature_schema_version || 'unknown'}
          {modelStatus.trained_at
            ? ` · trained ${formatWhen(modelStatus.trained_at)}`
            : ''}
        </small>
      ) : (
        <small>
          Predictions use the transparent rule-based baseline until a validated
          artifact is installed. The baseline is not a machine-learning
          prediction.
        </small>
      )}
    </div>
  )
}

function ComponentList({ components }) {
  const items = Array.isArray(components?.components)
    ? components.components
    : []
  if (items.length === 0) return null

  return (
    <div className="prediction-components">
      <span className="keyword-title">Baseline components</span>
      <div className="prediction-component-grid">
        {items.map((component) => (
          <div className="prediction-component" key={component.name}>
            <span>{COMPONENT_LABELS[component.name] || component.name}</span>
            <strong>{formatScore(component.score)}</strong>
            <small>
              Weight {formatWeight(component.weight)} ·{' '}
              {COMPONENT_SOURCES[component.source] || component.source}
            </small>
          </div>
        ))}
      </div>
      <span className="muted small">
        {components.weights_renormalized
          ? 'Missing components were excluded and the remaining weights were renormalized.'
          : 'All three components were available.'}{' '}
        Effective weight total: {formatWeight(components.effective_weight_total)}.
      </span>
    </div>
  )
}

function ModelComponent({ components }) {
  const model = components?.model
  if (!model) return null
  return (
    <div className="prediction-components">
      <span className="keyword-title">Model details</span>
      <div className="prediction-component-grid">
        <div className="prediction-component">
          <span>Model version</span>
          <strong>{model.model_version || 'unknown'}</strong>
        </div>
        <div className="prediction-component">
          <span>Trained</span>
          <strong>{formatWhen(model.trained_at) || 'unknown'}</strong>
        </div>
        <div className="prediction-component">
          <span>Validation</span>
          <strong>{model.validation_status || 'unknown'}</strong>
        </div>
      </div>
    </div>
  )
}

function NoteList({ title, notes, tone = '' }) {
  if (!notes?.length) return null
  return (
    <div className={`prediction-notes ${tone}`}>
      <span className="keyword-title">{title}</span>
      <ul>
        {notes.map((note) => (
          <li key={note}>{note}</li>
        ))}
      </ul>
    </div>
  )
}

export function PredictionCard({
  answer,
  prediction,
  modelStatus,
  loading,
  disabled,
  error,
  onPredict,
}) {
  if (!answer) return null

  const busy = loading === true
  const isStale = prediction?.is_stale === true
  const completed = prediction?.status === 'completed' && !isStale
  const hasScore = completed && prediction.score !== null
  const isBaseline = hasScore && prediction.method === 'rule_based_baseline'
  const isMl = hasScore && prediction.method === 'ml_model'

  return (
    <section className="prediction-block" aria-busy={busy}>
      <div className="prediction-head">
        <div>
          <span className="section-kicker">Phase 4 · Prediction</span>
          <h3>Overall answer score prediction</h3>
        </div>
        {prediction ? (
          <PredictionStatusBadge
            status={prediction.status}
            method={prediction.method}
            isStale={isStale}
          />
        ) : (
          <span className="badge">Not predicted</span>
        )}
      </div>

      {error ? (
        <div className="alert alert-error" role="alert">
          {error}
        </div>
      ) : null}

      {busy ? (
        <div className="evaluation-loading" role="status">
          <span className="spinner" aria-hidden="true" />
          <span>
            Building eligible features and scoring this answer…
          </span>
        </div>
      ) : null}

      {!prediction && !busy ? (
        <>
          <p className="muted">
            Combine the saved Phase 2 evaluation with any Phase 3 content and
            voice measurements into one 0–100 score. A score is labelled an ML
            prediction only when a validated model artifact is installed;
            otherwise the transparent rule-based baseline is used and labelled
            as such.
          </p>
          <ModelStatusNotice modelStatus={modelStatus} />
          <button
            type="button"
            className="button button-primary"
            onClick={() => onPredict(answer)}
            disabled={disabled}
          >
            Predict score
          </button>
        </>
      ) : null}

      {prediction && !busy ? (
        <>
          {isStale ? (
            <div className="alert alert-warning">
              The transcript, reference answer, Phase 2 evaluation, or Phase 3
              analysis changed after this prediction. The saved score is
              outdated; predict again to refresh it.
            </div>
          ) : null}

          {prediction.status === 'unavailable' ||
          prediction.status === 'failed' ? (
            <div
              className={`alert ${
                prediction.status === 'failed'
                  ? 'alert-error'
                  : 'alert-warning'
              }`}
            >
              <strong>No score produced.</strong>{' '}
              {prediction.status_detail ||
                'The required sources are not available for this answer.'}
            </div>
          ) : null}

          {hasScore ? (
            <>
              {isBaseline ? (
                <div className="alert alert-warning">
                  This is a <strong>rule-based baseline</strong>, not a
                  machine-learning prediction. Its fixed weights have not been
                  learned from or validated against human scores.
                </div>
              ) : null}

              <div className="score-grid">
                <div className="score-cell">
                  <span>
                    {isMl ? 'ML prediction' : 'Baseline score'}
                  </span>
                  <strong>{formatScore(prediction.score)}</strong>
                  <span className="score-scale">/ 100</span>
                </div>
                <div className="score-cell">
                  <span>Method</span>
                  <strong className="prediction-method-name">
                    {prediction.method_label ||
                      predictionMethodLabel(prediction.method)}
                  </strong>
                </div>
                <div className="score-cell">
                  <span>Feature schema</span>
                  <strong className="prediction-method-name">
                    {prediction.feature_schema_version}
                  </strong>
                </div>
              </div>

              {isBaseline ? (
                <ComponentList components={prediction.components} />
              ) : (
                <ModelComponent components={prediction.components} />
              )}

              {prediction.missing_features?.length ? (
                <div className="prediction-missing">
                  <span className="keyword-title">
                    Unavailable features (excluded, never treated as zero)
                  </span>
                  <ul className="keyword-list">
                    {prediction.missing_features.map((name) => (
                      <li className="keyword-chip" key={name}>
                        {FEATURE_LABELS[name] || name}
                      </li>
                    ))}
                  </ul>
                </div>
              ) : null}
            </>
          ) : null}

          <NoteList
            title="Warnings"
            notes={prediction.warnings}
            tone="warning"
          />
          <NoteList
            title="Limitations"
            notes={prediction.limitations}
            tone="limitation"
          />

          {prediction.source_versions?.evaluation_id ? (
            <p className="analysis-source">
              Sources: Phase 2 evaluation #{prediction.source_versions.evaluation_id}
              {prediction.source_versions.content_id
                ? ` · content #${prediction.source_versions.content_id}`
                : ' · no current content analysis'}
              {prediction.source_versions.voice_id
                ? ` · voice #${prediction.source_versions.voice_id}`
                : ' · no current voice analysis'}
            </p>
          ) : null}

          <div className="prediction-footer">
            <span className="muted small">
              Saved {formatWhen(prediction.updated_at)} · methodology{' '}
              {prediction.methodology_version}
              {prediction.model_version
                ? ` · model ${prediction.model_version}`
                : ''}
            </span>
            <button
              type="button"
              className="button button-secondary button-small"
              onClick={() => onPredict(answer)}
              disabled={disabled}
            >
              {isStale ? 'Re-predict score' : 'Predict again'}
            </button>
          </div>
        </>
      ) : null}

      <p className="fine-print">
        Predictions are a practice aid, not a hiring decision or proof of
        knowledge. An ML prediction appears only when a validated model artifact
        is installed; otherwise the labelled rule-based baseline is shown. The
        saved Phase 2 and Phase 3 results are never overwritten.
      </p>
    </section>
  )
}
