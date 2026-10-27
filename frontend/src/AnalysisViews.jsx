import { analysisStatusMeta } from './analysisStatus'

export function AnalysisStatusBadge({ kind, status }) {
  const meta = analysisStatusMeta(kind, status)
  return <span className={`badge badge-${meta.tone}`}>{meta.label}</span>
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

function formatValue(value, digits = 2, unit = '') {
  if (value === null || value === undefined) return 'Not available'
  const number = Number(value)
  if (Number.isNaN(number)) return 'Not available'
  return `${number.toFixed(digits)}${unit}`
}

function Metric({ label, value, hint }) {
  const missing = value === 'Not available'
  return (
    <div className="analysis-metric">
      <span>{label}</span>
      <strong className={missing ? 'metric-missing' : ''}>{value}</strong>
      {hint ? <small>{hint}</small> : null}
    </div>
  )
}

function ChipList({ title, items, emptyText = 'None' }) {
  return (
    <div className="keyword-group">
      <span className="keyword-title">{title}</span>
      {items?.length ? (
        <ul className="keyword-list">
          {items.map((item) => (
            <li className="keyword-chip" key={`${title}-${item.key}`}>
              {item.label}
            </li>
          ))}
        </ul>
      ) : (
        <span className="muted small">{emptyText}</span>
      )}
    </div>
  )
}

function NoteList({ title, notes }) {
  if (!notes?.length) return null
  return (
    <div className="analysis-note-group">
      <span className="keyword-title">{title}</span>
      <ul className="analysis-note-list">
        {notes.map((note) => (
          <li key={note}>{note}</li>
        ))}
      </ul>
    </div>
  )
}

const COVERAGE_SOURCE_TEXT = {
  phase2_evaluation:
    'Coverage reused from the current Phase 2 evaluation, so both features show the same number.',
  keyword_coverage:
    'Computed with the same weighted keyword coverage used by Phase 2.',
  none: 'No reference answer is available, so reference coverage is not calculated.',
}

export function ContentAnalysisCard({
  answer,
  hasReference = true,
  analysis,
  loading,
  disabled,
  error,
  onAnalyze,
}) {
  if (!answer) return null

  const isStale = analysis?.is_stale === true
  const status = isStale ? 'stale' : analysis?.status
  const busy = loading === true
  const coveragePossible = hasReference !== false

  return (
    <section
      className="analysis-block content-analysis"
      aria-busy={busy}
    >
      <div className="analysis-head">
        <div>
          <span className="section-kicker">Phase 3 · Content</span>
          <h3>Coverage and transcript statistics</h3>
        </div>
        {analysis ? (
          <AnalysisStatusBadge kind="content" status={status} />
        ) : (
          <span className="badge">Not analysed</span>
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
            Measuring lexical coverage, filler words, repetition, and sentence
            structure…
          </span>
        </div>
      ) : null}

      {!analysis && !busy ? (
        <>
          <p className="muted">
            Measure lexical coverage of the reference terminology, filler-word
            share, repeated content words, and sentence statistics. No new
            score is created: coverage reuses the Phase 2 coverage definition.
          </p>
          {!coveragePossible ? (
            <div className="alert alert-warning">
              This question has no reference answer. Transcript-only statistics
              can still be measured, but coverage will be unavailable.
            </div>
          ) : null}
          <button
            type="button"
            className="button button-primary"
            onClick={() => onAnalyze(answer)}
            disabled={disabled}
          >
            Analyse content
          </button>
        </>
      ) : null}

      {analysis && !busy ? (
        <>
          {isStale ? (
            <div className="alert alert-warning">
              The transcript or reference answer changed after this analysis.
              The saved statistics are outdated; re-analyse to refresh them.
            </div>
          ) : null}

          {analysis.status === 'not_evaluable' ? (
            <div className="alert alert-warning">
              {analysis.status_detail ||
                'This question has no reference answer, so coverage is not calculated.'}
            </div>
          ) : null}

          {analysis.status === 'unevaluable' ? (
            <div className="alert alert-warning">
              {analysis.status_detail ||
                'The saved transcript is too short or contains only filler words.'}
            </div>
          ) : null}

          <div className="analysis-grid">
            <Metric
              label="Reference coverage"
              value={formatValue(analysis.coverage_score, 2)}
              hint={analysis.coverage_score == null ? 'Lexical coverage' : '/ 100'}
            />
            <Metric
              label="Filler-word share"
              value={formatValue(analysis.filler_rate_percent, 1, '%')}
              hint={`${analysis.filler_word_count || 0} filler word(s)`}
            />
            <Metric
              label="Repeated content words"
              value={formatValue(analysis.repetition_rate_percent, 1, '%')}
              hint="Share of meaningful words"
            />
            <Metric
              label="Average sentence"
              value={formatValue(analysis.average_sentence_words, 1, ' words')}
              hint={`${analysis.sentence_count || 0} sentence(s)`}
            />
          </div>

          <p className="analysis-source">
            {COVERAGE_SOURCE_TEXT[analysis.coverage_source] ||
              COVERAGE_SOURCE_TEXT.none}
          </p>

          <ChipList
            title="Expected terms found"
            items={(analysis.matched_concepts || []).map((term) => ({
              key: term,
              label: term,
            }))}
          />
          <ChipList
            title="Expected terms not found"
            items={(analysis.missing_concepts || []).map((term) => ({
              key: term,
              label: term,
            }))}
          />
          <ChipList
            title="Filler words"
            items={(analysis.filler_words || []).map((item) => ({
              key: item.word,
              label: `${item.word} × ${item.count}`,
            }))}
          />
          <ChipList
            title="Repeated terms"
            items={(analysis.repeated_terms || []).map((item) => ({
              key: item.term,
              label: `${item.term} × ${item.count}`,
            }))}
          />

          <NoteList title="Structure observations" notes={analysis.structure_notes} />

          <div className="analysis-footer">
            <span className="muted small">
              Saved {formatWhen(analysis.updated_at)} · methodology{' '}
              {analysis.methodology_version}
            </span>
            <button
              type="button"
              className="button button-secondary button-small"
              onClick={() => onAnalyze(answer)}
              disabled={disabled}
            >
              {isStale ? 'Re-analyse content' : 'Analyse again'}
            </button>
          </div>
        </>
      ) : null}

      <p className="fine-print">
        Coverage is lexical: it shows whether expected terminology appears, not
        whether the answer is factually correct. Filler and repetition counts
        describe the transcript only and are not a competency rating.
      </p>
    </section>
  )
}

export function VoiceAnalysisCard({
  answer,
  analysis,
  loading,
  disabled,
  error,
  voiceAvailable,
  onAnalyze,
}) {
  if (!answer) return null

  const isStale = analysis?.is_stale === true
  const status = isStale ? 'stale' : analysis?.status
  const busy = loading === true
  const hasAudio = answer.has_audio === true
  const dependenciesMissing = voiceAvailable === false
  const pitchUnavailable = analysis?.pitch_reliable === false

  const canAnalyze = hasAudio && !dependenciesMissing

  return (
    <section className="analysis-block voice-analysis" aria-busy={busy}>
      <div className="analysis-head">
        <div>
          <span className="section-kicker">Phase 3 · Voice</span>
          <h3>Measured acoustic features</h3>
        </div>
        {analysis ? (
          <AnalysisStatusBadge kind="voice" status={status} />
        ) : (
          <span className="badge">Not analysed</span>
        )}
      </div>

      {error ? (
        <div className="alert alert-error" role="alert">
          {error}
        </div>
      ) : null}

      {!hasAudio ? (
        <div className="alert alert-warning">
          No original recording is stored for this answer. Voice analysis needs
          the saved audio, so duration, pauses, speaking rate, and pitch are
          unavailable. Phase 3 keeps recordings from new submissions or
          uploads; earlier answers were not retained.
        </div>
      ) : null}

      {dependenciesMissing ? (
        <div className="alert alert-warning">
          Voice analysis is not available on this server. Install the backend
          analysis dependencies (librosa, soundfile, imageio-ffmpeg) and
          restart FastAPI.
        </div>
      ) : null}

      {busy ? (
        <div className="evaluation-loading" role="status">
          <span className="spinner" aria-hidden="true" />
          <span>Measuring the stored recording…</span>
        </div>
      ) : null}

      {!analysis && !busy ? (
        <>
          <p className="muted">
            Measure duration, detected speech and silence, pauses, speaking
            rate from the saved transcript, energy, clipping, and pitch from
            the stored recording.
          </p>
          <button
            type="button"
            className="button button-primary"
            onClick={() => onAnalyze(answer)}
            disabled={disabled || !canAnalyze}
          >
            Analyse voice
          </button>
        </>
      ) : null}

      {analysis && !busy ? (
        <>
          {isStale ? (
            <div className="alert alert-warning">
              The transcript or recording changed after this analysis. The
              saved measurements are outdated; re-analyse to refresh them.
            </div>
          ) : null}

          {analysis.status === 'unavailable' ||
          analysis.status === 'failed' ? (
            <div className="alert alert-warning">
              {analysis.status_detail ||
                'Voice measurements could not be produced for this recording.'}
            </div>
          ) : null}

          {analysis.status === 'insufficient_speech' ? (
            <div className="alert alert-warning">
              {analysis.status_detail ||
                'No speech was detected above the energy threshold.'}
            </div>
          ) : null}

          <div className="analysis-grid">
            <Metric
              label="Duration"
              value={formatValue(analysis.duration_seconds, 2, ' s')}
            />
            <Metric
              label="Detected speech"
              value={formatValue(analysis.speech_duration_seconds, 2, ' s')}
              hint={formatValue(analysis.speech_ratio_percent, 1, '%')}
            />
            <Metric
              label="Pauses"
              value={formatValue(analysis.pause_count, 0)}
              hint={`Longest ${formatValue(
                analysis.longest_pause_seconds,
                2,
                ' s',
              )}`}
            />
            <Metric
              label="Pauses per minute"
              value={formatValue(analysis.pauses_per_minute, 1)}
            />
            <Metric
              label="Speaking rate (estimated)"
              value={formatValue(analysis.speaking_rate_wpm, 1, ' wpm')}
              hint={`${formatValue(analysis.word_count, 0)} transcript word(s)`}
            />
            <Metric
              label="Average level"
              value={formatValue(analysis.rms_mean_dbfs, 1, ' dBFS')}
            />
            <Metric
              label="Level variation"
              value={formatValue(analysis.rms_variation_percent, 1, '%')}
            />
            <Metric
              label="Peak amplitude"
              value={formatValue(analysis.peak_amplitude, 3)}
            />
            <Metric
              label="Clipping"
              value={formatValue(analysis.clipping_percent, 2, '%')}
            />
            <Metric
              label="Pitch median"
              value={
                pitchUnavailable
                  ? 'Not reliable'
                  : formatValue(analysis.pitch_median_hz, 1, ' Hz')
              }
            />
            <Metric
              label="Pitch range (P10–P90)"
              value={
                pitchUnavailable
                  ? 'Not reliable'
                  : formatValue(analysis.pitch_range_hz, 1, ' Hz')
              }
            />
            <Metric
              label="Voiced speech analysed"
              value={formatValue(analysis.pitch_voiced_seconds, 2, ' s')}
            />
          </div>

          {pitchUnavailable && analysis.speech_detected ? (
            <p className="analysis-source">
              Pitch could not be estimated reliably from the detected voiced
              audio. Pitch values are therefore shown as unavailable rather
              than guessed.
            </p>
          ) : null}

          <NoteList title="Audio quality notes" notes={analysis.quality_notes} />

          <div className="analysis-footer">
            <span className="muted small">
              Saved {formatWhen(analysis.updated_at)} · methodology{' '}
              {analysis.methodology_version}
            </span>
            <button
              type="button"
              className="button button-secondary button-small"
              onClick={() => onAnalyze(answer)}
              disabled={disabled || !canAnalyze}
            >
              {isStale ? 'Re-analyse voice' : 'Analyse again'}
            </button>
          </div>
        </>
      ) : null}

      <p className="fine-print">
        Speaking rate is transcript words divided by detected speech time, with
        pauses excluded. These are acoustic measurements, not scores of
        confidence, personality, honesty, intelligence, or emotion.
      </p>
    </section>
  )
}
