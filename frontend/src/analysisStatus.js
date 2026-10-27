const CONTENT_STATUS_META = {
  completed: { label: 'Content analysed', tone: 'ok' },
  unevaluable: { label: 'Transcript not evaluable', tone: 'warn' },
  not_evaluable: { label: 'No reference answer', tone: 'warn' },
  stale: { label: 'Needs re-analysis', tone: 'warn' },
}

const VOICE_STATUS_META = {
  completed: { label: 'Voice measured', tone: 'ok' },
  insufficient_speech: { label: 'Insufficient speech', tone: 'warn' },
  unavailable: { label: 'Audio unavailable', tone: 'warn' },
  failed: { label: 'Analysis failed', tone: 'error' },
  stale: { label: 'Needs re-analysis', tone: 'warn' },
}

export function analysisStatusMeta(kind, status) {
  const table = kind === 'voice' ? VOICE_STATUS_META : CONTENT_STATUS_META
  return table[status] || { label: status || 'Not analysed', tone: 'neutral' }
}

export function analysisStatusLabel(kind, status) {
  return analysisStatusMeta(kind, status).label
}
