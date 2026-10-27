const STATUS_META = {
  completed: { label: 'Feedback ready', tone: 'ok' },
  unavailable: { label: 'Feedback unavailable', tone: 'warn' },
  failed: { label: 'Feedback failed', tone: 'error' },
  stale: { label: 'Needs refresh', tone: 'warn' },
  unanswered: { label: 'Unanswered', tone: 'neutral' },
  partial: { label: 'Partial results', tone: 'warn' },
}

const BAND_META = {
  strong: { label: 'Strong answer', tone: 'ok' },
  average: { label: 'Average answer', tone: 'warn' },
  weak: { label: 'Needs revision', tone: 'error' },
}

export function feedbackStatusMeta(status, { isStale = false } = {}) {
  if (isStale) {
    return STATUS_META.stale
  }
  return STATUS_META[status] || { label: status || 'Unknown', tone: 'neutral' }
}

export function feedbackStatusLabel(status, options) {
  return feedbackStatusMeta(status, options).label
}

export function feedbackBandMeta(band) {
  if (!band) return null
  return BAND_META[band] || { label: band, tone: 'neutral' }
}

export function feedbackMethodLabel(method, isMl = false) {
  if (method === 'ml_model') {
    return 'ML prediction'
  }
  if (method === 'rule_based_baseline') {
    return 'Rule-based baseline'
  }
  return isMl ? 'ML prediction' : 'No prediction'
}
