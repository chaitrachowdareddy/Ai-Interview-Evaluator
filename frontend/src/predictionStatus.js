export const PREDICTION_METHOD_LABELS = {
  ml_model: 'ML prediction',
  rule_based_baseline: 'Rule-based baseline',
  none: 'No prediction',
}

const PREDICTION_STATUS_META = {
  completed: { label: 'Prediction ready', tone: 'ok' },
  unavailable: { label: 'Prediction unavailable', tone: 'warn' },
  failed: { label: 'Prediction failed', tone: 'error' },
  pending: { label: 'Predicting…', tone: 'neutral' },
  stale: { label: 'Prediction stale', tone: 'warn' },
}

export function predictionMethodLabel(method) {
  return PREDICTION_METHOD_LABELS[method] || 'Prediction'
}

export function predictionStatusMeta(status, { isStale = false, method } = {}) {
  if (isStale) return PREDICTION_STATUS_META.stale
  if (status === 'completed') {
    return {
      label: predictionMethodLabel(method),
      tone: method === 'ml_model' ? 'ok' : 'warn',
    }
  }
  return (
    PREDICTION_STATUS_META[status] || {
      label: status || 'Not predicted',
      tone: 'neutral',
    }
  )
}

export function predictionStatusLabel(status, options) {
  return predictionStatusMeta(status, options).label
}
