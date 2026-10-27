/**
 * Single place for the backend URL. Override locally with VITE_API_BASE_URL in
 * frontend/.env. This value is not a secret.
 */
export const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'
).replace(/\/+$/, '')

export class ApiError extends Error {
  constructor(message, status = 0, payload = null) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.payload = payload
  }
}

function extractErrorMessage(payload, status) {
  const detail = payload?.detail

  if (typeof detail === 'string' && detail.trim()) {
    return detail
  }

  if (Array.isArray(detail)) {
    const messages = detail
      .map((item) => item?.msg)
      .filter((message) => typeof message === 'string' && message)
    if (messages.length > 0) {
      return messages.join('; ')
    }
  }

  return `Request failed (HTTP ${status}).`
}

async function request(path, options = {}) {
  let response

  try {
    response = await fetch(`${API_BASE_URL}${path}`, options)
  } catch {
    throw new ApiError(
      `Cannot reach the backend at ${API_BASE_URL}. Make sure FastAPI is running and the API base URL is correct.`,
      0,
    )
  }

  let payload = null
  const contentType = response.headers.get('content-type') || ''

  if (contentType.includes('application/json')) {
    payload = await response.json().catch(() => null)
  }

  if (!response.ok) {
    throw new ApiError(
      extractErrorMessage(payload, response.status),
      response.status,
      payload,
    )
  }

  return payload
}

export function getHealth() {
  return request('/health')
}

export function getQuestions() {
  return request('/interview/questions')
}

export function startInterview() {
  return request('/interview/start', { method: 'POST' })
}

export function getAnswers(interviewId) {
  return request(`/interview/${encodeURIComponent(interviewId)}/answers`)
}

export function transcribeAnswer(formData) {
  return request('/interview/transcribe', {
    method: 'POST',
    body: formData,
  })
}

export function evaluateAnswer(answerId) {
  return request(`/evaluation/${encodeURIComponent(answerId)}`, {
    method: 'POST',
  })
}

export function getEvaluation(answerId) {
  return request(`/evaluation/${encodeURIComponent(answerId)}`)
}

export function getInterviewEvaluations(interviewId) {
  return request(
    `/interview/${encodeURIComponent(interviewId)}/evaluations`,
  )
}

export function getInterviewSummary(interviewId) {
  return request(`/interview/${encodeURIComponent(interviewId)}/summary`)
}

export function analyzeContent(answerId) {
  return request(`/analysis/${encodeURIComponent(answerId)}/content`, {
    method: 'POST',
  })
}

export function getContentAnalysis(answerId) {
  return request(`/analysis/${encodeURIComponent(answerId)}/content`)
}

export function analyzeVoice(answerId) {
  return request(`/analysis/${encodeURIComponent(answerId)}/voice`, {
    method: 'POST',
  })
}

export function getVoiceAnalysis(answerId) {
  return request(`/analysis/${encodeURIComponent(answerId)}/voice`)
}

export function getInterviewAnalyses(interviewId) {
  return request(`/interview/${encodeURIComponent(interviewId)}/analyses`)
}

export function predictAnswer(answerId) {
  return request(`/prediction/${encodeURIComponent(answerId)}`, {
    method: 'POST',
  })
}

export function getPrediction(answerId) {
  return request(`/prediction/${encodeURIComponent(answerId)}`)
}

export function getInterviewPredictions(interviewId) {
  return request(
    `/interview/${encodeURIComponent(interviewId)}/predictions`,
  )
}

export function getPredictionSchema() {
  return request('/prediction/schema')
}

export function generateAnswerFeedback(answerId) {
  return request(`/feedback/${encodeURIComponent(answerId)}`, {
    method: 'POST',
  })
}

export function getAnswerFeedback(answerId) {
  return request(`/feedback/${encodeURIComponent(answerId)}`)
}

export function generateInterviewFeedback(interviewId) {
  return request(`/interview/${encodeURIComponent(interviewId)}/feedback`, {
    method: 'POST',
  })
}

export function getInterviewFeedback(interviewId) {
  return request(
    `/interview/${encodeURIComponent(interviewId)}/feedback`,
  )
}
