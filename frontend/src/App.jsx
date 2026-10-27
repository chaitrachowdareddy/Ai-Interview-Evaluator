import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import './App.css'
import {
  API_BASE_URL,
  ApiError,
  analyzeContent,
  analyzeVoice,
  evaluateAnswer,
  generateAnswerFeedback,
  generateInterviewFeedback,
  getAnswers,
  getHealth,
  getInterviewAnalyses,
  getInterviewEvaluations,
  getInterviewFeedback,
  getInterviewPredictions,
  getInterviewSummary,
  getQuestions,
  predictAnswer,
  startInterview,
  transcribeAnswer,
} from './api'
import {
  ContentAnalysisCard,
  VoiceAnalysisCard,
} from './AnalysisViews'
import { analysisStatusLabel } from './analysisStatus'
import { EvaluationCard, EvaluationStatusBadge, InterviewSummary } from './EvaluationViews'
import {
  AnswerFeedbackCard,
  InterviewResultsDashboard,
  SavedAnswerFeedback,
} from './FeedbackViews'
import { PredictionCard } from './PredictionViews'
import { predictionStatusLabel } from './predictionStatus'
import {
  extensionForMimeType,
  formatBytes,
  formatDuration,
  getSupportedMimeType,
  isSupportedAudioFile,
  microphoneErrorMessage,
} from './audio'

const BUSY_RECORDER_STATES = new Set([
  'requesting',
  'recording',
  'stopping',
  'transcribing',
])

const SESSION_STORAGE_KEY = 'ai-interview-evaluator.active-session-id'
const DEFAULT_MAX_AUDIO_MB = 25

function readStoredSessionId() {
  try {
    const value = Number(window.localStorage.getItem(SESSION_STORAGE_KEY))
    return Number.isInteger(value) && value > 0 ? value : null
  } catch {
    return null
  }
}

function storeSessionId(sessionId) {
  try {
    window.localStorage.setItem(SESSION_STORAGE_KEY, String(sessionId))
  } catch {
    // Private browsing/storage restrictions must not break the workflow.
  }
}

function clearStoredSessionId() {
  try {
    window.localStorage.removeItem(SESSION_STORAGE_KEY)
  } catch {
    // Ignore storage failures; the in-memory session still works.
  }
}

function describeError(error) {
  if (error instanceof ApiError) return error.message
  if (error instanceof Error && error.message) return error.message
  return 'Something went wrong. Please try again.'
}

function formatTimestamp(value) {
  if (!value) return ''
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return ''
  return date.toLocaleString([], {
    dateStyle: 'medium',
    timeStyle: 'short',
  })
}

function formatScoreText(evaluation) {
  if (!evaluation) return ''
  if (evaluation.status === 'unevaluable') return 'Not scored'
  if (
    evaluation.combined_score === null ||
    evaluation.combined_score === undefined
  ) {
    return ''
  }
  return `${Number(evaluation.combined_score).toFixed(2)} / 100`
}

function upsertByAnswer(items, next) {
  return [
    ...items.filter((item) => item.answer_id !== next.answer_id),
    next,
  ]
}

function App() {
  const [questions, setQuestions] = useState([])
  const [questionsLoading, setQuestionsLoading] = useState(true)
  const [questionsError, setQuestionsError] = useState('')
  const [currentIndex, setCurrentIndex] = useState(0)

  const [interview, setInterview] = useState(null)
  const [starting, setStarting] = useState(false)
  const [answers, setAnswers] = useState([])
  const [answersLoading, setAnswersLoading] = useState(false)
  const [answersError, setAnswersError] = useState('')

  const [evaluations, setEvaluations] = useState([])
  const [summary, setSummary] = useState(null)
  const [summaryLoading, setSummaryLoading] = useState(false)
  const [evaluationError, setEvaluationError] = useState('')
  const [progressError, setProgressError] = useState('')
  const [evaluatingAnswerId, setEvaluatingAnswerId] = useState(null)

  const [contentAnalyses, setContentAnalyses] = useState([])
  const [voiceAnalyses, setVoiceAnalyses] = useState([])
  const [contentError, setContentError] = useState('')
  const [voiceError, setVoiceError] = useState('')
  const [analysisBusy, setAnalysisBusy] = useState(null)

  const [predictions, setPredictions] = useState([])
  const [modelStatus, setModelStatus] = useState(null)
  const [predictionError, setPredictionError] = useState('')
  const [predictingAnswerId, setPredictingAnswerId] = useState(null)

  const [answerFeedback, setAnswerFeedback] = useState([])
  const [feedbackError, setFeedbackError] = useState('')
  const [feedbackBusyAnswerId, setFeedbackBusyAnswerId] = useState(null)
  const [interviewFeedback, setInterviewFeedback] = useState(null)
  const [interviewFeedbackError, setInterviewFeedbackError] = useState('')
  const [interviewFeedbackLoading, setInterviewFeedbackLoading] =
    useState(false)
  const [interviewFeedbackGenerating, setInterviewFeedbackGenerating] =
    useState(false)

  const [health, setHealth] = useState(null)
  const [healthLoading, setHealthLoading] = useState(true)

  const [initialSessionId] = useState(() => readStoredSessionId())
  const [sessionRestoring, setSessionRestoring] = useState(
    () => Boolean(initialSessionId),
  )

  const [recorderState, setRecorderState] = useState('idle')
  const [recording, setRecording] = useState(null)
  const [elapsedMs, setElapsedMs] = useState(0)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')

  const mediaRecorderRef = useRef(null)
  const streamRef = useRef(null)
  const timerRef = useRef(null)
  const recordingUrlRef = useRef(null)
  const recordingStartRef = useRef(false)
  const submittingRef = useRef(false)
  const activeSessionIdRef = useRef(null)
  const evaluationBusyRef = useRef(false)
  const analysisBusyRef = useRef(false)
  const predictionBusyRef = useRef(false)
  const fileInputRef = useRef(null)

  const currentQuestion = questions[currentIndex] || null

  const isBusy = BUSY_RECORDER_STATES.has(recorderState)
  const navigationLocked = isBusy || Boolean(recording)

  const currentAnswer = currentQuestion
    ? answers.find((answer) => answer.question_id === currentQuestion.id)
    : null

  const currentEvaluation = currentAnswer
    ? evaluations.find(
        (evaluation) => evaluation.answer_id === currentAnswer.id,
      ) || null
    : null

  const contentByAnswer = useMemo(
    () =>
      new Map(
        contentAnalyses.map((analysis) => [analysis.answer_id, analysis]),
      ),
    [contentAnalyses],
  )

  const voiceByAnswer = useMemo(
    () =>
      new Map(voiceAnalyses.map((analysis) => [analysis.answer_id, analysis])),
    [voiceAnalyses],
  )

  const predictionByAnswer = useMemo(
    () => new Map(predictions.map((item) => [item.answer_id, item])),
    [predictions],
  )

  const feedbackByAnswer = useMemo(
    () => new Map(answerFeedback.map((item) => [item.answer_id, item])),
    [answerFeedback],
  )

  const currentContentAnalysis = currentAnswer
    ? contentByAnswer.get(currentAnswer.id) || null
    : null

  const currentVoiceAnalysis = currentAnswer
    ? voiceByAnswer.get(currentAnswer.id) || null
    : null

  const currentPrediction = currentAnswer
    ? predictionByAnswer.get(currentAnswer.id) || null
    : null

  const currentFeedback = currentAnswer
    ? feedbackByAnswer.get(currentAnswer.id) || null
    : null

  const evaluationsByAnswer = useMemo(
    () =>
      new Map(
        evaluations.map((evaluation) => [evaluation.answer_id, evaluation]),
      ),
    [evaluations],
  )

  const answeredEntries = useMemo(
    () =>
      questions
        .map((question, index) => ({
          question,
          index,
          answer: answers.find(
            (candidate) => candidate.question_id === question.id,
          ),
        }))
        .filter((entry) => entry.answer),
    [answers, questions],
  )

  const apiOnline = Boolean(health)
  const databaseOnline = health?.database === 'connected'
  const transcriptionConfigured = health?.transcription_configured === true
  const voiceAnalysisAvailable =
    health?.voice_analysis_available === undefined
      ? null
      : health.voice_analysis_available === true
  const mlModelAvailable =
    health?.ml_model_available === undefined
      ? null
      : health.ml_model_available === true
  const maxAudioMb =
    Number(health?.max_audio_mb) > 0
      ? Number(health.max_audio_mb)
      : DEFAULT_MAX_AUDIO_MB
  const maxAudioBytes = maxAudioMb * 1024 * 1024

  const stopTimer = useCallback(() => {
    if (timerRef.current !== null) {
      window.clearInterval(timerRef.current)
      timerRef.current = null
    }
  }, [])

  const releaseRecording = useCallback(() => {
    if (recordingUrlRef.current) {
      URL.revokeObjectURL(recordingUrlRef.current)
      recordingUrlRef.current = null
    }
    setRecording(null)
    setRecorderState('idle')
    setElapsedMs(0)
  }, [])

  const loadQuestions = useCallback(async () => {
    try {
      const data = await getQuestions()
      const list = Array.isArray(data) ? data : []
      setQuestions(list)
      setCurrentIndex((previous) =>
        list.length === 0 ? 0 : Math.min(previous, list.length - 1),
      )
    } catch (requestError) {
      setQuestions([])
      setQuestionsError(describeError(requestError))
    } finally {
      setQuestionsLoading(false)
    }
  }, [])

  useEffect(() => {
    // Initial data fetch is an intentional sync with the backend.
    // oxlint-disable-next-line react/set-state-in-effect
    loadQuestions()
  }, [loadQuestions])

  const checkHealth = useCallback(async () => {
    try {
      const data = await getHealth()
      setHealth(data)
    } catch (requestError) {
      if (
        requestError instanceof ApiError &&
        requestError.status === 503 &&
        requestError.payload
      ) {
        // The API is reachable, but MySQL is unavailable.
        setHealth(requestError.payload)
      } else {
        setHealth(null)
      }
    } finally {
      setHealthLoading(false)
    }
  }, [])

  function handleReloadQuestions() {
    setQuestionsLoading(true)
    setQuestionsError('')
    setHealthLoading(true)
    loadQuestions()
    checkHealth()
  }

  useEffect(() => {
    // Initial health check is an intentional sync with the backend.
    // oxlint-disable-next-line react/set-state-in-effect
    checkHealth()
  }, [checkHealth])

  useEffect(() => {
    return () => {
      if (timerRef.current !== null) {
        window.clearInterval(timerRef.current)
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop())
      }
      if (recordingUrlRef.current) {
        URL.revokeObjectURL(recordingUrlRef.current)
      }
    }
  }, [])

  const loadInterviewFeedback = useCallback(async (sessionId) => {
    setInterviewFeedbackLoading(true)
    setInterviewFeedbackError('')

    try {
      const data = await getInterviewFeedback(sessionId)
      if (activeSessionIdRef.current !== sessionId) return

      setInterviewFeedback(data?.interview_feedback || null)
      setAnswerFeedback(
        Array.isArray(data?.answer_feedback) ? data.answer_feedback : [],
      )
    } catch (requestError) {
      if (activeSessionIdRef.current === sessionId) {
        setInterviewFeedbackError(describeError(requestError))
      }
    } finally {
      setInterviewFeedbackLoading(false)
    }
  }, [])

  const loadSessionProgress = useCallback(async (sessionId) => {
    setSummaryLoading(true)
    setProgressError('')

    try {
      const [evaluationData, summaryData, analysesData, predictionData] =
        await Promise.all([
          getInterviewEvaluations(sessionId),
          getInterviewSummary(sessionId),
          getInterviewAnalyses(sessionId).catch(() => null),
          getInterviewPredictions(sessionId).catch(() => null),
        ])

      if (activeSessionIdRef.current !== sessionId) return

      setEvaluations(
        Array.isArray(evaluationData?.evaluations)
          ? evaluationData.evaluations
          : [],
      )
      setSummary(summaryData)
      if (analysesData) {
        setContentAnalyses(
          Array.isArray(analysesData.content_analyses)
            ? analysesData.content_analyses
            : [],
        )
        setVoiceAnalyses(
          Array.isArray(analysesData.voice_analyses)
            ? analysesData.voice_analyses
            : [],
        )
      }
      if (predictionData) {
        setPredictions(
          Array.isArray(predictionData.predictions)
            ? predictionData.predictions
            : [],
        )
        if (predictionData.model_status) {
          setModelStatus(predictionData.model_status)
        }
      }

      await loadInterviewFeedback(sessionId)
    } catch (requestError) {
      if (activeSessionIdRef.current === sessionId) {
        setProgressError(describeError(requestError))
      }
    } finally {
      setSummaryLoading(false)
    }
  }, [loadInterviewFeedback])

  useEffect(() => {
    if (!initialSessionId) return undefined

    let cancelled = false

    async function restoreSession() {
      try {
        const detail = await getAnswers(initialSessionId)
        if (cancelled) return

        const restored = {
          id: detail.interview_id,
          status: detail.status,
          started_at: detail.started_at,
        }

        activeSessionIdRef.current = restored.id
        setInterview(restored)
        setAnswers(Array.isArray(detail.answers) ? detail.answers : [])
        setAnswersError('')
        setCurrentIndex(0)
        setNotice(
          `Restored interview session #${restored.id} with ${
            detail.answers?.length || 0
          } saved answer(s) from MySQL.`,
        )
        await loadSessionProgress(restored.id)
      } catch (requestError) {
        if (cancelled) return

        if (requestError instanceof ApiError && requestError.status === 404) {
          // Only a genuinely missing session should discard the saved ID.
          clearStoredSessionId()
          setNotice(
            'The previously saved interview session no longer exists. Start a new interview.',
          )
        } else {
          // Keep the ID across transient database/API failures so a later
          // refresh can restore the same session.
          setAnswersError(describeError(requestError))
        }
      } finally {
        if (!cancelled) setSessionRestoring(false)
      }
    }

    restoreSession()
    return () => {
      cancelled = true
    }
  }, [initialSessionId, loadSessionProgress])

  async function handleEvaluateAnswer(answer) {
    if (!interview || !answer || evaluationBusyRef.current) return

    if (answer.interview_id !== interview.id) {
      setEvaluationError(
        'This saved answer belongs to a different interview session.',
      )
      return
    }

    evaluationBusyRef.current = true
    setEvaluatingAnswerId(answer.id)
    setEvaluationError('')
    setNotice('')

    try {
      const result = await evaluateAnswer(answer.id)
      if (activeSessionIdRef.current !== interview.id) return

      const saved = result.evaluation
      setEvaluations((previous) => [
        ...previous.filter(
          (evaluation) => evaluation.answer_id !== saved.answer_id,
        ),
        saved,
      ])
      setNotice(
        result.reused
          ? 'Loaded the existing saved evaluation for this answer.'
          : 'Answer evaluated and the result was saved to MySQL.',
      )

      await loadSessionProgress(interview.id)
    } catch (requestError) {
      setEvaluationError(describeError(requestError))
    } finally {
      evaluationBusyRef.current = false
      setEvaluatingAnswerId(null)
    }
  }

  async function handleAnalyzeContent(answer) {
    if (!interview || !answer || analysisBusyRef.current) return

    analysisBusyRef.current = true
    setAnalysisBusy(`content:${answer.id}`)
    setContentError('')
    setNotice('')

    try {
      const result = await analyzeContent(answer.id)
      if (activeSessionIdRef.current !== interview.id) return

      setContentAnalyses((previous) =>
        upsertByAnswer(previous, result.content_analysis),
      )
      setNotice(
        result.reused
          ? 'Loaded the existing saved content analysis for this answer.'
          : 'Content analysis saved to MySQL.',
      )
      await loadInterviewFeedback(interview.id)
    } catch (requestError) {
      setContentError(describeError(requestError))
    } finally {
      analysisBusyRef.current = false
      setAnalysisBusy(null)
    }
  }

  async function handleAnalyzeVoice(answer) {
    if (!interview || !answer || analysisBusyRef.current) return

    analysisBusyRef.current = true
    setAnalysisBusy(`voice:${answer.id}`)
    setVoiceError('')
    setNotice('')

    try {
      const result = await analyzeVoice(answer.id)
      if (activeSessionIdRef.current !== interview.id) return

      setVoiceAnalyses((previous) =>
        upsertByAnswer(previous, result.voice_analysis),
      )
      setNotice(
        result.reused
          ? 'Loaded the existing saved voice analysis for this answer.'
          : 'Voice analysis result saved to MySQL.',
      )
      await loadInterviewFeedback(interview.id)
    } catch (requestError) {
      setVoiceError(describeError(requestError))
    } finally {
      analysisBusyRef.current = false
      setAnalysisBusy(null)
    }
  }

  async function handlePredictAnswer(answer) {
    if (!interview || !answer || predictionBusyRef.current) return

    predictionBusyRef.current = true
    setPredictingAnswerId(answer.id)
    setPredictionError('')
    setNotice('')

    try {
      const result = await predictAnswer(answer.id)
      if (activeSessionIdRef.current !== interview.id) return

      setPredictions((previous) =>
        upsertByAnswer(previous, result.prediction),
      )
      if (result.model_status) {
        setModelStatus(result.model_status)
      }

      if (result.prediction.method === 'ml_model') {
        setNotice('Validated ML prediction saved to MySQL.')
      } else if (result.prediction.status === 'completed') {
        setNotice(
          'Rule-based baseline saved to MySQL. This is not an ML prediction.',
        )
      } else {
        setNotice(
          'Prediction status saved. No score was produced for this answer.',
        )
      }

      await loadSessionProgress(interview.id)
    } catch (requestError) {
      setPredictionError(describeError(requestError))
    } finally {
      predictionBusyRef.current = false
      setPredictingAnswerId(null)
    }
  }

  async function handleGenerateAnswerFeedback(answer) {
    if (!interview || !answer || feedbackBusyAnswerId !== null) return

    if (answer.interview_id !== interview.id) {
      setFeedbackError(
        'This saved answer belongs to a different interview session.',
      )
      return
    }

    setFeedbackBusyAnswerId(answer.id)
    setFeedbackError('')
    setNotice('')

    try {
      const result = await generateAnswerFeedback(answer.id)
      if (activeSessionIdRef.current !== interview.id) return

      setAnswerFeedback((previous) =>
        upsertByAnswer(previous, result.feedback),
      )
      if (result.feedback.status === 'completed') {
        setNotice(
          result.reused
            ? 'Loaded the saved feedback for this answer.'
            : 'Evidence-based answer feedback saved to MySQL.',
        )
      } else {
        setNotice(
          'Feedback status saved. Missing or invalid evidence is explained instead of being scored as zero.',
        )
      }

      await loadInterviewFeedback(interview.id)
    } catch (requestError) {
      setFeedbackError(describeError(requestError))
    } finally {
      setFeedbackBusyAnswerId(null)
    }
  }

  async function handleGenerateInterviewFeedback() {
    if (!interview || interviewFeedbackGenerating) return

    setInterviewFeedbackGenerating(true)
    setInterviewFeedbackError('')
    setNotice('')

    try {
      const result = await generateInterviewFeedback(interview.id)
      if (activeSessionIdRef.current !== interview.id) return

      setInterviewFeedback(result.interview_feedback || null)
      setAnswerFeedback(
        Array.isArray(result.answer_feedback) ? result.answer_feedback : [],
      )
      setNotice(
        result.reused
          ? 'Loaded the saved interview feedback.'
          : 'Interview feedback generated from current saved evidence and stored in MySQL.',
      )
    } catch (requestError) {
      setInterviewFeedbackError(describeError(requestError))
    } finally {
      setInterviewFeedbackGenerating(false)
    }
  }

  async function handleStartInterview() {
    if (starting || navigationLocked || questionsLoading) return

    setStarting(true)
    setError('')
    setNotice('')
    setEvaluations([])
    setSummary(null)
    setEvaluationError('')
    setProgressError('')
    setContentAnalyses([])
    setVoiceAnalyses([])
    setContentError('')
    setVoiceError('')
    setAnalysisBusy(null)
    setPredictions([])
    setPredictionError('')
    setPredictingAnswerId(null)
    setAnswerFeedback([])
    setFeedbackError('')
    setFeedbackBusyAnswerId(null)
    setInterviewFeedback(null)
    setInterviewFeedbackError('')
    setInterviewFeedbackGenerating(false)

    try {
      const created = await startInterview()
      activeSessionIdRef.current = created.id
      storeSessionId(created.id)
      setInterview(created)
      setAnswers([])
      setAnswersError('')
      setCurrentIndex(0)
      setNotice(
        `Interview session #${created.id} started. Record an answer for each question.`,
      )

      const sessionId = created.id
      setAnswersLoading(true)
      try {
        const detail = await getAnswers(sessionId)
        if (activeSessionIdRef.current === sessionId) {
          setAnswers(detail.answers || [])
        }
      } catch (answersRequestError) {
        setAnswersError(describeError(answersRequestError))
      } finally {
        setAnswersLoading(false)
      }

      await loadSessionProgress(sessionId)
    } catch (startError) {
      setError(describeError(startError))
    } finally {
      setStarting(false)
    }
  }

  async function handleRefreshAnswers() {
    if (!interview || answersLoading) return

    const sessionId = interview.id
    setAnswersLoading(true)
    setAnswersError('')

    try {
      const detail = await getAnswers(sessionId)
      if (activeSessionIdRef.current === sessionId) {
        setAnswers(detail.answers || [])
        setNotice('Saved answers refreshed from the database.')
      }
    } catch (requestError) {
      setAnswersError(describeError(requestError))
    } finally {
      setAnswersLoading(false)
    }

    await loadSessionProgress(sessionId)
  }

  async function handleStartRecording() {
    if (!interview || !currentQuestion) {
      setError('Start an interview session before recording an answer.')
      return
    }
    if (isBusy || recording || recordingStartRef.current) return

    setError('')
    setNotice('')

    if (
      !navigator.mediaDevices?.getUserMedia ||
      typeof MediaRecorder === 'undefined'
    ) {
      setError(
        'This browser does not support microphone recording. Use a current version of Chrome, Edge, Firefox, or Safari.',
      )
      return
    }

    recordingStartRef.current = true
    setRecorderState('requesting')

    let stream
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      })
    } catch (microphoneError) {
      setRecorderState('idle')
      setError(microphoneErrorMessage(microphoneError))
      recordingStartRef.current = false
      return
    }

    streamRef.current = stream
    const questionId = currentQuestion.id
    const interviewId = interview.id
    const startedAt = Date.now()
    const chunks = []
    let failed = false
    let recorder

    try {
      const mimeType = getSupportedMimeType()
      recorder = mimeType
        ? new MediaRecorder(stream, { mimeType })
        : new MediaRecorder(stream)

      recorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          chunks.push(event.data)
        }
      }

      recorder.onerror = () => {
        failed = true
        stopTimer()
        stream.getTracks().forEach((track) => track.stop())
        streamRef.current = null
        mediaRecorderRef.current = null
        setRecorderState('idle')
        setError(
          'The microphone recorder stopped unexpectedly. Check the microphone and try again.',
        )
      }

      recorder.onstop = () => {
        stopTimer()
        stream.getTracks().forEach((track) => track.stop())
        streamRef.current = null
        mediaRecorderRef.current = null

        if (failed) return

        const durationMs = Date.now() - startedAt
        setElapsedMs(durationMs)

        const actualMimeType =
          recorder.mimeType || mimeType || 'audio/webm'
        const blob = new Blob(chunks, { type: actualMimeType })

        if (blob.size === 0) {
          setRecorderState('idle')
          setError(
            'The recording was empty. Check that the microphone is working and try again.',
          )
          return
        }

        if (recordingUrlRef.current) {
          URL.revokeObjectURL(recordingUrlRef.current)
        }
        const url = URL.createObjectURL(blob)
        recordingUrlRef.current = url

        setRecording({
          blob,
          url,
          mimeType: actualMimeType,
          questionId,
          interviewId,
          durationMs,
        })
        setRecorderState('recorded')
      }

      recorder.start(250)
    } catch {
      stopTimer()
      stream.getTracks().forEach((track) => track.stop())
      streamRef.current = null
      mediaRecorderRef.current = null
      setRecorderState('idle')
      setError(
        'Could not start the audio recorder in this browser. Try a different browser or microphone.',
      )
      recordingStartRef.current = false
      return
    }

    mediaRecorderRef.current = recorder
    setElapsedMs(0)
    setRecorderState('recording')

    timerRef.current = window.setInterval(() => {
      setElapsedMs(Date.now() - startedAt)
    }, 200)

    recordingStartRef.current = false
  }

  function handleStopRecording() {
    const recorder = mediaRecorderRef.current
    if (!recorder || recorder.state === 'inactive') return

    stopTimer()
    setRecorderState('stopping')
    recorder.stop()
  }

  function handleDiscardRecording() {
    if (recorderState === 'transcribing') return

    setError('')
    setNotice('')
    releaseRecording()
  }

  function handleAudioFileSelected(event) {
    const file = event.target.files?.[0]
    // Reset the input so selecting the same file again still fires onChange.
    event.target.value = ''
    if (!file) return

    if (!interview || !currentQuestion) {
      setError('Start an interview session before uploading an answer.')
      return
    }
    if (isBusy || recording) return

    if (!isSupportedAudioFile(file)) {
      setError(
        'Unsupported audio format. Choose a WebM, Ogg, MP4/M4A, MP3, WAV, or FLAC file.',
      )
      return
    }
    if (file.size === 0) {
      setError('The selected audio file is empty. Choose another file.')
      return
    }
    if (file.size > maxAudioBytes) {
      setError(
        `The selected audio file is too large. Maximum size is ${maxAudioMb} MB.`,
      )
      return
    }

    setError('')
    setNotice('')

    if (recordingUrlRef.current) {
      URL.revokeObjectURL(recordingUrlRef.current)
    }
    const url = URL.createObjectURL(file)
    recordingUrlRef.current = url

    setRecording({
      blob: file,
      url,
      mimeType: (file.type || '').split(';', 1)[0].trim(),
      questionId: currentQuestion.id,
      interviewId: interview.id,
      durationMs: 0,
      fileName: file.name || 'answer',
      source: 'upload',
    })
    setRecorderState('recorded')
  }

  async function handleSubmitRecording() {
    if (!recording || !interview || !currentQuestion) return
    if (submittingRef.current) return

    if (
      recording.interviewId !== interview.id ||
      recording.questionId !== currentQuestion.id
    ) {
      setError(
        'This recording belongs to a different question. Discard it and record the current question again.',
      )
      return
    }

    submittingRef.current = true
    setRecorderState('transcribing')
    setError('')
    setNotice('')
    setEvaluationError('')
    setContentError('')
    setVoiceError('')

    const formData = new FormData()
    const uploadFileName =
      recording.fileName ||
      `answer.${extensionForMimeType(recording.mimeType)}`
    formData.append('file', recording.blob, uploadFileName)
    formData.append('interview_id', String(recording.interviewId))
    formData.append('question_id', String(recording.questionId))

    try {
      const result = await transcribeAnswer(formData)

      setAnswers((previous) => {
        const remaining = previous.filter(
          (answer) => answer.question_id !== result.answer.question_id,
        )
        return [...remaining, result.answer].sort(
          (first, second) => first.question_id - second.question_id,
        )
      })

      releaseRecording()
      setNotice(
        result.updated
          ? 'Transcript updated and saved to MySQL. Any previous evaluation is now marked for re-evaluation.'
          : 'Answer transcribed and saved to MySQL.',
      )
      await loadSessionProgress(result.answer.interview_id)
    } catch (transcriptionError) {
      setError(describeError(transcriptionError))
      // Keep the recording and let the user retry without re-recording.
      setRecorderState('recorded')
    } finally {
      submittingRef.current = false
    }
  }

  function goToQuestion(nextIndex) {
    if (navigationLocked) return
    if (nextIndex < 0 || nextIndex >= questions.length) return

    setError('')
    setNotice('')
    setEvaluationError('')
    setContentError('')
    setVoiceError('')
    setPredictionError('')
    setFeedbackError('')
    setCurrentIndex(nextIndex)
  }

  if (questionsLoading || sessionRestoring) {
    return (
      <main className="app">
        <Header
          apiOnline={false}
          databaseOnline={false}
          transcriptionConfigured={false}
          voiceAnalysisAvailable={voiceAnalysisAvailable}
          mlModelAvailable={mlModelAvailable}
          healthLoading
        />
        <div className="card state-card" role="status">
          <span className="spinner" aria-hidden="true" />
          <div>
            <h2>
              {sessionRestoring
                ? 'Restoring your saved session'
                : 'Loading interview questions'}
            </h2>
            <p>
              {sessionRestoring
                ? 'Reloading saved answers, transcripts, and evaluations from MySQL.'
                : 'The React app is asking FastAPI for questions stored in MySQL.'}
            </p>
          </div>
        </div>
      </main>
    )
  }

  if (questionsError) {
    return (
      <main className="app">
        <Header
          apiOnline={apiOnline}
          databaseOnline={databaseOnline}
          transcriptionConfigured={transcriptionConfigured}
          voiceAnalysisAvailable={voiceAnalysisAvailable}
          mlModelAvailable={mlModelAvailable}
          healthLoading={healthLoading}
        />
        <div className="card state-card error-state">
          <div>
            <h2>Could not load interview questions</h2>
            <p>{questionsError}</p>
            <p className="muted">
              Check that FastAPI is running at {API_BASE_URL}, MySQL is running,
              and backend/.env has the correct DATABASE_URL.
            </p>
          </div>
          <button
            type="button"
            className="button button-primary"
            onClick={handleReloadQuestions}
          >
            Retry
          </button>
        </div>
      </main>
    )
  }

  return (
    <main className="app">
      <Header
        apiOnline={apiOnline}
        databaseOnline={databaseOnline}
        transcriptionConfigured={transcriptionConfigured}
        voiceAnalysisAvailable={voiceAnalysisAvailable}
        mlModelAvailable={mlModelAvailable}
        healthLoading={healthLoading}
      />

      {questions.length === 0 ? (
        <div className="card state-card error-state">
          <div>
            <h2>No interview questions found</h2>
            <p>
              The backend returned an empty question list. Check the database
              connection and seed data, then retry.
            </p>
          </div>
          <button
            type="button"
            className="button button-primary"
            onClick={handleReloadQuestions}
          >
            Retry
          </button>
        </div>
      ) : (
        <>
          <section className="session-bar" aria-label="Interview session">
            <div className="session-meta">
              <span className="section-kicker">Interview session</span>
              <strong>
                {interview ? `Session #${interview.id}` : 'Not started'}
              </strong>
              <span className="muted small">
                {interview
                  ? `Started ${formatTimestamp(interview.started_at)}`
                  : 'Start a session before recording an answer.'}
              </span>
            </div>
            <button
              type="button"
              className="button button-primary"
              onClick={handleStartInterview}
              disabled={starting || navigationLocked}
            >
              {starting
                ? 'Starting session…'
                : interview
                  ? 'Start new session'
                  : 'Start interview'}
            </button>
          </section>

          {evaluationError ? (
            <div className="alert alert-error" role="alert">
              {evaluationError}
            </div>
          ) : null}

          <div className="workspace">
            <section className="card panel question-panel">
              <div className="panel-heading">
                <span className="section-kicker">
                  Question {currentIndex + 1} of {questions.length}
                </span>
                {currentQuestion ? (
                  <span className="badge category-badge">
                    {currentQuestion.category}
                  </span>
                ) : null}
              </div>

              <h2 className="question-text">
                {currentQuestion?.question_text}
              </h2>

              <div className="question-nav">
                <button
                  type="button"
                  className="button button-secondary"
                  onClick={() => goToQuestion(currentIndex - 1)}
                  disabled={navigationLocked || currentIndex === 0}
                >
                  Previous
                </button>

                <div className="question-dots" aria-label="Question shortcuts">
                  {questions.map((question, index) => {
                    const isAnswered = answers.some(
                      (answer) => answer.question_id === question.id,
                    )
                    return (
                      <button
                        key={question.id}
                        type="button"
                        className={[
                          'question-dot',
                          index === currentIndex ? 'active' : '',
                          isAnswered ? 'answered' : '',
                        ]
                          .filter(Boolean)
                          .join(' ')}
                        onClick={() => goToQuestion(index)}
                        disabled={navigationLocked}
                        aria-label={`Go to question ${index + 1}${
                          isAnswered ? ' (answered)' : ''
                        }`}
                        aria-current={index === currentIndex ? 'step' : undefined}
                        title={isAnswered ? 'Answer saved' : 'Not answered'}
                      >
                        {index + 1}
                      </button>
                    )
                  })}
                </div>

                <button
                  type="button"
                  className="button button-secondary"
                  onClick={() => goToQuestion(currentIndex + 1)}
                  disabled={
                    navigationLocked || currentIndex === questions.length - 1
                  }
                >
                  Next
                </button>
              </div>

              <p className="progress-note">
                {answeredEntries.length} of {questions.length} questions answered
              </p>

              {navigationLocked ? (
                <p className="hint hint-warning">
                  Submit or discard the current recording before changing
                  questions.
                </p>
              ) : null}
            </section>

            <section className="card panel answer-panel">
              <div className="panel-heading">
                <span className="section-kicker">Your answer</span>
                <span className={`badge ${currentAnswer ? 'saved' : ''}`}>
                  {currentAnswer ? 'Saved' : 'Not saved'}
                </span>
              </div>

              {error ? (
                <div className="alert alert-error" role="alert">
                  {error}
                </div>
              ) : null}

              {notice ? (
                <div className="alert alert-success" role="status">
                  {notice}
                </div>
              ) : null}

              {health && !transcriptionConfigured ? (
                <div className="alert alert-warning">
                  Recording is available, but Groq transcription is not
                  configured. Add GROQ_API_KEY to backend/.env and restart
                  FastAPI.
                </div>
              ) : null}

              {!interview ? (
                <div className="empty-answer">
                  <h3>Start an interview session</h3>
                  <p>
                    Use the <strong>Start interview</strong> button above to
                    create a session, then record your answer to this question.
                  </p>
                </div>
              ) : (
                <>
                  <div className="recorder-status" aria-live="polite">
                    {recorderState === 'requesting' ? (
                      <>
                        <span className="spinner" aria-hidden="true" />
                        <span>Waiting for microphone permission…</span>
                      </>
                    ) : null}
                    {recorderState === 'recording' ? (
                      <>
                        <span className="recording-dot" aria-hidden="true" />
                        <span>
                          Recording <strong>{formatDuration(elapsedMs)}</strong>
                        </span>
                      </>
                    ) : null}
                    {recorderState === 'stopping' ? (
                      <>
                        <span className="spinner" aria-hidden="true" />
                        <span>Finishing the recording…</span>
                      </>
                    ) : null}
                    {recorderState === 'transcribing' ? (
                      <>
                        <span className="spinner" aria-hidden="true" />
                        <span>Transcribing with Groq Whisper…</span>
                      </>
                    ) : null}
                    {recorderState === 'recorded' && recording ? (
                      <span>
                        {recording.source === 'upload'
                          ? `File ready · ${recording.fileName}`
                          : `Recording ready · ${formatDuration(
                              recording.durationMs,
                            )}`}{' '}
                        · {formatBytes(recording.blob.size)}
                      </span>
                    ) : null}
                    {recorderState === 'idle' ? (
                      <span className="muted">
                        Record with your microphone or upload an audio file,
                        then submit it for transcription.
                      </span>
                    ) : null}
                  </div>

                  {recording ? (
                    <audio
                      className="audio-player"
                      controls
                      src={recording.url}
                    >
                      Your browser does not support audio playback.
                    </audio>
                  ) : null}

                  <div className="recorder-actions">
                    {recorderState === 'recording' ? (
                      <button
                        type="button"
                        className="button button-stop"
                        onClick={handleStopRecording}
                      >
                        Stop recording
                      </button>
                    ) : null}

                    {recorderState === 'idle' ? (
                      <>
                        <button
                          type="button"
                          className="button button-record"
                          onClick={handleStartRecording}
                          disabled={isBusy}
                        >
                          <span className="record-icon" aria-hidden="true" />
                          Start recording
                        </button>
                        <button
                          type="button"
                          className="button button-secondary"
                          onClick={() => fileInputRef.current?.click()}
                          disabled={isBusy}
                        >
                          Upload audio file
                        </button>
                      </>
                    ) : null}

                    {recorderState === 'requesting' ||
                    recorderState === 'stopping' ||
                    recorderState === 'transcribing' ? (
                      <button type="button" className="button" disabled>
                        {recorderState === 'requesting'
                          ? 'Requesting microphone…'
                          : recorderState === 'stopping'
                            ? 'Finishing recording…'
                            : 'Transcribing…'}
                      </button>
                    ) : null}

                    {recording && recorderState === 'recorded' ? (
                      <>
                        <button
                          type="button"
                          className="button button-primary"
                          onClick={handleSubmitRecording}
                        >
                          Submit &amp; transcribe
                        </button>
                        <button
                          type="button"
                          className="button button-secondary"
                          onClick={handleDiscardRecording}
                        >
                          Discard
                        </button>
                      </>
                    ) : null}
                  </div>

                  <input
                    ref={fileInputRef}
                    className="visually-hidden"
                    type="file"
                    accept=".webm,.ogg,.oga,.m4a,.mp4,.mp3,.mpga,.wav,.flac,audio/*,video/webm,video/mp4"
                    onChange={handleAudioFileSelected}
                    tabIndex={-1}
                    aria-label="Upload an audio answer file"
                  />

                  <p className="hint">
                    Upload WebM, Ogg, MP4/M4A, MP3, WAV, or FLAC · maximum{' '}
                    {maxAudioMb} MB.
                  </p>

                  {recording && currentAnswer ? (
                    <p className="hint">
                      Submitting this recording will replace the saved
                      transcript for this question.
                    </p>
                  ) : null}

                  <div
                    className="transcript-block"
                    aria-busy={recorderState === 'transcribing'}
                  >
                    <div className="transcript-heading">
                      <h3>Saved transcript</h3>
                      {currentAnswer ? (
                        <time dateTime={currentAnswer.updated_at}>
                          {formatTimestamp(currentAnswer.updated_at)}
                        </time>
                      ) : null}
                    </div>

                    {currentAnswer ? (
                      <p className="transcript-text">
                        {currentAnswer.transcript}
                      </p>
                    ) : (
                      <p className="muted">
                        No transcript has been saved for this question yet.
                      </p>
                    )}
                  </div>

                  {currentAnswer ? (
                    <EvaluationCard
                      answer={currentAnswer}
                      evaluation={currentEvaluation}
                      evaluating={evaluatingAnswerId === currentAnswer.id}
                      disabled={evaluatingAnswerId !== null}
                      evaluable={currentQuestion?.has_reference_answer !== false}
                      onEvaluate={handleEvaluateAnswer}
                    />
                  ) : null}

                  {currentAnswer ? (
                    <ContentAnalysisCard
                      answer={currentAnswer}
                      hasReference={
                        currentQuestion?.has_reference_answer !== false
                      }
                      analysis={currentContentAnalysis}
                      loading={
                        analysisBusy === `content:${currentAnswer.id}`
                      }
                      disabled={analysisBusy !== null}
                      error={contentError}
                      onAnalyze={handleAnalyzeContent}
                    />
                  ) : null}

                  {currentAnswer ? (
                    <VoiceAnalysisCard
                      answer={currentAnswer}
                      analysis={currentVoiceAnalysis}
                      loading={analysisBusy === `voice:${currentAnswer.id}`}
                      disabled={analysisBusy !== null}
                      error={voiceError}
                      voiceAvailable={voiceAnalysisAvailable}
                      onAnalyze={handleAnalyzeVoice}
                    />
                  ) : null}

                  {currentAnswer ? (
                    <PredictionCard
                      answer={currentAnswer}
                      prediction={currentPrediction}
                      modelStatus={modelStatus}
                      loading={predictingAnswerId === currentAnswer.id}
                      disabled={predictingAnswerId !== null}
                      error={predictionError}
                      onPredict={handlePredictAnswer}
                    />
                  ) : null}

                  {currentAnswer ? (
                    <AnswerFeedbackCard
                      answer={currentAnswer}
                      feedback={currentFeedback}
                      loading={feedbackBusyAnswerId === currentAnswer.id}
                      disabled={feedbackBusyAnswerId !== null}
                      error={feedbackError}
                      onGenerate={handleGenerateAnswerFeedback}
                    />
                  ) : null}
                </>
              )}
            </section>
          </div>

          <section className="card saved-section">
            <div className="panel-heading">
              <div>
                <span className="section-kicker">Session record</span>
                <h2>Saved answers</h2>
              </div>
              <button
                type="button"
                className="button button-secondary button-small"
                onClick={handleRefreshAnswers}
                disabled={!interview || answersLoading}
              >
                {answersLoading ? 'Refreshing…' : 'Refresh'}
              </button>
            </div>

            {answersError ? (
              <div className="alert alert-error" role="alert">
                {answersError}
              </div>
            ) : null}

            {!interview ? (
              <p className="muted">
                Start a session to save and retrieve answer transcripts.
              </p>
            ) : answeredEntries.length === 0 ? (
              <p className="muted">
                No transcripts saved in this session yet.
              </p>
            ) : (
              <div className="saved-list">
                {answeredEntries.map(({ question, index, answer }) => {
                  const savedEvaluation = evaluationsByAnswer.get(answer.id)
                  const savedContent = contentByAnswer.get(answer.id)
                  const savedVoice = voiceByAnswer.get(answer.id)
                  const savedPrediction = predictionByAnswer.get(answer.id)
                  const savedFeedback = feedbackByAnswer.get(answer.id)
                  const isEvaluable = question.has_reference_answer !== false
                  const isEvaluating = evaluatingAnswerId === answer.id

                  return (
                    <article className="saved-answer" key={answer.id}>
                      <div className="saved-answer-head">
                        <span className="badge category-badge">
                          {question.category}
                        </span>
                        <span className="muted small">
                          Question {index + 1}
                        </span>
                      </div>
                      <h3>{question.question_text}</h3>
                      <p>{answer.transcript}</p>

                      <div className="saved-answer-evaluation">
                        <div className="saved-evaluation-state">
                          {savedEvaluation ? (
                            <>
                              <EvaluationStatusBadge
                                status={
                                  savedEvaluation.is_stale
                                    ? 'stale'
                                    : savedEvaluation.status
                                }
                              />
                              <span className="saved-evaluation-score">
                                {formatScoreText(savedEvaluation)}
                              </span>
                            </>
                          ) : (
                            <span className="badge">
                              {isEvaluable ? 'Not evaluated' : 'Not evaluable'}
                            </span>
                          )}
                        </div>
                        <button
                          type="button"
                          className="button button-secondary button-small"
                          onClick={() => handleEvaluateAnswer(answer)}
                          disabled={
                            !isEvaluable ||
                            isEvaluating ||
                            evaluatingAnswerId !== null
                          }
                          title={
                            isEvaluable
                              ? 'Evaluate this saved transcript'
                              : 'This question has no reference answer'
                          }
                        >
                          {isEvaluating
                            ? 'Evaluating…'
                            : savedEvaluation
                              ? 'Re-evaluate'
                              : 'Evaluate'}
                        </button>
                      </div>

                      <div className="saved-analysis-state">
                        <span className="muted small">
                          Content:{' '}
                          {savedContent
                            ? analysisStatusLabel(
                                'content',
                                savedContent.is_stale
                                  ? 'stale'
                                  : savedContent.status,
                              )
                            : 'not analysed'}
                        </span>
                        <span className="muted small">
                          Voice:{' '}
                          {savedVoice
                            ? analysisStatusLabel(
                                'voice',
                                savedVoice.is_stale
                                  ? 'stale'
                                  : savedVoice.status,
                              )
                            : 'not analysed'}
                        </span>
                        <span className="muted small">
                          Prediction:{' '}
                          {savedPrediction
                            ? predictionStatusLabel(savedPrediction.status, {
                                isStale: savedPrediction.is_stale,
                                method: savedPrediction.method,
                              })
                            : 'not predicted'}
                        </span>
                      </div>

                      <SavedAnswerFeedback
                        answer={answer}
                        feedback={savedFeedback}
                        loading={feedbackBusyAnswerId === answer.id}
                        disabled={feedbackBusyAnswerId !== null}
                        onGenerate={handleGenerateAnswerFeedback}
                      />
                    </article>
                  )
                })}
              </div>
            )}
          </section>

          <InterviewResultsDashboard
            feedback={interviewFeedback}
            loading={interviewFeedbackLoading}
            generating={interviewFeedbackGenerating}
            error={interviewFeedbackError}
            canGenerate={Boolean(interview)}
            onGenerate={handleGenerateInterviewFeedback}
          />

          <InterviewSummary
            summary={summary}
            loading={summaryLoading}
            error={progressError}
            canRefresh={Boolean(interview)}
            onRefresh={() => {
              if (interview) loadSessionProgress(interview.id)
            }}
          />
        </>
      )}

      <footer className="app-footer">
        <span>
          API: <code>{API_BASE_URL}</code>
        </span>
        <span>
          Phase 5 · Transcription, evaluation, content statistics, voice
          measurements, ML/baseline predictions, evidence-based feedback
        </span>
      </footer>
    </main>
  )
}

function Header({
  apiOnline,
  databaseOnline,
  transcriptionConfigured,
  voiceAnalysisAvailable,
  mlModelAvailable,
  healthLoading,
}) {
  return (
    <header className="app-header">
      <div className="brand-lockup">
        <div className="brand-mark" aria-hidden="true">
          <svg viewBox="0 0 24 24" fill="none">
            <path
              d="M12 3a3 3 0 0 0-3 3v6a3 3 0 0 0 6 0V6a3 3 0 0 0-3-3Z"
              stroke="currentColor"
              strokeWidth="1.7"
            />
            <path
              d="M5.5 11.5a6.5 6.5 0 0 0 13 0M12 18v3M9 21h6"
              stroke="currentColor"
              strokeWidth="1.7"
              strokeLinecap="round"
            />
          </svg>
        </div>
        <div>
          <p className="eyebrow">AI Interview Evaluator</p>
          <h1>Technical interview practice</h1>
          <p className="lede">
            Answer one question at a time, record with your microphone, and get
            a saved transcript, an explainable evaluation, separate content and
            voice measurements, a clearly labelled score prediction, and
            evidence-based question and interview feedback.
          </p>
        </div>
      </div>

      <div className="status-stack" aria-label="Service status">
        <span
          className={`status-pill ${
            healthLoading ? 'pending' : apiOnline ? 'ok' : 'error'
          }`}
          title={apiOnline ? 'FastAPI is reachable' : 'FastAPI is unreachable'}
        >
          <span className="status-dot" aria-hidden="true" />
          {healthLoading ? 'Checking API' : apiOnline ? 'API online' : 'API offline'}
        </span>
        <span
          className={`status-pill ${
            healthLoading
              ? 'pending'
              : databaseOnline
                ? 'ok'
                : 'error'
          }`}
          title={
            databaseOnline
              ? 'SQLAlchemy can reach MySQL'
              : 'MySQL is unavailable or not configured'
          }
        >
          <span className="status-dot" aria-hidden="true" />
          {healthLoading
            ? 'Checking MySQL'
            : databaseOnline
              ? 'MySQL connected'
              : 'MySQL unavailable'}
        </span>
        <span
          className={`status-pill ${
            healthLoading
              ? 'pending'
              : !apiOnline
                ? 'unknown'
                : transcriptionConfigured
                  ? 'ok'
                  : 'warn'
          }`}
          title={
            transcriptionConfigured
              ? 'GROQ_API_KEY is configured'
              : 'GROQ_API_KEY is missing in backend/.env'
          }
        >
          <span className="status-dot" aria-hidden="true" />
          {healthLoading
            ? 'Checking Groq'
            : !apiOnline
              ? 'Groq unknown'
              : transcriptionConfigured
                ? 'Groq configured'
                : 'Groq key missing'}
        </span>
        <span
          className={`status-pill ${
            healthLoading
              ? 'pending'
              : !apiOnline || voiceAnalysisAvailable === null
                ? 'unknown'
                : voiceAnalysisAvailable
                  ? 'ok'
                  : 'warn'
          }`}
          title={
            voiceAnalysisAvailable === null
              ? 'Voice analysis support is unknown until the API responds'
              : voiceAnalysisAvailable
                ? 'Librosa, soundfile, and FFmpeg are available'
                : 'Voice analysis dependencies are missing'
          }
        >
          <span className="status-dot" aria-hidden="true" />
          {healthLoading
            ? 'Checking voice analysis'
            : !apiOnline || voiceAnalysisAvailable === null
              ? 'Voice analysis unknown'
              : voiceAnalysisAvailable
                ? 'Voice analysis ready'
                : 'Voice analysis unavailable'}
        </span>
        <span
          className={`status-pill ${
            healthLoading
              ? 'pending'
              : !apiOnline || mlModelAvailable === null
                ? 'unknown'
                : mlModelAvailable
                  ? 'ok'
                  : 'warn'
          }`}
          title={
            mlModelAvailable === null
              ? 'ML model support is unknown until the API responds'
              : mlModelAvailable
                ? 'A validated ML model artifact is installed'
                : 'No validated ML model is installed; predictions use the rule-based baseline'
          }
        >
          <span className="status-dot" aria-hidden="true" />
          {healthLoading
            ? 'Checking ML model'
            : !apiOnline || mlModelAvailable === null
              ? 'ML model unknown'
              : mlModelAvailable
                ? 'ML model ready'
                : 'Baseline mode'}
        </span>
      </div>
    </header>
  )
}

export default App
