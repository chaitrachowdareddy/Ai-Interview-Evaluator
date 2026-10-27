/** Browser audio helpers: MIME negotiation, naming, and microphone errors. */

const MIME_CANDIDATES = [
  'audio/webm;codecs=opus',
  'audio/webm',
  'audio/ogg;codecs=opus',
  'audio/ogg',
  'audio/mp4',
  'audio/wav',
]

export function getSupportedMimeType() {
  if (typeof MediaRecorder === 'undefined') {
    return ''
  }

  if (typeof MediaRecorder.isTypeSupported !== 'function') {
    return ''
  }

  return (
    MIME_CANDIDATES.find((mimeType) => {
      try {
        return MediaRecorder.isTypeSupported(mimeType)
      } catch {
        return false
      }
    }) || ''
  )
}

export function extensionForMimeType(mimeType) {
  const baseType = (mimeType || '').split(';', 1)[0].trim().toLowerCase()

  if (baseType.includes('webm')) return 'webm'
  if (baseType.includes('ogg')) return 'ogg'
  if (baseType.includes('mp4') || baseType.includes('m4a')) return 'm4a'
  if (baseType.includes('mpeg') || baseType.includes('mp3')) return 'mp3'
  if (baseType.includes('wav')) return 'wav'
  if (baseType.includes('flac')) return 'flac'

  // Chrome/Firefox/Edge default. The backend also falls back to the extension.
  return 'webm'
}

export function isSupportedAudioFile(file) {
  if (!file) return false

  const mimeType = (file.type || '').split(';', 1)[0].trim().toLowerCase()
  const supportedMimeTypes = new Set([
    'audio/webm',
    'video/webm',
    'audio/ogg',
    'application/ogg',
    'audio/mp4',
    'audio/x-m4a',
    'audio/m4a',
    'video/mp4',
    'audio/mpeg',
    'audio/mp3',
    'audio/mpga',
    'audio/wav',
    'audio/x-wav',
    'audio/wave',
    'audio/flac',
    'audio/x-flac',
  ])
  if (supportedMimeTypes.has(mimeType)) return true

  const name = (file.name || '').toLowerCase()
  const extension = name.includes('.') ? name.split('.').pop() : ''
  return new Set([
    'webm',
    'ogg',
    'oga',
    'm4a',
    'mp4',
    'mp3',
    'mpga',
    'wav',
    'flac',
  ]).has(extension)
}

export function formatDuration(milliseconds) {
  const totalSeconds = Math.max(0, Math.round((milliseconds || 0) / 1000))
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60
  return `${minutes}:${String(seconds).padStart(2, '0')}`
}

export function formatBytes(bytes) {
  if (!bytes) return '0 KB'
  if (bytes < 1024 * 1024) {
    return `${Math.max(1, Math.round(bytes / 1024))} KB`
  }
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

export function microphoneErrorMessage(error) {
  switch (error?.name) {
    case 'NotAllowedError':
    case 'PermissionDeniedError':
      return 'Microphone permission was denied. Allow microphone access in the browser site settings, then try again.'
    case 'NotFoundError':
    case 'DevicesNotFoundError':
      return 'No microphone was found. Connect a microphone and try again.'
    case 'NotReadableError':
    case 'TrackStartError':
      return 'The microphone is already in use by another application. Close it and try again.'
    case 'SecurityError':
      return 'Microphone access is blocked. Open the frontend through http://localhost:5173 or HTTPS and try again.'
    case 'OverconstrainedError':
      return 'The selected microphone does not support the requested audio settings.'
    default:
      return 'Could not access the microphone. Check Windows microphone permissions and try again.'
  }
}
