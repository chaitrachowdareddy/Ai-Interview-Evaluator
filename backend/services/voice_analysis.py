"""Phase 3 voice analysis: objective acoustic measurements, not inferences.

What is measured
----------------
- Duration, source sample rate/channels when the decoder exposes them.
- RMS energy (linear and dBFS), its variability, peak amplitude, clipping.
- Energy-based voice activity using ``librosa.effects.split``. "Speech" here
  means non-silent audio above the threshold; it is not speaker recognition.
- Pauses: silent gaps of at least ``VOICE_MIN_PAUSE_SECONDS`` between detected
  speech segments. Leading/trailing silence is reported separately as silence.
- Speaking rate: transcript word count divided by the duration of detected
  speech (words per minute, silence excluded). The saved transcript is used,
  so it is an estimate for the recording, not a second transcription.
- Pitch (F0) with ``librosa.pyin`` on the first
  ``VOICE_PITCH_MAX_SECONDS`` of detected speech. Values are reported only when
  enough voiced frames exist; otherwise pitch fields stay null.

These numbers describe audio. They must not be presented as confidence,
personality, honesty, intelligence, or emotion.
"""

from __future__ import annotations

import importlib.util
import logging
import math
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import (
    VOICE_MAX_ANALYSIS_SECONDS,
    VOICE_MIN_PAUSE_SECONDS,
    VOICE_MIN_SPEECH_SECONDS,
    VOICE_MIN_VOICED_SECONDS,
    VOICE_PITCH_MAX_HZ,
    VOICE_PITCH_MAX_SECONDS,
    VOICE_PITCH_MIN_HZ,
    VOICE_TARGET_SAMPLE_RATE,
    VOICE_VAD_TOP_DB,
)
from services.nlp_preprocessing import normalize_text, tokenize

logger = logging.getLogger(__name__)

STATUS_COMPLETED = "completed"
STATUS_INSUFFICIENT_SPEECH = "insufficient_speech"
STATUS_UNAVAILABLE = "unavailable"
STATUS_FAILED = "failed"

_MIN_SPEAKING_RATE_SPEECH_SECONDS = 1.0
_MIN_PITCH_ANALYSIS_SECONDS = 0.5
_MIN_VOICED_FRAMES = 10
# -100 dBFS. Anything quieter has no usable speech energy and is treated as
# digital silence, which avoids relative VAD thresholds inventing speech.
_MIN_AUDIBLE_AMPLITUDE = 1e-5
_FFMPEG_TIMEOUT_SECONDS = 120


class VoiceAnalysisUnavailable(Exception):
    """The audio-analysis stack is not installed or cannot run."""

    status_code = 503

    def __init__(self, message: str | None = None) -> None:
        self.message = message or (
            "Voice analysis is unavailable. Install the backend analysis "
            "dependencies (librosa, soundfile, imageio-ffmpeg) and restart "
            "FastAPI."
        )
        super().__init__(self.message)


class VoiceAnalysisError(Exception):
    """The stored recording could not be decoded/analysed safely."""

    status_code = 422

    def __init__(self, message: str | None = None) -> None:
        self.message = message or (
            "The stored recording could not be decoded. Re-record or upload a "
            "supported audio file."
        )
        super().__init__(self.message)


@dataclass(frozen=True)
class VoiceAnalysisResult:
    status: str
    status_detail: str | None
    duration_seconds: float | None
    analyzed_duration_seconds: float | None
    sample_rate_hz: int | None
    channels: int | None
    truncated_for_analysis: bool
    speech_detected: bool
    speech_duration_seconds: float | None
    silence_duration_seconds: float | None
    speech_ratio_percent: float | None
    pause_count: int | None
    total_pause_seconds: float | None
    mean_pause_seconds: float | None
    longest_pause_seconds: float | None
    pauses_per_minute: float | None
    word_count: int | None
    speaking_rate_wpm: float | None
    rms_mean: float | None
    rms_std: float | None
    rms_variation_percent: float | None
    rms_mean_dbfs: float | None
    peak_amplitude: float | None
    clipping_percent: float | None
    pitch_median_hz: float | None
    pitch_mean_hz: float | None
    pitch_std_hz: float | None
    pitch_p10_hz: float | None
    pitch_p90_hz: float | None
    pitch_range_hz: float | None
    pitch_voiced_seconds: float | None
    pitch_reliable: bool
    quality_notes: tuple[str, ...]


def _round(value: float | None, digits: int = 2) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return round(number, digits)


def _dbfs(amplitude: float) -> float:
    if amplitude <= 0:
        return -120.0
    return max(-120.0, 20.0 * math.log10(amplitude))


def voice_analysis_available() -> bool:
    """Cheap dependency check used by ``/health`` (does not import librosa)."""

    for module_name in ("librosa", "soundfile", "imageio_ffmpeg"):
        if importlib.util.find_spec(module_name) is None:
            return False

    try:
        import imageio_ffmpeg

        return Path(imageio_ffmpeg.get_ffmpeg_exe()).is_file()
    except Exception:  # noqa: BLE001 - health must never fail because of this
        return False


def _import_audio_stack():
    try:
        import imageio_ffmpeg
        import librosa
        import soundfile
    except Exception as exc:  # noqa: BLE001 - report a safe dependency error
        raise VoiceAnalysisUnavailable() from exc
    return librosa, soundfile, imageio_ffmpeg


def _decode_with_soundfile(soundfile, path: Path):
    info = soundfile.info(str(path))
    if info.frames <= 0 or info.samplerate <= 0:
        raise ValueError("empty audio stream")

    max_frames = int(VOICE_MAX_ANALYSIS_SECONDS * info.samplerate) + 1
    frames_to_read = min(info.frames, max_frames)
    data, sample_rate = soundfile.read(
        str(path),
        frames=frames_to_read,
        dtype="float32",
        always_2d=True,
    )
    if data.size == 0:
        raise ValueError("empty audio samples")

    channels = int(data.shape[1])
    mono = data.mean(axis=1, dtype=np.float32)
    full_duration = info.frames / float(info.samplerate)
    analyzed_duration = len(mono) / float(info.samplerate)
    return (
        mono,
        int(info.samplerate),
        int(info.samplerate),
        channels,
        full_duration,
        full_duration > analyzed_duration + 0.05,
        None,
    )


def _decode_with_ffmpeg(imageio_ffmpeg, soundfile, path: Path):
    """Decode containers/codecs that libsndfile cannot read (WebM, M4A, ...).

    The output is mono 16 kHz because that is the analysis rate. Source sample
    rate/channel count are intentionally reported as unknown rather than
    guessed. The temporary WAV is bounded by VOICE_MAX_ANALYSIS_SECONDS.
    """

    try:
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # noqa: BLE001
        raise VoiceAnalysisUnavailable(
            "The bundled FFmpeg executable could not be located."
        ) from exc

    with tempfile.TemporaryDirectory(prefix="ai-voice-") as temporary_dir:
        output = Path(temporary_dir) / "decoded.wav"
        command = [
            ffmpeg,
            "-nostdin",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-vn",
            "-t",
            str(VOICE_MAX_ANALYSIS_SECONDS),
            "-ac",
            "1",
            "-ar",
            str(VOICE_TARGET_SAMPLE_RATE),
            "-f",
            "wav",
            str(output),
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                timeout=_FFMPEG_TIMEOUT_SECONDS,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise VoiceAnalysisError(
                "The stored recording could not be decoded within the time limit."
            ) from exc

        if completed.returncode != 0 or not output.is_file():
            raise VoiceAnalysisError(
                "The stored recording could not be decoded. It may be "
                "malformed or use an unsupported codec."
            )

        try:
            data, sample_rate = soundfile.read(
                str(output),
                dtype="float32",
                always_2d=True,
            )
        except Exception as exc:  # noqa: BLE001 - decoder details stay internal
            raise VoiceAnalysisError(
                "The stored recording could not be decoded after conversion."
            ) from exc

    if data.size == 0:
        raise VoiceAnalysisError("The stored recording contains no audio samples.")

    mono = data.mean(axis=1, dtype=np.float32)
    duration = len(mono) / float(sample_rate)
    truncated = duration >= VOICE_MAX_ANALYSIS_SECONDS - 0.05
    return (
        mono,
        int(sample_rate),
        None,
        None,
        duration,
        truncated,
        "The recording used a compressed/container format and was decoded "
        "with the bundled FFmpeg for analysis.",
    )


def _decode_audio(librosa, soundfile, imageio_ffmpeg, path: Path):
    try:
        return _decode_with_soundfile(soundfile, path)
    except Exception:  # noqa: BLE001 - fall back to FFmpeg for containers
        decoded = _decode_with_ffmpeg(imageio_ffmpeg, soundfile, path)

    mono, sample_rate, source_rate, channels, duration, truncated, note = decoded
    if not np.all(np.isfinite(mono)):
        mono = np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0)

    if sample_rate != VOICE_TARGET_SAMPLE_RATE:
        try:
            mono = librosa.resample(
                mono,
                orig_sr=sample_rate,
                target_sr=VOICE_TARGET_SAMPLE_RATE,
                res_type="soxr_hq",
            )
        except Exception as exc:  # noqa: BLE001
            raise VoiceAnalysisError(
                "The recording could not be resampled for analysis."
            ) from exc
        sample_rate = VOICE_TARGET_SAMPLE_RATE

    return (
        np.asarray(mono, dtype=np.float32),
        sample_rate,
        source_rate,
        channels,
        duration,
        truncated,
        note,
    )


def _word_count(transcript: str | None) -> int | None:
    text = normalize_text(transcript)
    if not text:
        return None
    return len(tokenize(text))


def _pause_statistics(
    intervals: np.ndarray,
    sample_rate: int,
    speech_duration: float,
) -> tuple[int, float, float | None, float | None, float | None]:
    gaps: list[float] = []
    for previous, following in zip(intervals, intervals[1:], strict=False):
        gap = (int(following[0]) - int(previous[1])) / float(sample_rate)
        if gap >= VOICE_MIN_PAUSE_SECONDS:
            gaps.append(gap)

    if not gaps:
        return 0, 0.0, None, None, None

    total = sum(gaps)
    mean = total / len(gaps)
    longest = max(gaps)
    per_minute = (
        len(gaps) / (speech_duration / 60.0)
        if speech_duration > 0
        else None
    )
    return len(gaps), total, mean, longest, per_minute


def _pitch_statistics(
    librosa,
    speech_audio: np.ndarray,
    sample_rate: int,
    analyzed_seconds: float,
):
    if (
        speech_audio.size == 0
        or analyzed_seconds < _MIN_PITCH_ANALYSIS_SECONDS
    ):
        return None, None, None, None, None, None, None, False

    try:
        f0, voiced_flag, _voiced_probability = librosa.pyin(
            speech_audio,
            fmin=VOICE_PITCH_MIN_HZ,
            fmax=VOICE_PITCH_MAX_HZ,
            sr=sample_rate,
            frame_length=1024,
            hop_length=256,
        )
    except Exception:  # noqa: BLE001 - pitch is optional, never fail analysis
        return None, None, None, None, None, None, None, False

    hop_seconds = 256 / float(sample_rate)
    valid = np.isfinite(f0) & np.asarray(voiced_flag, dtype=bool)
    values = f0[valid]
    voiced_seconds = float(len(values) * hop_seconds)

    if (
        len(values) < _MIN_VOICED_FRAMES
        or voiced_seconds < VOICE_MIN_VOICED_SECONDS
    ):
        return None, None, None, None, None, None, voiced_seconds, False

    reliable_ratio = voiced_seconds / analyzed_seconds
    if reliable_ratio < 0.1:
        return None, None, None, None, None, None, voiced_seconds, False

    p10, p90 = np.percentile(values, [10, 90])
    return (
        float(np.median(values)),
        float(np.mean(values)),
        float(np.std(values)),
        float(p10),
        float(p90),
        float(p90 - p10),
        voiced_seconds,
        True,
    )


def _quality_notes(
    *,
    speech_detected: bool,
    speech_ratio: float,
    rms_mean_dbfs: float,
    clipping_percent: float,
    truncated: bool,
    channels: int | None,
    decoder_note: str | None,
    pitch_reliable: bool,
    word_count: int | None,
) -> tuple[str, ...]:
    notes: list[str] = []

    if not speech_detected:
        notes.append(
            "No speech was detected above the energy threshold; speaking "
            "rate, pauses, and pitch are unavailable."
        )
    elif speech_ratio < 20:
        notes.append(
            f"Detected speech occupies only {speech_ratio:.1f}% of the "
            "analysed audio."
        )

    if rms_mean_dbfs <= -45:
        notes.append(
            "The average recording level is very low "
            f"({rms_mean_dbfs:.1f} dBFS)."
        )

    if clipping_percent >= 0.5:
        notes.append(
            f"About {clipping_percent:.1f}% of decoded samples are at or near "
            "full scale (clipping)."
        )

    if truncated:
        notes.append(
            f"Only the first {VOICE_MAX_ANALYSIS_SECONDS} seconds were "
            "analysed; duration statistics describe the analysed portion."
        )

    if channels is not None and channels > 1:
        notes.append(
            f"The recording has {channels} channels and was downmixed to mono "
            "for analysis."
        )

    if decoder_note:
        notes.append(decoder_note)

    if speech_detected and not pitch_reliable:
        notes.append(
            "Pitch could not be estimated reliably from the detected voiced "
            "audio."
        )

    if word_count is None:
        notes.append(
            "No transcript is saved, so speaking rate is unavailable."
        )
    elif word_count == 0:
        notes.append(
            "The saved transcript contains no words, so speaking rate is "
            "unavailable."
        )

    return tuple(notes)


def analyze_audio(
    path: Path,
    transcript: str | None,
) -> VoiceAnalysisResult:
    """Measure one stored recording. Never interprets the measurements."""

    librosa, soundfile, imageio_ffmpeg = _import_audio_stack()

    (
        audio,
        sample_rate,
        source_rate,
        channels,
        full_duration,
        truncated,
        decoder_note,
    ) = _decode_audio(librosa, soundfile, imageio_ffmpeg, path)

    if audio.size == 0:
        raise VoiceAnalysisError("The stored recording contains no audio samples.")

    analyzed_duration = len(audio) / float(sample_rate)
    duration = (
        float(full_duration)
        if full_duration is not None
        else analyzed_duration
    )

    frame_length = max(1, int(0.025 * sample_rate))
    hop_length = max(1, int(0.010 * sample_rate))

    try:
        rms_frames = librosa.feature.rms(
            y=audio,
            frame_length=frame_length,
            hop_length=hop_length,
        )
    except Exception as exc:  # noqa: BLE001
        raise VoiceAnalysisError(
            "The recording's energy could not be measured."
        ) from exc

    rms_mean = float(np.mean(rms_frames))
    rms_std = float(np.std(rms_frames))
    rms_variation = (
        (rms_std / rms_mean) * 100.0 if rms_mean > 1e-6 else None
    )
    peak_amplitude = float(np.max(np.abs(audio)))
    clipping_percent = float(np.mean(np.abs(audio) >= 0.999) * 100.0)

    try:
        intervals = librosa.effects.split(
            audio,
            top_db=VOICE_VAD_TOP_DB,
            frame_length=frame_length,
            hop_length=hop_length,
        )
    except Exception as exc:  # noqa: BLE001
        raise VoiceAnalysisError(
            "Voice activity could not be detected in this recording."
        ) from exc

    # ``librosa.effects.split`` uses a relative threshold. A digitally silent
    # file has no reference level, so it can be returned as one long "speech"
    # interval. Treat a signal with no measurable amplitude as silence instead
    # of inventing speech, pauses, or speaking rate.
    if peak_amplitude < _MIN_AUDIBLE_AMPLITUDE:
        intervals = np.empty((0, 2), dtype=int)

    speech_duration = min(
        analyzed_duration,
        float(
            sum(
                (int(end) - int(start)) / float(sample_rate)
                for start, end in intervals
                if int(end) > int(start)
            )
        ),
    )
    silence_duration = max(0.0, analyzed_duration - speech_duration)
    speech_ratio = (
        (speech_duration / analyzed_duration) * 100.0
        if analyzed_duration > 0
        else 0.0
    )
    speech_detected = speech_duration >= VOICE_MIN_SPEECH_SECONDS

    if speech_detected:
        (
            pause_count,
            total_pause,
            mean_pause,
            longest_pause,
            pauses_per_minute,
        ) = _pause_statistics(intervals, sample_rate, speech_duration)
    else:
        pause_count = 0
        total_pause = 0.0
        mean_pause = None
        longest_pause = None
        pauses_per_minute = None

    words = _word_count(transcript)
    speaking_rate = None
    if (
        words is not None
        and words > 0
        and speech_detected
        and speech_duration >= _MIN_SPEAKING_RATE_SPEECH_SECONDS
    ):
        speaking_rate = words / (speech_duration / 60.0)

    speech_chunks = [
        audio[int(start) : int(end)]
        for start, end in intervals
        if int(end) > int(start)
    ]
    pitch_audio = (
        np.concatenate(speech_chunks)
        if speech_chunks
        else np.array([], dtype=np.float32)
    )

    if pitch_audio.size:
        pitch_audio = pitch_audio[
            : int(VOICE_PITCH_MAX_SECONDS * sample_rate)
        ]

    (
        pitch_median,
        pitch_mean,
        pitch_std,
        pitch_p10,
        pitch_p90,
        pitch_range,
        pitch_voiced_seconds,
        pitch_reliable,
    ) = _pitch_statistics(
        librosa,
        pitch_audio,
        sample_rate,
        len(pitch_audio) / float(sample_rate),
    )

    rms_mean_dbfs = _dbfs(rms_mean)
    notes = _quality_notes(
        speech_detected=speech_detected,
        speech_ratio=speech_ratio,
        rms_mean_dbfs=rms_mean_dbfs,
        clipping_percent=clipping_percent,
        truncated=truncated,
        channels=channels,
        decoder_note=decoder_note,
        pitch_reliable=pitch_reliable,
        word_count=words,
    )

    status = STATUS_COMPLETED if speech_detected else STATUS_INSUFFICIENT_SPEECH
    status_detail = (
        None
        if speech_detected
        else (
            "No speech was detected above the energy threshold. Duration and "
            "energy are valid, but speaking rate, pauses, and pitch are not "
            "estimable."
        )
    )

    return VoiceAnalysisResult(
        status=status,
        status_detail=status_detail,
        duration_seconds=_round(duration),
        analyzed_duration_seconds=_round(analyzed_duration),
        sample_rate_hz=source_rate,
        channels=channels,
        truncated_for_analysis=truncated,
        speech_detected=speech_detected,
        speech_duration_seconds=_round(speech_duration),
        silence_duration_seconds=_round(silence_duration),
        speech_ratio_percent=_round(speech_ratio),
        pause_count=pause_count,
        total_pause_seconds=_round(total_pause),
        mean_pause_seconds=_round(mean_pause),
        longest_pause_seconds=_round(longest_pause),
        pauses_per_minute=_round(pauses_per_minute),
        word_count=words,
        speaking_rate_wpm=_round(speaking_rate),
        rms_mean=_round(rms_mean, 4),
        rms_std=_round(rms_std, 4),
        rms_variation_percent=_round(rms_variation),
        rms_mean_dbfs=_round(rms_mean_dbfs),
        peak_amplitude=_round(peak_amplitude, 4),
        clipping_percent=_round(clipping_percent),
        pitch_median_hz=_round(pitch_median),
        pitch_mean_hz=_round(pitch_mean),
        pitch_std_hz=_round(pitch_std),
        pitch_p10_hz=_round(pitch_p10),
        pitch_p90_hz=_round(pitch_p90),
        pitch_range_hz=_round(pitch_range),
        pitch_voiced_seconds=_round(pitch_voiced_seconds),
        pitch_reliable=pitch_reliable,
        quality_notes=notes,
    )
