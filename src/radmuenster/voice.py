"""Speech in and out for the AI assistant.

A spoken question goes through three steps: speech-to-text (transcribe), the
assistant (assistant.ask with spoken=True) and text-to-speech (speak). The
language model in the middle is Claude; Claude does not hear or speak, so
both ends use OpenAI's audio models.

Like assistant.py, this module does not import the openai package: the
client is passed in (in the app the OpenAI client, in the tests a fake one
with the same audio.transcriptions.create() and audio.speech.create()).
"""

from __future__ import annotations

import io
import re
import wave

from radmuenster.sites import SITES

# The small transcription and speech models: questions are short, and
# speed matters more in a conversation than the last bit of quality
# (decided 2026-10-24; voice_eval measures the transcription)
STT_MODEL = "gpt-4o-mini-transcribe"
TTS_MODEL = "gpt-4o-mini-tts"
TTS_VOICE = "nova"
TTS_INSTRUCTIONS = (
    "Speak in a calm, even voice at a moderate pace. "
    "Pronounce German street names the German way."
)

# Limits against cost and misuse, like the limits of the text questions.
# A question takes a few seconds; the audio input records 16 kHz mono.
MAX_AUDIO_SECONDS = 30
# The length of a recording that cannot be read as WAV is unknown; its size
# still has to be limited. 30 s of 48 kHz stereo WAV are 5.8 MB.
MAX_AUDIO_BYTES = 6_000_000
# Spoken answers are short (assistant.SPOKEN_NOTE); a longer text is cut at
# the last sentence that fits. The API takes at most 4,096 characters.
MAX_SPEECH_CHARS = 1500

# The transcription model takes a prompt with words it should expect: the
# site names are German and partly rare words ("Gasselstiege",
# "Lütkenbecker Weg") that it would otherwise spell as it hears them. Only
# the names, without a German sentence: the model follows the language of
# the prompt and would turn an English question into German.
TRANSCRIPTION_PROMPT = ", ".join(site.name for site in SITES) + "."


class AudioError(ValueError):
    """The recording cannot be used. The message is meant for the user."""


def audio_seconds(audio: bytes) -> float | None:
    """The length of a WAV recording in seconds; None for other formats,
    whose length is not checked."""
    try:
        with wave.open(io.BytesIO(audio)) as recording:
            return recording.getnframes() / recording.getframerate()
    # A header with a frame rate of 0 is broken too
    except (wave.Error, EOFError, ZeroDivisionError):
        return None


def check_recording(audio: bytes) -> None:
    """Raises AudioError if the recording is empty or too long."""
    seconds = audio_seconds(audio)
    if not audio or seconds == 0:
        raise AudioError("The recording is empty. Please try again.")
    too_long = seconds > MAX_AUDIO_SECONDS if seconds is not None else len(audio) > MAX_AUDIO_BYTES
    if too_long:
        raise AudioError(f"Please ask in at most {MAX_AUDIO_SECONDS} seconds.")


def transcribe(
    client,
    audio: bytes,
    filename: str = "question.wav",
    model: str = STT_MODEL,
    language: str | None = None,
) -> str:
    """The text of a spoken question. filename only tells the API the
    format. Without language the model detects it, so that the assistant
    can answer in the language of the question."""
    request = {
        "model": model,
        "file": (filename, audio),
        "prompt": TRANSCRIPTION_PROMPT,
        "response_format": "text",
    }
    if language:
        request["language"] = language
    result = client.audio.transcriptions.create(**request)
    # response_format text returns a string; older SDKs an object with .text
    return (result if isinstance(result, str) else result.text).strip()


# Markdown that would be read aloud as characters
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_EMPHASIS = re.compile(r"(\*\*|__|\*|`)")
_HEADING = re.compile(r"^\s{0,3}#+\s*", re.MULTILINE)
_BULLET = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+", re.MULTILINE)
_TABLE_RULE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")


def _table_row(line: str) -> str:
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return ", ".join(cell for cell in cells if cell) + "."


def speakable(text: str, max_chars: int = MAX_SPEECH_CHARS) -> str:
    """The answer as plain text for text-to-speech: without markdown, table
    rows as sentences, and cut at the last sentence within max_chars."""
    lines = []
    for line in text.splitlines():
        if _TABLE_RULE.match(line):
            continue
        if line.lstrip().startswith("|"):
            line = _table_row(line)
        lines.append(line)
    text = "\n".join(lines)
    text = _LINK.sub(r"\1", text)
    text = _HEADING.sub("", text)
    text = _BULLET.sub("", text)
    text = _EMPHASIS.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    return cut[: end + 1] if end > 0 else cut.rsplit(" ", 1)[0]


def speak(
    client,
    text: str,
    model: str = TTS_MODEL,
    voice: str = TTS_VOICE,
) -> bytes:
    """The answer read aloud, as MP3."""
    response = client.audio.speech.create(
        model=model,
        voice=voice,
        input=speakable(text),
        instructions=TTS_INSTRUCTIONS,
        response_format="mp3",
    )
    return response.content
