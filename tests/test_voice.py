import io
import wave
from types import SimpleNamespace

import pytest

from radmuenster.voice import (
    MAX_AUDIO_SECONDS,
    STT_MODEL,
    TRANSCRIPTION_PROMPT,
    AudioError,
    audio_seconds,
    check_recording,
    speak,
    speakable,
    transcribe,
)


def wav(seconds: float, rate: int = 16000) -> bytes:
    """Silence as the audio input records it: 16 kHz mono, 16 bit."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as recording:
        recording.setnchannels(1)
        recording.setsampwidth(2)
        recording.setframerate(rate)
        recording.writeframes(b"\0\0" * int(seconds * rate))
    return buffer.getvalue()


class FakeAudioClient:
    """Plays OpenAI's audio API: client.audio.transcriptions.create() and
    client.audio.speech.create(), keeping a copy of every request."""

    def __init__(self, transcript="Wie viele Fahrten am Neutor?"):
        self.requests = []
        self.audio = SimpleNamespace(
            transcriptions=SimpleNamespace(create=self._transcribe),
            speech=SimpleNamespace(create=self._speak),
        )
        self.transcript = transcript

    def _transcribe(self, **request):
        self.requests.append(request)
        return self.transcript

    def _speak(self, **request):
        self.requests.append(request)
        return SimpleNamespace(content=b"mp3")


def test_audio_seconds():
    assert audio_seconds(wav(2.5)) == 2.5
    assert audio_seconds(b"not a wav file") is None


def test_check_recording():
    check_recording(wav(5))
    # Other formats cannot be measured and pass
    check_recording(b"mp3 data")
    with pytest.raises(AudioError, match="empty"):
        check_recording(b"")
    with pytest.raises(AudioError, match="empty"):
        check_recording(wav(0))
    with pytest.raises(AudioError, match=f"{MAX_AUDIO_SECONDS} seconds"):
        check_recording(wav(MAX_AUDIO_SECONDS + 1))


def test_transcribe_sends_the_site_names():
    client = FakeAudioClient(transcript="  Wie viele Fahrten am Neutor?\n")
    assert transcribe(client, b"audio") == "Wie viele Fahrten am Neutor?"
    [request] = client.requests
    assert request["model"] == STT_MODEL
    assert request["file"] == ("question.wav", b"audio")
    assert request["prompt"] == TRANSCRIPTION_PROMPT
    assert "Gasselstiege" in TRANSCRIPTION_PROMPT
    # The model detects the language unless one is given
    assert "language" not in request
    transcribe(client, b"audio", language="de")
    assert client.requests[-1]["language"] == "de"


def test_transcribe_takes_an_object_with_text():
    client = FakeAudioClient(transcript=SimpleNamespace(text="Hallo"))
    assert transcribe(client, b"audio") == "Hallo"


def test_speakable_removes_markdown():
    text = "## Morgen\n\n**Etwa 4.300 Fahrten** am [Neutor](https://example.org).\n\n- Regen: `2 mm`\n1. Kühl"
    assert speakable(text) == "Morgen Etwa 4.300 Fahrten am Neutor. Regen: 2 mm Kühl"


def test_speakable_reads_table_rows_as_sentences():
    text = "| Site | Trips |\n|---|---:|\n| Neutor | 4,300 |\n| Bohlweg | 2,100 |"
    assert speakable(text) == "Site, Trips. Neutor, 4,300. Bohlweg, 2,100."


def test_speakable_cuts_at_a_sentence():
    text = "One sentence. Two sentences. Three sentences."
    assert speakable(text, max_chars=30) == "One sentence. Two sentences."
    # Without a sentence end, at a word
    assert speakable("one two three four", max_chars=10) == "one two"


def test_speak():
    client = FakeAudioClient()
    assert speak(client, "**Etwa 4.300** Fahrten.") == b"mp3"
    [request] = client.requests
    assert request["input"] == "Etwa 4.300 Fahrten."
    assert request["response_format"] == "mp3"
