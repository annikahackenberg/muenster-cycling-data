from types import SimpleNamespace

import pytest


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


@pytest.fixture
def audio_client():
    """FakeAudioClient, to be called with the transcript."""
    return FakeAudioClient
