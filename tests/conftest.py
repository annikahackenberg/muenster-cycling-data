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


def _category_axes(spec):
    """The axis of every nominal y encoding in an Altair spec (as dict)."""
    if isinstance(spec, dict):
        y = spec.get("encoding", {}).get("y", {})
        if y.get("type") == "nominal":
            yield y.get("axis", {})
        for value in spec.values():
            yield from _category_axes(value)
    elif isinstance(spec, list):
        for value in spec:
            yield from _category_axes(value)


@pytest.fixture
def assert_all_labels_shown():
    """Checks that a horizontal bar chart keeps every label in Streamlit
    (see explore.category_axis): called with the chart."""

    def check(chart):
        axes = list(_category_axes(chart.to_dict()))
        assert axes
        for axis in axes:
            assert axis.get("labelOverlap") is False and axis.get("labelFlush") is False

    return check
