import pandas as pd
import pytest

from radmuenster.voice_eval import (
    evaluate,
    normalize,
    site_keywords,
    sites_mentioned,
    summary,
    word_errors,
)


def test_normalize():
    assert normalize("Wie viele Fahrten, Weißenburgstraße?") == ["wie", "viele", "fahrten", "weissenburgstrasse"]


@pytest.mark.parametrize(
    ("reference", "hypothesis", "errors"),
    [
        ("wie viele fahrten am neutor", "Wie viele Fahrten am Neutor?", 0),
        ("wie viele fahrten am neutor", "wie viele fahrten am neuen tor", 2),  # 1 substitution, 1 insertion
        ("wie viele fahrten am neutor", "wie fahrten am neutor", 1),  # 1 deletion
        ("wie viele fahrten", "", 3),
        ("", "hallo", 1),
    ],
)
def test_word_errors(reference, hypothesis, errors):
    assert word_errors(reference, hypothesis) == (errors, len(reference.split()))


def test_site_keywords():
    keywords = site_keywords()
    assert "Kanalpromenade" in keywords
    assert "Promenade" in keywords
    assert "Kanalpromenade, Abschnitt 5" not in keywords
    assert len(keywords) == len(set(keywords))


def test_sites_mentioned_matches_whole_words():
    assert sites_mentioned("Wie voll war die Kanalpromenade?") == {"Kanalpromenade"}
    assert sites_mentioned("Und an der Promenade und am Neutor?") == {"Promenade", "Neutor"}
    assert sites_mentioned("am Neuen Tor") == set()


def test_evaluate(tmp_path, audio_client):
    (tmp_path / "one.wav").write_bytes(b"audio")
    questions = pd.DataFrame(
        {"file": ["one.wav"], "text": ["Wie viele Fahrten am Neutor?"], "tool": ["get_counts"]}
    )
    asked = []

    def ask_assistant(question):
        asked.append(question)
        return ["list_sites", "get_counts"]

    client = audio_client(transcript="Wie viele Fahrten am neuen Tor?")
    [row] = evaluate(questions, tmp_path, client, ask_assistant).to_dict("records")
    assert client.requests[0]["file"] == ("one.wav", b"audio")
    assert asked == ["Wie viele Fahrten am neuen Tor?"]
    assert (row["errors"], row["words"], row["wer"]) == (2, 5, 0.4)
    assert (row["sites"], row["sites_understood"]) == (1, 0)
    assert row["tool_ok"]


def test_summary_pools_the_errors():
    results = pd.DataFrame(
        {"errors": [0, 3], "words": [10, 2], "sites": [1, 1], "sites_understood": [1, 0], "stt_seconds": [0.5, 1.5]}
    )
    text = summary(results)
    assert "25.0%" in text  # 3 errors of 12 words, not the mean of 0% and 150%
    assert "1 of 2 understood" in text
    assert "Expected tool" not in text


def test_summary_without_reference_text():
    assert "no reference text" in summary(pd.DataFrame())
    assert "no reference text" in summary(pd.DataFrame({"errors": [1], "words": [0]}))
