"""How well the speech-to-text of the assistant understands spoken questions.

    python -m radmuenster.voice_eval data/voice
    python -m radmuenster.voice_eval data/voice --assistant

The folder holds recordings of questions and questions.csv with one row per
recording: file (the audio file), text (what was said) and, optionally, tool
(the tool the assistant should call). The recordings are not committed
(data/ is ignored): they are someone's voice.

For each recording it measures the word error rate of the transcript, whether
the site names were understood and how long the transcription took; with
--assistant it also asks the assistant the transcribed question and checks
whether it called the expected tool. The results go to results.csv in the
folder.

Needs OPENAI_API_KEY (and ANTHROPIC_API_KEY with --assistant) in the
environment.
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import pandas as pd

from radmuenster.sites import SITES
from radmuenster.voice import transcribe


def normalize(text: str) -> list[str]:
    """The words of text, for comparing: lower case (casefold also makes ß
    ss), without punctuation."""
    return re.sub(r"[^\w\s]", " ", text.casefold()).split()


def word_errors(reference: str, hypothesis: str) -> tuple[int, int]:
    """Substitutions, deletions and insertions to turn the reference into the
    hypothesis (word-level Levenshtein distance), and the number of
    reference words. The word error rate is the first divided by the second."""
    ref, hyp = normalize(reference), normalize(hypothesis)
    # previous[j]: edits between the reference words so far and hyp[:j]
    previous = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, start=1):
        current = [i]
        for j, hyp_word in enumerate(hyp, start=1):
            current.append(
                min(
                    previous[j] + 1,  # deletion
                    current[j - 1] + 1,  # insertion
                    previous[j - 1] + (ref_word != hyp_word),  # substitution or match
                )
            )
        previous = current
    return previous[-1], len(ref)


def site_keywords() -> list[str]:
    """What people say for a site: its name without the part in brackets or
    after the comma ("Kanalpromenade" for "Kanalpromenade, Abschnitt 5")."""
    keywords = {re.split(r"[(,]", site.name)[0].strip() for site in SITES}
    return sorted(keywords)


def sites_mentioned(text: str, keywords: list[str] | None = None) -> set[str]:
    """The site keywords that occur in text, as whole words."""
    words = f" {' '.join(normalize(text))} "
    return {k for k in keywords or site_keywords() if f" {' '.join(normalize(k))} " in words}


def evaluate(questions: pd.DataFrame, folder: Path, client, ask_assistant=None) -> pd.DataFrame:
    """One row per recording with the transcript and the measures.
    ask_assistant(question) returns the names of the tools the assistant
    called; without it the tools are not checked."""
    rows = []
    for question in questions.itertuples():
        audio = (folder / question.file).read_bytes()
        start = time.perf_counter()
        transcript = transcribe(client, audio, filename=question.file)
        seconds = time.perf_counter() - start
        errors, words = word_errors(question.text, transcript)
        said = sites_mentioned(question.text)
        row = {
            "file": question.file,
            "text": question.text,
            "transcript": transcript,
            "errors": errors,
            "words": words,
            "wer": errors / words if words else 0.0,
            "sites": len(said),
            "sites_understood": len(said & sites_mentioned(transcript)),
            "stt_seconds": round(seconds, 2),
        }
        expected = getattr(question, "tool", None)
        if ask_assistant is not None and isinstance(expected, str) and expected:
            tools = ask_assistant(transcript)
            row["tools"] = ", ".join(tools)
            row["tool_ok"] = expected in tools
        rows.append(row)
    return pd.DataFrame(rows)


def summary(results: pd.DataFrame) -> str:
    """The measures over all recordings. The word error rate is pooled: all
    errors divided by all reference words, so short questions do not count
    more than long ones."""
    if results.empty or not results["words"].sum():
        return f"Recordings:        {len(results)}, no reference text to compare with"
    lines = [
        f"Recordings:        {len(results)}",
        f"Word error rate:   {results['errors'].sum() / results['words'].sum():.1%}",
        f"Without errors:    {(results['errors'] == 0).mean():.0%} of the recordings",
    ]
    if results["sites"].sum():
        lines.append(
            f"Site names:        {results['sites_understood'].sum()} of {results['sites'].sum()} understood"
        )
    lines.append(f"Transcription:     median {results['stt_seconds'].median():.1f} s")
    if "tool_ok" in results:
        checked = results["tool_ok"].dropna()
        lines.append(f"Expected tool:     {int(checked.sum())} of {len(checked)} questions")
    return "\n".join(lines)


def _assistant():
    """ask_assistant for evaluate(), with the published data of today."""
    import anthropic

    from radmuenster.assistant import MODEL, ask
    from radmuenster.assistant_tools import read_assistant_data
    from radmuenster.forecast import today_in_muenster

    client = anthropic.Anthropic()
    data = read_assistant_data(today_in_muenster())

    def ask_assistant(question: str) -> list[str]:
        return [call.name for call in ask(client, data, question, MODEL, spoken=True).tool_calls]

    return ask_assistant


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, help="folder with the recordings and questions.csv")
    parser.add_argument("--assistant", action="store_true", help="also check which tools the assistant calls")
    args = parser.parse_args()

    import openai

    questions = pd.read_csv(args.folder / "questions.csv")
    results = evaluate(questions, args.folder, openai.OpenAI(), _assistant() if args.assistant else None)
    results.to_csv(args.folder / "results.csv", index=False)
    print(summary(results))
    print(f"\nDetails: {args.folder / 'results.csv'}")


if __name__ == "__main__":
    main()
