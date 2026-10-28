"""Assistant page: questions about the data, answered by a language model
that calls the tools of radmuenster.assistant_tools. Questions can be typed
or spoken; a spoken question gets a spoken answer (radmuenster.voice)."""

from __future__ import annotations

import json
import logging
import time
from datetime import date

import anthropic
import streamlit as st

try:
    import openai
except ImportError:
    # Voice is optional: without the package, questions can only be typed
    openai = None

from radmuenster.assistant import (
    MAX_QUESTION_CHARS,
    MAX_QUESTIONS_PER_CONVERSATION,
    MAX_QUESTIONS_PER_DAY,
    MODEL,
    DailyLimit,
    ask,
    conversation_problem,
    question_problem,
)
from radmuenster.assistant_tools import AssistantData, read_assistant_data
from radmuenster.forecast import today_in_muenster
from radmuenster.voice import AudioError, check_recording, speak, transcribe

# The cached data is reloaded at most once an hour
CACHE_SECONDS = 3600

EXAMPLES = [
    "How many trips are expected tomorrow, and why?",
    "Which counting site was busiest last month?",
    "How much does rain reduce cycling?",
    "Were there any data problems at the Neutor this year?",
]

DAY_LIMIT_TEXT = "The assistant has answered as many questions as it can today. Please come back tomorrow."

logger = logging.getLogger(__name__)

# Streamlit's chat input has no visible border and looks like a hint; a grey
# that works on both themes makes it look like a field to type in.
# A table in an answer can be wider than a phone screen, and the chat message
# cuts off its last column; display: block lets the table scroll sideways.
PAGE_STYLE = """<style>
[data-testid="stChatInput"] > div { border: 2px solid rgba(128, 128, 128, 0.7); }
[data-testid="stChatMessage"] table { display: block; max-width: 100%; overflow-x: auto; }
</style>"""


@st.cache_data(ttl=CACHE_SECONDS)
def load_data(today: date) -> AssistantData:
    """Everything the tools read, as published by the daily job. today is part
    of the cache key, so that a new day loads new data."""
    return read_assistant_data(today)


@st.cache_resource
def get_client(api_key: str) -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=api_key, timeout=60)


@st.cache_resource
def get_voice_client(api_key: str) -> openai.OpenAI:
    return openai.OpenAI(api_key=api_key, timeout=30)


@st.cache_resource
def daily_limit() -> DailyLimit:
    # cache_resource: one object shared by all sessions, so all users count
    # towards the same limit
    return DailyLimit()


def api_key(name: str = "ANTHROPIC_API_KEY") -> str | None:
    """The key from .streamlit/secrets.toml (locally) or the app's secrets
    (Streamlit Cloud); None if there is none."""
    try:
        return st.secrets[name]
    except (FileNotFoundError, KeyError):
        return None


def timings(seconds: dict[str, float]) -> str:
    """How long each step of an answer took, e.g. "transcription 0.8 s, answer 3.1 s"."""
    return ", ".join(f"{step} {value:.1f} s" for step, value in seconds.items())


def show(turn: dict) -> None:
    """One chat message, with the tool calls behind an answer."""
    with st.chat_message(turn["role"]):
        st.markdown(turn["text"])
        if turn.get("spoken"):
            st.caption("Spoken, as transcribed")
        if turn.get("audio"):
            # Only a new answer plays by itself, not the earlier ones when the
            # page is drawn again
            st.audio(turn["audio"], format="audio/mpeg", autoplay=turn.pop("autoplay", False))
        elif turn.get("speech_failed"):
            st.caption("This answer could not be read aloud.")
        if turn.get("tool_calls") is not None:
            with st.expander(f"How this answer was found ({len(turn['tool_calls'])} tool calls)"):
                for call in turn["tool_calls"]:
                    arguments = json.dumps(call.arguments, ensure_ascii=False)
                    st.code(f"{call.name}({arguments})", language=None)
                    if call.error:
                        st.caption(f"Error: {call.error}")
                st.caption(f"{turn['input_tokens']:,} input and {turn['output_tokens']:,} output tokens")
                if turn.get("seconds"):
                    st.caption(timings(turn["seconds"]))


def transcribe_recording(voice_client: openai.OpenAI, recording: bytes) -> tuple[str, float] | None:
    """The spoken question and how long the transcription took, or None
    (with a message on the page) if it cannot be used."""
    try:
        check_recording(recording)
        with st.spinner("Listening …"):
            start = time.perf_counter()
            question = transcribe(voice_client, recording)
            seconds = time.perf_counter() - start
    except AudioError as error:
        st.warning(str(error))
        return None
    except openai.APIError as error:
        logger.exception("Transcription failed")
        st.error(f"The voice input is not available right now ({type(error).__name__}). Please type your question.")
        return None
    if not question:
        st.warning("No question was understood. Please try again.")
        return None
    return question, seconds


def answer_question(
    client: anthropic.Anthropic,
    data: AssistantData,
    question: str,
    voice_client: openai.OpenAI | None = None,
    stt_seconds: float | None = None,
) -> None:
    """Asks the model, adds both turns to the conversation and draws the page
    again. With voice_client, the question was spoken and the answer is read
    aloud too."""
    spoken = voice_client is not None
    user_turn = {"role": "user", "text": question, "spoken": spoken}
    show(user_turn)
    seconds = {"transcription": stt_seconds} if spoken else {}
    with st.spinner("Looking at the data …"):
        try:
            start = time.perf_counter()
            answer = ask(client, data, question, MODEL, history=st.session_state.history, spoken=spoken)
            seconds["answer"] = time.perf_counter() - start
        except anthropic.APIError as error:
            # The history stays as it was, so the next question works
            logger.exception("Assistant request failed")
            st.error(f"The assistant is not available right now ({type(error).__name__}). Please try again later.")
            st.stop()
    for call in answer.tool_calls:
        if call.error:
            logger.warning("Tool %s(%s) failed: %s", call.name, call.arguments, call.error)
    audio = None
    speech_failed = False
    if spoken:
        with st.spinner("Speaking …"):
            try:
                start = time.perf_counter()
                audio = speak(voice_client, answer.text)
                seconds["speech"] = time.perf_counter() - start
            except openai.APIError:
                # The answer is there as text; it only cannot be read aloud
                logger.exception("Speech failed")
                speech_failed = True
    assistant_turn = {
        "role": "assistant",
        "text": answer.text,
        "tool_calls": answer.tool_calls,
        "input_tokens": answer.input_tokens,
        "output_tokens": answer.output_tokens,
        "audio": audio,
        "autoplay": True,
        "speech_failed": speech_failed,
        "seconds": seconds,
    }
    st.session_state.history = answer.messages
    st.session_state.chat += [user_turn, assistant_turn]
    # Draw the page again from the chat, without the example buttons
    st.rerun()


st.html(PAGE_STYLE)
st.title("Leeze AI")
st.write(
    "An AI chat about Münster's bicycle traffic (*Leeze* is Münster dialect for bicycle): the "
    "forecast and how accurate it is, the counts at all 23 sites, the weather, and problems in "
    "the data. A language model (Claude) chooses which of ten data tools to call and words the "
    "answer; the numbers come from the tools. Type a question or record it with the microphone "
    "in the input field: a spoken question gets a short spoken answer."
)
st.caption(
    "Your questions are sent to Anthropic's API, recordings and spoken answers to OpenAI's API. Answers can still be wrong: check the tool "
    f"calls below each answer. Up to {MAX_QUESTIONS_PER_CONVERSATION} questions per conversation "
    f"and {MAX_QUESTIONS_PER_DAY} questions per day for all users together."
)

key = api_key()
if key is None:
    st.info("The assistant is not set up here: the API key is missing.")
    st.stop()

today = today_in_muenster()
data = load_data(today)
client = get_client(key)
limit = daily_limit()
# Without an OpenAI key (or the openai package) the assistant only takes
# typed questions
voice_key = api_key("OPENAI_API_KEY") if openai is not None else None
voice_client = get_voice_client(voice_key) if voice_key else None

# history: the conversation as sent to the API (with tool calls);
# chat: what the page shows
st.session_state.setdefault("history", [])
st.session_state.setdefault("chat", [])

for turn in st.session_state.chat:
    show(turn)

if limit.remaining(today) <= 0:
    st.info(DAY_LIMIT_TEXT)
    st.stop()

submitted = st.chat_input(
    "Type or record your own question here …",
    max_chars=MAX_QUESTION_CHARS,
    accept_audio=voice_client is not None,
)
# A string without audio input, else text and audio (one of them empty)
question = submitted if isinstance(submitted, str) else submitted and submitted.text
recording = None if isinstance(submitted, str) or submitted is None else submitted.audio
stt_seconds = None
if not st.session_state.chat:
    st.write("Or try one of these:")
    for example in EXAMPLES:
        if st.button(example):
            question = example

questions_asked = len(st.session_state.chat) // 2
if recording is not None:
    if question:
        st.info("Only the recording was used as the question, not the typed text.")
    # Before the transcription, which is paid for too
    problem = conversation_problem(questions_asked)
    if problem:
        st.warning(problem)
        question = None
    else:
        transcribed = transcribe_recording(voice_client, recording.getvalue())
        question, stt_seconds = transcribed or (None, None)

if question:
    # The input field limits the length too, but only in the browser
    problem = question_problem(question, questions_asked)
    if problem:
        st.warning(problem)
    elif not limit.take(today):
        st.info(DAY_LIMIT_TEXT)
    else:
        answer_question(client, data, question, voice_client if stt_seconds is not None else None, stt_seconds)

if st.session_state.chat and st.button("New conversation"):
    st.session_state.history = []
    st.session_state.chat = []
    st.rerun()
