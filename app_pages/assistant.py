"""Assistant page: questions about the data, answered by a language model
that calls the tools of radmuenster.assistant_tools."""

from __future__ import annotations

import json
import logging
from datetime import date

import anthropic
import streamlit as st

from radmuenster.assistant import MAX_QUESTION_CHARS, MODEL, DailyLimit, ask, question_problem
from radmuenster.assistant_tools import AssistantData
from radmuenster.forecast import today_in_muenster
from radmuenster.published import read_forecast_log, read_status
from radmuenster.static_page import DAILY_DATA, SITES_DATA, WEATHER_DATA, read_published

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
# that works on both themes makes it look like a field to type in
INPUT_STYLE = """<style>
[data-testid="stChatInput"] > div { border: 2px solid rgba(128, 128, 128, 0.7); }
</style>"""


@st.cache_data(ttl=CACHE_SECONDS)
def load_data(today: date) -> AssistantData:
    """Everything the tools read, as published by the daily job. today is part
    of the cache key, so that a new day loads new data."""
    return AssistantData(
        sites_daily=read_published(SITES_DATA),
        daily=read_published(DAILY_DATA),
        weather=read_published(WEATHER_DATA),
        log=read_forecast_log(),
        today=today,
        warnings=read_status()["warnings"],
    )


@st.cache_resource
def get_client(api_key: str) -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=api_key, timeout=60)


@st.cache_resource
def daily_limit() -> DailyLimit:
    # cache_resource: one object shared by all sessions, so all users count
    # towards the same limit
    return DailyLimit()


def api_key() -> str | None:
    """The key from .streamlit/secrets.toml (locally) or the app's secrets
    (Streamlit Cloud); None if there is none."""
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except (FileNotFoundError, KeyError):
        return None


def show(turn: dict) -> None:
    """One chat message, with the tool calls behind an answer."""
    with st.chat_message(turn["role"]):
        st.markdown(turn["text"])
        if turn.get("tool_calls") is not None:
            with st.expander(f"How this answer was found ({len(turn['tool_calls'])} tool calls)"):
                for call in turn["tool_calls"]:
                    arguments = json.dumps(call.arguments, ensure_ascii=False)
                    st.code(f"{call.name}({arguments})", language=None)
                    if call.error:
                        st.caption(f"Error: {call.error}")
                st.caption(f"{turn['input_tokens']:,} input and {turn['output_tokens']:,} output tokens")


def answer_question(client: anthropic.Anthropic, data: AssistantData, question: str) -> None:
    """Asks the model, adds both turns to the conversation and draws the page again."""
    user_turn = {"role": "user", "text": question}
    show(user_turn)
    with st.spinner("Looking at the data …"):
        try:
            answer = ask(client, data, question, MODEL, history=st.session_state.history)
        except anthropic.APIError as error:
            # The history stays as it was, so the next question works
            logger.exception("Assistant request failed")
            st.error(f"The assistant is not available right now ({type(error).__name__}). Please try again later.")
            st.stop()
    for call in answer.tool_calls:
        if call.error:
            logger.warning("Tool %s(%s) failed: %s", call.name, call.arguments, call.error)
    assistant_turn = {
        "role": "assistant",
        "text": answer.text,
        "tool_calls": answer.tool_calls,
        "input_tokens": answer.input_tokens,
        "output_tokens": answer.output_tokens,
    }
    st.session_state.history = answer.messages
    st.session_state.chat += [user_turn, assistant_turn]
    # Draw the page again from the chat, without the example buttons
    st.rerun()


st.html(INPUT_STYLE)
st.title("Leeze AI")
st.write(
    "An AI chat about Münster's bicycle traffic (*Leeze* is Münster dialect for bicycle): the "
    "forecast and how accurate it is, the counts at all 23 sites, the weather, and problems in "
    "the data. A language model (Claude) chooses which of nine data tools to call and words the "
    "answer; the numbers come from the tools."
)
st.caption(
    "Your questions are sent to Anthropic's API. Answers can still be wrong: check the tool "
    "calls below each answer."
)

key = api_key()
if key is None:
    st.info("The assistant is not set up here: the API key is missing.")
    st.stop()

today = today_in_muenster()
data = load_data(today)
client = get_client(key)
limit = daily_limit()

# history: the conversation as sent to the API (with tool calls);
# chat: what the page shows
st.session_state.setdefault("history", [])
st.session_state.setdefault("chat", [])

for turn in st.session_state.chat:
    show(turn)

if limit.remaining(today) <= 0:
    st.info(DAY_LIMIT_TEXT)
    st.stop()

question = st.chat_input("Type your own question here …", max_chars=MAX_QUESTION_CHARS)
if not st.session_state.chat:
    st.write("Or try one of these:")
    for example in EXAMPLES:
        if st.button(example):
            question = example

if question:
    # The input field limits the length too, but only in the browser
    problem = question_problem(question, questions_asked=len(st.session_state.chat) // 2)
    if problem:
        st.warning(problem)
    elif not limit.take(today):
        st.info(DAY_LIMIT_TEXT)
    else:
        answer_question(client, data, question)

if st.session_state.chat and st.button("New conversation"):
    st.session_state.history = []
    st.session_state.chat = []
    st.rerun()
