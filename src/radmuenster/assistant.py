"""The tool-use loop of the AI assistant.

ask() sends the conversation to the model. As long as the model asks for
tools, it runs them (assistant_tools.run_tool) and sends the results back;
when the model answers in text, the loop ends.

The client is passed in: in the app the Anthropic client, in the tests a
fake one with the same messages.create(). This module does not import the
anthropic package; it only relies on the shape of the Messages API (content
blocks with type text or tool_use, stop_reason, usage).
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time

from radmuenster.assistant_tools import FORECAST_SITE, TOOL_SPECS, AssistantData, ToolError, run_tool
from radmuenster.forecast import TIMEZONE

# The smallest current Claude model: the tools do the work, the model picks
# them and words the answer (decided 2026-10-14; compare with Sonnet later)
MODEL = "claude-haiku-5-5"

# Tool rounds per question. Then the model gets one last call without tools
# and has to answer with what it has.
MAX_ROUNDS = 6

MAX_TOKENS = 4000

# Limits against cost and misuse of the public app. The whole conversation is
# sent with every question, so long conversations cost more per question.
# The hard limit is the spending limit in the Anthropic Console.
MAX_QUESTION_CHARS = 500
MAX_QUESTIONS_PER_CONVERSATION = 10
MAX_QUESTIONS_PER_DAY = 100

# When the published data and the forecast are usually updated: the daily job
# is scheduled for 18:00 UTC (.github/workflows/daily-forecast.yml), starts
# about 10 minutes late and runs a few minutes
DAILY_UPDATE_UTC = time(18, 15)

SYSTEM_PROMPT = f"""\
You answer questions about bicycle traffic in Münster, Germany, from the open \
counting data of the City of Münster (23 automatic counting sites) and daily \
forecasts for all of them: the {FORECAST_SITE} with its own model, the other sites \
with one model for all.

Always answer in the language of the user's latest message. Site names and tool \
results are partly German, and earlier answers may be in another language; \
neither changes the language of your answer.

- Take every number from a tool result. If no tool can answer the question, say so; \
do not estimate from general knowledge.
- Explain with what the tools return. Do not claim effects (of the weather, a \
holiday, ...) that no tool result states. The findings of get_insights state typical \
effects (e.g. how much rain lowers a type of route): you may use them to explain a \
forecast, as the typical effect, not as the model's own breakdown of that forecast.
- If the latest forecast or the latest counts are for an earlier day than asked, \
say which day they are for and when the next update comes, and still answer with them.
- Mention flagged days and data problems when they affect the answer.
- Answer briefly, with numbers rounded sensibly.
- Questions unrelated to cycling in Münster: say briefly that you only answer those."""

# Sent before a spoken question (radmuenster.voice), in the same message: the
# answer is read aloud, and the question may have recognition errors. Not in
# the system prompt: the thinking blocks in the history are bound to the
# system prompt, so it must stay the same for typed and spoken questions.
SPOKEN_NOTE = """\
The following question was spoken and transcribed automatically, so it may contain \
recognition errors, most likely in site names: if a site name is unknown, use \
the closest name from list_sites. The answer is read aloud: answer in at most \
three short sentences of plain text, without tables, lists, headings, markdown \
or emoji; give at most three numbers, rounded (about 4,300 trips), and dates \
in words (Tuesday, 21 October). Answer in the language of the question, not \
of this note."""

LIMIT_NOTE = "The tool limit for this question is reached. Answer with the results you have."
TRUNCATED_NOTE = "(The answer was cut off because it got too long.)"
REFUSAL_TEXT = "I can't answer this question."


@dataclass
class ToolCall:
    """One tool call of the model, for display and logging."""

    name: str
    arguments: dict
    error: str | None = None


@dataclass
class Answer:
    text: str
    messages: list
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str | None = None


def tool_definitions() -> list[dict]:
    """TOOL_SPECS in the format of the Messages API. strict makes the API
    check the arguments against the schema."""
    return [
        {
            "name": spec.name,
            "description": spec.description,
            "input_schema": spec.input_schema,
            "strict": True,
        }
        for spec in TOOL_SPECS
    ]


def system_prompt(today: date) -> str:
    # The model needs today's date for "yesterday" or "last week". It changes
    # daily, so it goes last: a prompt cache (not used yet) matches the
    # request from the start up to the first change
    return f"{SYSTEM_PROMPT}\n\n{update_note(today)}\nToday is {today:%A, %Y-%m-%d}."


def update_note(today: date) -> str:
    """When the data is updated, in Münster time: the model should not
    convert time zones, and the hour changes with daylight saving time."""
    update = datetime.combine(today, DAILY_UPDATE_UTC, tzinfo=UTC).astimezone(TIMEZONE)
    return (
        f"The counts, the weather and the forecast are updated once a day, in the evening at "
        f"about {update:%H:%M} Münster time. Until then the latest counts are usually for the "
        f"day before yesterday and the latest forecast is for today; after the update, the "
        f"forecast is for tomorrow."
    )


def _execute(data: AssistantData, block) -> tuple[dict, ToolCall]:
    """Run one tool_use block; returns the tool_result block and the log entry."""
    call = ToolCall(block.name, dict(block.input))
    try:
        content = json.dumps(run_tool(data, block.name, block.input), ensure_ascii=False, default=str)
    except ToolError as error:
        # A wrong call: the model can read the message and correct it
        call.error = content = str(error)
    except Exception as error:  # noqa: BLE001 - a bug in a tool must not end the chat
        call.error = f"{type(error).__name__}: {error}"
        content = f"The tool {block.name} failed with an internal error."
    result = {"type": "tool_result", "tool_use_id": block.id, "content": content}
    if call.error:
        result["is_error"] = True
    return result, call


def _text(response) -> str:
    return "\n\n".join(block.text for block in response.content if block.type == "text").strip()


def ask(
    client,
    data: AssistantData,
    question: str,
    model: str,
    history: list | None = None,
    max_rounds: int = MAX_ROUNDS,
    spoken: bool = False,
) -> Answer:
    """Answer question, continuing the conversation history.

    Returns the answer with the new history (history is not changed). The
    history keeps the model's content blocks unchanged, including tool calls
    and thinking, and is only ever appended to. spoken asks for a short
    answer to be read aloud (SPOKEN_NOTE).
    """
    messages = list(history or [])
    if spoken:
        content = [{"type": "text", "text": SPOKEN_NOTE}, {"type": "text", "text": question}]
    else:
        content = question
    messages.append({"role": "user", "content": content})
    answer = Answer("", messages)
    request = {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "system": system_prompt(data.today),
        "tools": tool_definitions(),
    }

    for round_ in range(max_rounds + 1):
        last = round_ == max_rounds
        # In the last round the model must answer without tools
        extra = {"tool_choice": {"type": "none"}} if last else {}
        response = client.messages.create(**request, **extra, messages=messages)
        answer.input_tokens += response.usage.input_tokens
        answer.output_tokens += response.usage.output_tokens
        answer.stop_reason = response.stop_reason
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use" or last:
            break

        # All results of one response go back in one message, in order
        results = []
        for block in response.content:
            if block.type == "tool_use":
                result, call = _execute(data, block)
                results.append(result)
                answer.tool_calls.append(call)
        if round_ == max_rounds - 1:
            results.append({"type": "text", "text": LIMIT_NOTE})
        messages.append({"role": "user", "content": results})

    if response.stop_reason == "refusal":
        answer.text = REFUSAL_TEXT
    else:
        answer.text = _text(response)
        if response.stop_reason == "max_tokens":
            answer.text = f"{answer.text}\n\n{TRUNCATED_NOTE}".strip()
    return answer


def conversation_problem(questions_asked: int) -> str | None:
    """Why the conversation takes no more questions, or None if it does.
    questions_asked counts the questions of this conversation so far. Checked
    before a recording is transcribed too, which costs money as well."""
    if questions_asked >= MAX_QUESTIONS_PER_CONVERSATION:
        return (
            f"This conversation has reached {MAX_QUESTIONS_PER_CONVERSATION} questions. "
            "Please start a new conversation."
        )
    return None


def question_problem(question: str, questions_asked: int) -> str | None:
    """Why a question is not sent to the model, or None if it may be.
    questions_asked counts the questions of this conversation so far."""
    if len(question) > MAX_QUESTION_CHARS:
        return f"Please ask in at most {MAX_QUESTION_CHARS} characters."
    return conversation_problem(questions_asked)


class DailyLimit:
    """Questions per day across all users of the app.

    The count is kept in memory: a restart of the app resets it. That is
    good enough against misuse; the spending limit in the Anthropic Console
    is the hard limit. Streamlit runs every session in its own thread, so
    the count is changed under a lock.
    """

    def __init__(self, per_day: int = MAX_QUESTIONS_PER_DAY):
        self.per_day = per_day
        self._day: date | None = None
        self._count = 0
        self._lock = threading.Lock()

    def _start(self, today: date) -> None:
        # A new day starts with a new count
        if today != self._day:
            self._day, self._count = today, 0

    def remaining(self, today: date) -> int:
        with self._lock:
            self._start(today)
            return self.per_day - self._count

    def take(self, today: date) -> bool:
        """Counts one question for today; False if none is left."""
        with self._lock:
            self._start(today)
            if self._count >= self.per_day:
                return False
            self._count += 1
            return True
