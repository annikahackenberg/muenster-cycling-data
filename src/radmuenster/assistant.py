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
from dataclasses import dataclass, field
from datetime import date

from radmuenster.assistant_tools import FORECAST_SITE, TOOL_SPECS, AssistantData, ToolError, run_tool

# Tool rounds per question. Then the model gets one last call without tools
# and has to answer with what it has.
MAX_ROUNDS = 6

MAX_TOKENS = 4000

SYSTEM_PROMPT = f"""\
You answer questions about bicycle traffic in Münster, Germany, from the open \
counting data of the City of Münster (23 automatic counting sites) and a daily \
forecast for the {FORECAST_SITE}.

- Take every number from a tool result. If no tool can answer the question, say so; \
do not estimate from general knowledge.
- Mention flagged days and data problems when they affect the answer.
- Answer in the language of the question, briefly, with numbers rounded sensibly.
- Questions unrelated to cycling in Münster: say briefly that you only answer those."""

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
    return f"{SYSTEM_PROMPT}\n\nToday is {today:%A, %Y-%m-%d}."


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
) -> Answer:
    """Answer question, continuing the conversation history.

    Returns the answer with the new history (history is not changed). The
    history keeps the model's content blocks unchanged, including tool calls
    and thinking, and is only ever appended to.
    """
    messages = list(history or [])
    messages.append({"role": "user", "content": question})
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
