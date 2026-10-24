import copy
import json
import threading
from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest

from radmuenster.assistant import (
    LIMIT_NOTE,
    MAX_QUESTION_CHARS,
    MAX_QUESTIONS_PER_CONVERSATION,
    REFUSAL_TEXT,
    SPOKEN_NOTE,
    TRUNCATED_NOTE,
    DailyLimit,
    ask,
    question_problem,
    system_prompt,
    tool_definitions,
)
from radmuenster.assistant_tools import TOOL_SPECS, AssistantData

MODEL = "test-model"


def text(value):
    return SimpleNamespace(type="text", text=value)


def tool_use(id_, name, arguments):
    return SimpleNamespace(type="tool_use", id=id_, name=name, input=arguments)


def response(*content, stop_reason=None, input_tokens=100, output_tokens=10):
    """A response as the Messages API returns it; stop_reason follows from
    the content unless given."""
    if stop_reason is None:
        stop_reason = "tool_use" if any(b.type == "tool_use" for b in content) else "end_turn"
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    return SimpleNamespace(content=list(content), stop_reason=stop_reason, usage=usage)


class FakeClient:
    """Plays the model: returns the given responses in order and keeps a copy
    of every request."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []
        self.messages = self  # client.messages.create

    def create(self, **request):
        self.requests.append(copy.deepcopy(request))
        return self.responses.pop(0)


@pytest.fixture
def data():
    sites_daily = pd.DataFrame(
        {"site_name": ["Neutor", "Neutor"], "date": pd.to_datetime(["2026-09-01", "2026-10-12"])}
    )
    empty = pd.DataFrame()
    return AssistantData(sites_daily, empty, empty, empty, date(2026, 10, 13))


def test_answer_without_tools(data):
    client = FakeClient(response(text("Hello!")))
    answer = ask(client, data, "Hi", MODEL)
    assert answer.text == "Hello!"
    assert answer.tool_calls == []
    [request] = client.requests
    assert request["model"] == MODEL
    assert request["messages"] == [{"role": "user", "content": "Hi"}]
    assert "Tuesday, 2026-10-13" in request["system"]
    assert "tool_choice" not in request


def test_runs_the_tool_and_sends_the_result(data):
    client = FakeClient(
        response(text("Let me look."), tool_use("t1", "list_sites", {})),
        response(text("There is one site: Neutor.")),
    )
    answer = ask(client, data, "Which sites are there?", MODEL)
    assert answer.text == "There is one site: Neutor."
    assert [call.name for call in answer.tool_calls] == ["list_sites"]

    [result] = client.requests[1]["messages"][-1]["content"]
    assert result["tool_use_id"] == "t1"
    assert "is_error" not in result
    sites = json.loads(result["content"])["sites"]
    assert sites == [{"name": "Neutor", "first_day": "2026-09-01", "last_day": "2026-10-12"}]


def test_history_keeps_all_blocks_in_order(data):
    first = response(tool_use("t1", "list_sites", {}))
    final = response(text("Done."))
    answer = ask(FakeClient(first, final), data, "Q", MODEL)
    roles = [message["role"] for message in answer.messages]
    assert roles == ["user", "assistant", "user", "assistant"]
    # The model's blocks are kept as they came, not rebuilt
    assert answer.messages[1]["content"] == first.content
    assert answer.messages[3]["content"] == final.content


def test_continues_the_conversation(data):
    history = ask(FakeClient(response(text("A1"))), data, "Q1", MODEL).messages
    client = FakeClient(response(text("A2")))
    answer = ask(client, data, "Q2", MODEL, history=history)
    assert [m["role"] for m in client.requests[0]["messages"]] == ["user", "assistant", "user"]
    assert answer.text == "A2"
    assert len(history) == 2  # the old history is not changed


def test_wrong_call_goes_back_as_error(data):
    client = FakeClient(
        response(tool_use("t1", "get_flagged_periods", {"site": "Nutor", "start": "2026-09-01", "end": "2026-09-30"})),
        response(text("Did you mean Neutor?")),
    )
    answer = ask(client, data, "Problems at Nutor?", MODEL)
    [result] = client.requests[1]["messages"][-1]["content"]
    assert result["is_error"] is True
    assert "Did you mean: Neutor?" in result["content"]
    assert answer.tool_calls[0].error == result["content"]


def test_unknown_tool_goes_back_as_error(data):
    client = FakeClient(response(tool_use("t1", "delete_everything", {})), response(text("Sorry.")))
    ask(client, data, "Q", MODEL)
    [result] = client.requests[1]["messages"][-1]["content"]
    assert result["is_error"] is True
    assert "Unknown tool" in result["content"]


def test_bug_in_a_tool_does_not_end_the_chat(data):
    broken = AssistantData(pd.DataFrame(), data.daily, data.weather, data.log, data.today)
    client = FakeClient(response(tool_use("t1", "list_sites", {})), response(text("Sorry.")))
    answer = ask(client, broken, "Q", MODEL)
    [result] = client.requests[1]["messages"][-1]["content"]
    assert result["is_error"] is True
    # The model only learns that the tool failed; the details are logged
    assert result["content"] == "The tool list_sites failed with an internal error."
    assert answer.tool_calls[0].error.startswith("KeyError")
    assert answer.text == "Sorry."


def test_parallel_calls_go_back_in_one_message(data):
    client = FakeClient(
        response(tool_use("t1", "list_sites", {}), tool_use("t2", "nope", {})),
        response(text("Done.")),
    )
    ask(client, data, "Q", MODEL)
    last = client.requests[1]["messages"][-1]
    assert last["role"] == "user"
    assert [r["tool_use_id"] for r in last["content"]] == ["t1", "t2"]


def test_round_limit_forces_an_answer(data):
    calls = [response(tool_use(f"t{i}", "list_sites", {})) for i in range(2)]
    client = FakeClient(*calls, response(text("Best I can say.")))
    answer = ask(client, data, "Q", MODEL, max_rounds=2)
    assert answer.text == "Best I can say."
    assert len(client.requests) == 3
    assert "tool_choice" not in client.requests[1]
    assert client.requests[2]["tool_choice"] == {"type": "none"}
    # The results of the last tool round are still sent, with the note
    assert client.requests[2]["messages"][-1]["content"][-1] == {"type": "text", "text": LIMIT_NOTE}


def test_tokens_add_up(data):
    client = FakeClient(
        response(tool_use("t1", "list_sites", {}), input_tokens=1000, output_tokens=50),
        response(text("Done."), input_tokens=1200, output_tokens=30),
    )
    answer = ask(client, data, "Q", MODEL)
    assert (answer.input_tokens, answer.output_tokens) == (2200, 80)


def test_cut_off_answer_gets_a_note(data):
    client = FakeClient(response(text("A long"), stop_reason="max_tokens"))
    answer = ask(client, data, "Q", MODEL)
    assert answer.text == f"A long\n\n{TRUNCATED_NOTE}"
    assert answer.stop_reason == "max_tokens"


def test_refusal(data):
    client = FakeClient(response(stop_reason="refusal"))
    assert ask(client, data, "Q", MODEL).text == REFUSAL_TEXT


def test_tool_definitions():
    definitions = tool_definitions()
    assert [d["name"] for d in definitions] == [spec.name for spec in TOOL_SPECS]
    for definition in definitions:
        assert definition["strict"] is True
        assert set(definition) == {"name", "description", "input_schema", "strict"}


@pytest.mark.parametrize(
    ("today", "local_time"),
    [
        (date(2026, 10, 16), "20:15"),  # summer time
        (date(2026, 10, 26), "19:15"),  # winter time since 25 October
    ],
)
def test_system_prompt_gives_the_update_time_in_muenster(today, local_time):
    prompt = system_prompt(today)
    assert f"at about {local_time} Münster time" in prompt
    assert prompt.endswith(f"Today is {today:%A, %Y-%m-%d}.")


def test_question_problem_checks_length_and_conversation():
    assert question_problem("How many trips?", questions_asked=0) is None
    assert "500 characters" in question_problem("x" * (MAX_QUESTION_CHARS + 1), questions_asked=0)
    assert question_problem("x" * MAX_QUESTION_CHARS, questions_asked=MAX_QUESTIONS_PER_CONVERSATION - 1) is None
    assert "new conversation" in question_problem("Hi", questions_asked=MAX_QUESTIONS_PER_CONVERSATION)


def test_daily_limit_counts_per_day():
    limit = DailyLimit(per_day=2)
    monday, tuesday = date(2026, 10, 19), date(2026, 10, 20)
    assert limit.remaining(monday) == 2
    assert limit.take(monday) and limit.take(monday)
    assert not limit.take(monday)
    assert limit.remaining(monday) == 0
    # A new day starts with a new count
    assert limit.remaining(tuesday) == 2
    assert limit.take(tuesday)


def test_daily_limit_holds_across_threads():
    # Streamlit runs every session in its own thread
    limit = DailyLimit(per_day=50)
    today = date(2026, 10, 19)
    results = []
    threads = [threading.Thread(target=lambda: results.append(limit.take(today))) for _ in range(80)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert results.count(True) == 50


def test_spoken_question_asks_for_a_short_answer(data):
    client = FakeClient(response(text("Morgen etwa 4.300 Fahrten.")), response(text("About 4,300 trips.")))
    ask(client, data, "Wie viele Fahrten morgen?", MODEL, spoken=True)
    ask(client, data, "How many trips tomorrow?", MODEL)
    spoken, typed = client.requests
    assert SPOKEN_NOTE in spoken["system"]
    assert SPOKEN_NOTE not in typed["system"]
    # The date stays last, after the note
    assert spoken["system"].endswith("Today is Tuesday, 2026-10-13.")
