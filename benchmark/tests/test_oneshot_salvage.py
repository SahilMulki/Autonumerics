"""The C0 invocation must not throw away a solver the model actually finished.

Measured 2026-09-27: 34 of 44 C0@Opus units on the no-closed-form suite came back
`subtype: error_max_turns` with an EMPTY `result` field -- Opus writes the solver,
reflexively reaches for Bash to test it, the NoTool allow-list denies that, and the
turn cap ends the run before a final answer. The completion (13k output tokens of
working solver) was discarded and the unit scored CRASHED/no_solve_fn. These tests
pin the recovery: read the code out of the streamed assistant turns, and do not turn
that recovery into an extra retry, an extra turn, or a silent pass.
"""
import contextlib
import json

import pytest

from benchmark import oneshot as O


def _ev_assistant(*blocks):
    return json.dumps({"type": "assistant", "message": {"content": list(blocks)}})


def _text(t):
    return {"type": "text", "text": t}


def _ev_write(content, path="/tmp/solver.py"):
    return _ev_assistant({"type": "tool_use", "name": "Write",
                          "input": {"file_path": path, "content": content}})


def _result(**kw):
    ev = {"type": "result", "subtype": "success", "result": "", "num_turns": 1,
          "usage": {"input_tokens": 10, "output_tokens": 20}, "total_cost_usd": 0.1}
    ev.update(kw)
    return json.dumps(ev)


SOLVER = "```python\ndef solve_pde(N):\n    return {'u': N}\n```"


# --- the parser understands both output formats -----------------------------


def test_stream_json_yields_the_result_event_and_the_assistant_text():
    stdout = "\n".join([
        json.dumps({"type": "system", "subtype": "init"}),
        _ev_assistant({"type": "thinking", "thinking": "def solve_pde(N): WRONG"}),
        _ev_assistant(_text(SOLVER)),
        _result(num_turns=3, subtype="error_max_turns"),
    ])
    parsed = O._parse_output(stdout)
    assert parsed["num_turns"] == 3
    assert parsed["stop_subtype"] == "error_max_turns"
    assert parsed["result_text"] == ""
    assert SOLVER in parsed["assistant_text"]
    # reasoning about a solver is not the solver
    assert "WRONG" not in parsed["assistant_text"]


def test_the_single_object_json_format_still_parses():
    """--output-format json is what every pre-2026-09-27 record was made with."""
    parsed = O._parse_output(json.dumps({
        "result": "hello", "num_turns": 1, "total_cost_usd": 0.5,
        "usage": {"input_tokens": 3, "cache_read_input_tokens": 7, "output_tokens": 9},
    }))
    assert parsed["result_text"] == "hello"
    assert parsed["input_tokens"] == 10      # base + cache read
    assert parsed["assistant_text"] == ""


def test_a_truncated_stream_still_yields_the_turns_that_completed():
    stdout = _ev_assistant(_text(SOLVER)) + "\n{ this line is not json"
    parsed = O._parse_output(stdout)
    assert SOLVER in parsed["assistant_text"]


def test_permission_denials_are_recorded_by_tool_name():
    parsed = O._parse_output(_result(permission_denials=[
        {"tool_name": "Bash"}, {"tool_name": "Bash"}, {"tool_name": "Read"}]))
    assert parsed["denied_tools"] == ["Bash", "Read"]


# --- salvage ----------------------------------------------------------------


@pytest.fixture
def fake_cli(monkeypatch):
    """Stub subprocess.run inside invoke(); returns the list of invocations made."""
    calls = []

    def _install(stdout, returncode=1):
        class _Proc:
            def __init__(self):
                self.stdout, self.stderr, self.returncode = stdout, "", returncode

        def _run(cmd, **kw):
            calls.append(cmd)
            return _Proc()

        import subprocess
        monkeypatch.setattr(subprocess, "run", _run)
        return calls

    return _install


EMPTY_MAX_TURNS = "\n".join([
    _ev_assistant(_text("Here is the solver.\n\n" + SOLVER)),
    _result(subtype="error_max_turns", num_turns=3,
            permission_denials=[{"tool_name": "Bash"}]),
])


def test_an_empty_result_is_salvaged_from_the_assistant_turns(fake_cli):
    fake_cli(EMPTY_MAX_TURNS)
    run = O.invoke("p", "opus", "c0", ".", max_turns=2, timeout=60, salvage=True)
    assert run["result_text"] == ""            # the CLI really did return nothing
    assert run["salvage_used"] is True
    assert run["code_source"] == "assistant_turns"
    assert "def solve_pde" in O.extract_code(run["code_text"])
    assert run["stop_subtype"] == "error_max_turns"
    assert run["denied_tools"] == ["Bash"]


def test_salvage_is_off_by_default_so_c1_is_unchanged(fake_cli):
    fake_cli(EMPTY_MAX_TURNS)
    run = O.invoke("p", "opus", "c1", ".", max_turns=30, timeout=60)
    assert run["salvage_used"] is False
    assert run["code_source"] == "result"
    assert run["code_text"] == ""


def test_a_salvaged_run_is_not_retried(fake_cli):
    """The 2026-09-27 run spent 2 extra invocations per unit re-earning a completion
    it already had -- $16.49 of $22.85. A recovered run must end the retry loop."""
    calls = fake_cli(EMPTY_MAX_TURNS)
    run = O.invoke_with_retry("p", "opus", "c0", ".", max_turns=2, timeout=60,
                              retries=2, salvage=True)
    assert run["salvage_used"] is True
    assert run["retries_used"] == 0
    assert len(calls) == 1


def test_an_empty_error_with_no_code_anywhere_still_retries(fake_cli):
    """Salvage must not mask a genuinely lost completion."""
    calls = fake_cli("\n".join([_ev_assistant(_text("I cannot do this.")),
                                _result(subtype="error_max_turns")]))
    run = O.invoke_with_retry("p", "opus", "c0", ".", max_turns=2, timeout=60,
                              retries=2, salvage=True)
    assert run["salvage_used"] is False
    assert run["retries_used"] == 2
    assert len(calls) == 3


def test_a_usage_limit_is_never_retried_or_salvaged(fake_cli):
    limit = _ev_assistant(_text("Claude AI usage limit reached")) + "\n" + \
        _result(subtype="error_max_turns")
    calls = fake_cli(limit)
    run = O.invoke_with_retry("p", "opus", "c0", ".", max_turns=2, timeout=60,
                              retries=2, salvage=True)
    assert run["limit_hit"] is True
    assert len(calls) == 1


def test_a_normal_success_does_not_report_salvage(fake_cli):
    fake_cli(_ev_assistant(_text(SOLVER)) + "\n" + _result(result=SOLVER), returncode=0)
    run = O.invoke("p", "opus", "c0", ".", max_turns=2, timeout=60, salvage=True)
    assert run["agent_status"] == "ok"
    assert run["salvage_used"] is False
    assert run["code_source"] == "result"


# --- the invocation itself stays a naked one-shot ---------------------------


def test_c0_forbids_tools_at_every_turn_cap():
    """A larger cap is still tool-less: NoTool means no tool ever executes, it only
    lets the model answer after its denied attempts."""
    for cap in (2, 4, 8):
        cmd = O.claude_command("opus", "c0", max_turns=cap)
        assert cmd[cmd.index("--allowed-tools") + 1] == "NoTool"
        assert cmd[cmd.index("--max-turns") + 1] == str(cap)
        assert "--dangerously-skip-permissions" not in cmd
        assert cmd[cmd.index("--output-format") + 1] == "stream-json"
        assert "--verbose" in cmd


def test_salvage_does_not_raise_the_turn_cap(fake_cli):
    """Salvage recovers work already done; it must not buy turns."""
    fake_cli(EMPTY_MAX_TURNS)
    O.invoke("p", "opus", "c0", ".", max_turns=2, timeout=60, salvage=True)
    cmd = O.claude_command("opus", "c0", max_turns=2)
    assert cmd[cmd.index("--max-turns") + 1] == "2"


def test_c1_still_gets_its_tools_and_the_pipeline_settings():
    cmd = O.claude_command("opus", "c1", max_turns=30)
    assert cmd[cmd.index("--max-turns") + 1] == "30"
    for tool in ("Bash", "Read", "Write", "Edit"):
        assert tool in cmd
    assert "--settings" in cmd


# --- a solver the model tried to Write is still the model's solver -----------

RAW_SOLVER = "import numpy as np\n\ndef solve_sde(num_paths, dt, T, seed=42):\n    return {}\n"
STUDY_SCRIPT = ("import numpy as np\n\ndef drift_flow(x, tau):\n    return x\n\n"
                "def run_coupled(npaths, seed):\n    return 0\n")


def test_a_denied_write_of_a_solver_is_recovered(fake_cli):
    """Measured 2026-09-27: on the real failing units Opus emits NO text blocks at
    all -- only thinking and tool_use. Where the tool_use is a Write of the solver,
    that file body is the answer."""
    fake_cli("\n".join([
        _ev_assistant({"type": "thinking", "thinking": "let me write it"}),
        _ev_write(RAW_SOLVER),
        _result(subtype="error_max_turns", num_turns=3),
    ]))
    run = O.invoke("p", "opus", "c0", ".", max_turns=2, timeout=60, salvage=True)
    assert run["salvage_used"] is True
    assert run["code_source"] == "denied_write"
    assert "def solve_sde" in O.extract_code(run["code_text"])


def test_a_scratch_experiment_script_is_not_mistaken_for_an_answer(fake_cli):
    """The actual conv.py Opus tried to write on sde_double_well_langevin rep1 was a
    weak-convergence study with no contract function. Recovering that as the solver
    would manufacture a submission the model never made."""
    fake_cli("\n".join([
        _ev_write(STUDY_SCRIPT, path="/tmp/conv.py"),
        _result(subtype="error_max_turns", num_turns=3),
    ]))
    run = O.invoke("p", "opus", "c0", ".", max_turns=2, timeout=60, salvage=True)
    assert run["salvage_used"] is False
    assert run["code_source"] == "result"


def test_the_last_write_wins_over_an_earlier_draft(fake_cli):
    first = RAW_SOLVER.replace("return {}", "return {'first': 1}")
    last = RAW_SOLVER.replace("return {}", "return {'last': 2}")
    fake_cli("\n".join([_ev_write(first), _ev_write(last),
                         _result(subtype="error_max_turns")]))
    run = O.invoke("p", "opus", "c0", ".", max_turns=2, timeout=60, salvage=True)
    assert "'last': 2" in run["code_text"]


def test_a_text_block_is_preferred_over_a_write_payload(fake_cli):
    fake_cli("\n".join([_ev_assistant(_text(SOLVER)), _ev_write(RAW_SOLVER),
                         _result(subtype="error_max_turns")]))
    run = O.invoke("p", "opus", "c0", ".", max_turns=2, timeout=60, salvage=True)
    assert run["code_source"] == "assistant_turns"


def test_thinking_is_never_a_source_of_code(fake_cli):
    """Draft code inside reasoning is not a submission."""
    fake_cli("\n".join([
        _ev_assistant({"type": "thinking", "thinking": "```python\ndef solve_sde(): pass\n```"}),
        _result(subtype="error_max_turns"),
    ]))
    run = O.invoke("p", "opus", "c0", ".", max_turns=2, timeout=60, salvage=True)
    assert run["salvage_used"] is False


# --- extract_code -----------------------------------------------------------


def test_unfenced_code_is_accepted_only_when_it_defines_the_contract():
    assert O.extract_code(RAW_SOLVER).startswith("import numpy")
    assert O.extract_code("I am not able to solve this problem.") is None
    assert O.extract_code(STUDY_SCRIPT) is None


def test_a_fenced_contract_block_still_wins_over_a_longer_one():
    text = ("```python\n" + "# padding\n" * 50 + "```\n"
            "```python\ndef solve_pde(N):\n    return {}\n```")
    assert "def solve_pde" in O.extract_code(text)


# --- a stream the agent_timeout cut off -------------------------------------


def test_a_truncated_stream_is_not_returned_as_the_answer():
    """Measured 2026-09-27 on pde_gross_pitaevskii_1d c0-naked rep1: the 900s
    agent_timeout killed the CLI mid-stream, so there is no result event. The old
    fallback returned all 101,331 chars of raw NDJSON as `result_text`, which reads as
    a non-empty answer and so blocks salvage on the turns the stream did deliver."""
    stdout = "\n".join([
        json.dumps({"type": "system", "subtype": "thinking_tokens"}),
        _ev_write(RAW_SOLVER),
        # no result event: the process was killed here
    ])
    parsed = O._parse_output(stdout)
    assert parsed["result_text"] == ""
    assert parsed["stop_subtype"] == "stream_truncated"
    assert any(O.defines_contract(t) for t in parsed["tool_payloads"])


def test_a_solver_survives_the_timeout_kill(fake_cli):
    fake_cli("\n".join([json.dumps({"type": "system", "subtype": "init"}),
                        _ev_write(RAW_SOLVER)]))
    run = O.invoke("p", "opus", "c0", ".", max_turns=2, timeout=60, salvage=True)
    assert run["salvage_used"] is True
    assert run["code_source"] == "denied_write"
    assert run["stop_subtype"] == "stream_truncated"


def test_output_that_was_never_a_stream_still_comes_back_verbatim():
    """A plain-text CLI error must not be silently turned into an empty result."""
    parsed = O._parse_output("claude: command failed, no JSON here")
    assert parsed["result_text"] == "claude: command failed, no JSON here"
    assert parsed.get("stop_subtype") is None


# --- re-running only the units that failed ----------------------------------


def _parse(argv):
    """Run the real argparse over argv and return the namespace, via --list."""
    import argparse as _ap
    holder = {}
    real = _ap.ArgumentParser.parse_args

    def _capture(self, args=None, namespace=None):
        ns = real(self, args, namespace)
        holder["ns"] = ns
        return ns

    _ap.ArgumentParser.parse_args = _capture
    try:
        with contextlib.suppress(SystemExit):
            O.main(argv)
    finally:
        _ap.ArgumentParser.parse_args = real
    return holder.get("ns")


def test_c0_max_turns_defaults_to_two():
    ns = _parse(["--list", "--no-closed-form"])
    assert ns.c0_max_turns == 2, "the default C0 must stay the one prior runs measured"
    assert ns.redo == ""


def test_c0_max_turns_and_redo_are_settable():
    ns = _parse(["--list", "--no-closed-form", "--c0-max-turns", "4",
                 "--redo", "CRASHED,RUN_ERROR"])
    assert ns.c0_max_turns == 4
    assert ns.redo == "CRASHED,RUN_ERROR"
