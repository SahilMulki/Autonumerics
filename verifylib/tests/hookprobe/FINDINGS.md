# Hook mechanism — measured behaviour

`claude 2.1.226`, macOS. Seven controlled runs. These results decide the whole
enforcement model in §3 of `docs/plan-hallucination-guardrails.md`; re-run them
after a Claude Code upgrade before trusting the plan's design.

| Question | Answer |
|---|---|
| Does `matcher` accept a **path glob**? | **No.** A `"**/target.txt"` entry never fires. `matcher` is tool-name only; filter on `tool_input.file_path` inside the hook |
| Does `PostToolUse` exit 2 **block** a write? | **No.** stderr reaches the agent, the file stays on disk, nothing is reverted |
| Does `PreToolUse` exit 2 block? | **Yes.** The file is never created. Payload has `tool_input`, no `tool_response` |
| Does a `Stop` hook exit 2 block session end? | **Yes** — and it fired 8 times, ending the run at max-turns. A `Stop` gate **requires** an attempt-counter bail-out |
| Do hooks fire inside a Task **subagent**? | **Yes**, identically |
| Do **plugin-declared** hooks load? | **Yes** — `<plugin>/hooks/hooks.json`, via `--plugin-dir` |
| Does `${CLAUDE_PLUGIN_ROOT}` expand in a hook `command`? | **Yes** → the plugin directory. `CLAUDE_PROJECT_DIR` is separately set to the user's cwd |
| Does a `Bash` heredoc bypass a `Write\|Edit` matcher? | **Yes.** `cat > f.json` fires nothing. Include `Bash` and parse the redirect target |
| Does reason-fed retry converge? | **Yes**, one iteration: REJECT → 3.2 s → PASS |
| What happens when the hook contradicts the user? | The agent **stops, leaves the bad file, and escalates**. It does not loop. In a headless run nobody answers — which is why every guard also runs out-of-band |

`hooks.example.json` is the layout that worked; `probe.sh` is the instrumented
hook that produced the table.
