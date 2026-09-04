#!/bin/bash
OUT=/private/tmp/claude-502/-Users-sahilmulki-Autonumerics/2ef32b74-4dd1-44d8-a592-06253d9b424d/scratchpad/plugtest/log.txt
IN=$(cat)
{
  echo "--- fired: $1"
  echo "    CLAUDE_PLUGIN_ROOT=[${CLAUDE_PLUGIN_ROOT:-UNSET}]"
  echo "    CLAUDE_PROJECT_DIR=[${CLAUDE_PROJECT_DIR:-UNSET}]"
  echo "    argv0=[$0]"
  echo "    event=$(python3 -c "import json,sys;print(json.loads(sys.argv[1]).get('hook_event_name'))" "$IN" 2>/dev/null)"
} >> $OUT
if [ "$1" = "stop" ]; then
  echo "STOP-HOOK: refusing to end the session -- target.json failed validation." >&2
  exit 2
fi
exit 0
