#!/bin/bash
# PostToolUse: lint the edited Python file with the project's ruff config.
# Silent on success; on findings, exit 2 so Claude sees only the concise list.
# No-op when ruff isn't installed in .venv (install via `pip install -e '.[dev]'`).

FILE_PATH=$(jq -r '.tool_input.file_path // empty')
RUFF="$CLAUDE_PROJECT_DIR/.venv/bin/ruff"

[[ "$FILE_PATH" == *.py && -f "$FILE_PATH" && -x "$RUFF" ]] || exit 0

if ! OUTPUT=$("$RUFF" check --quiet --output-format concise --config "$CLAUDE_PROJECT_DIR/pyproject.toml" "$FILE_PATH" 2>&1); then
  echo "$OUTPUT" >&2
  exit 2
fi
exit 0
