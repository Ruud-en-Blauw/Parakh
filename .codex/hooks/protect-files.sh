#!/bin/bash
# Blocks Edit/Write to evidence and byte-preserved source files.
# Mirrors docs/audit/PLAN.md standing rule #6 and root AGENTS.md's
# instruction to preserve product-source/handoff evidence. Deterministic
# enforcement so this can't be skipped by an under-context agent.

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')
FILE_PATH="${FILE_PATH//\\//}"

PROTECTED_PATTERNS=(
  "/Memory.md"
  "/docs/handoffs/"
  "/benchmarks/"
  "/docs/product_sources/original/"
  "/docs/product_sources/IMPORT_AUDIT.jsonl"
)

for pattern in "${PROTECTED_PATTERNS[@]}"; do
  if [[ "$FILE_PATH" == *"$pattern"* ]]; then
    echo "Blocked: $FILE_PATH matches protected evidence pattern '$pattern'. These are append-only/byte-preserved records (see docs/audit/PLAN.md rule 6). If this edit is genuinely required, ask the user to make it directly." >&2
    exit 2
  fi
done

exit 0
