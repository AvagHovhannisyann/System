#!/usr/bin/env bash
# Pre-commit guard: refuse to commit .env files (DIRECTIVE.md §7, I5).
set -euo pipefail
staged=$(git diff --cached --name-only)
bad=$(echo "$staged" | grep -E '(^|/)\.env(\..+)?$' | grep -v -E '\.env\.example$' || true)
if [ -n "$bad" ]; then
  echo "BLOCKED: attempting to commit environment file(s):" >&2
  echo "$bad" >&2
  echo "Secrets are never committed. Remove them from the index: git restore --staged <file>" >&2
  exit 1
fi
