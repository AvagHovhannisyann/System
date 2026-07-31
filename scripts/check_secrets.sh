#!/usr/bin/env bash
# Pre-commit guard: block obvious credential material in staged content (I5).
# Pattern list is deliberately narrow — high-precision patterns only, so the
# hook never trains people to bypass it.
set -euo pipefail
patterns='(sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{20,}|-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----)'
files=$(git diff --cached --name-only --diff-filter=ACM)
[ -z "$files" ] && exit 0
found=0
while IFS= read -r f; do
  # binary files and this script itself are skipped
  [ "$f" = "scripts/check_secrets.sh" ] && continue
  if git show ":$f" | grep -InE "$patterns" >/dev/null 2>&1; then
    echo "BLOCKED: possible credential in staged file: $f" >&2
    git show ":$f" | grep -InE "$patterns" | head -5 | sed 's/^/  /' >&2 || true
    found=1
  fi
done <<< "$files"
exit $found
