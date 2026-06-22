#!/bin/sh
# TokenWatt dev setup — activate the repo's git hooks and scaffold the local
# leak deny-list. Idempotent; safe to re-run.
set -eu

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
root=$(CDPATH= cd -- "$here/.." && pwd)
cd "$root"

# 1) Point git at the tracked hooks. They are INERT until this is set, and it is
#    local config that does NOT travel with a clone — so every clone needs it.
git config core.hooksPath .githooks
echo "ok:   core.hooksPath -> .githooks (pre-commit + pre-push leak guard active)"

# 2) Scaffold the untracked personal deny-list (employer / internal literals)
#    from the example, unless you already have one.
example=.githooks/leak-patterns.local.example
local=.githooks/leak-patterns.local
if [ -f "$local" ]; then
  echo "ok:   $local already present"
elif [ -f "$example" ]; then
  cp "$example" "$local"
  echo "new:  created $local from the example — edit in this project's literal strings"
else
  echo "warn: $example missing; create $local yourself to scan for personal literals" >&2
fi

echo "done. Hooks scan ADDED diff content for secrets/PII; bypass once with --no-verify if needed."
