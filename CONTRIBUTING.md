# Contributing to TokenWatt

Thanks for your interest! TokenWatt is a transparent, OpenAI-compatible proxy that meters the
electricity cost of local LLM inference on Apple Silicon.

## Development setup

Requires Python ≥ 3.10 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/mmmugh/tokenwatt
cd tokenwatt
./scripts/setup.sh     # activate the git hooks (the leak-scan pre-commit/pre-push guard)
uv sync --extra dev
uv run pytest          # the energy meter is mocked (FakeMeter), so no special hardware needed
```

## Git hooks

`./scripts/setup.sh` points git at the tracked hooks in `.githooks/` (via
`core.hooksPath`) and scaffolds your personal deny-list. The hooks **scan added
diff content and new file paths for secrets and PII** before they can be
committed or pushed:

- generic shapes (home paths, private LAN IPs, API-key/token formats) live in the
  tracked `.githooks/leak-patterns`;
- your own literal strings (an employer name, internal codenames) go in
  `.githooks/leak-patterns.local`, which is gitignored and never committed —
  `setup.sh` creates it from `leak-patterns.local.example`.

It's a local speed bump (clear a false positive with `git commit --no-verify`),
not a guarantee — the server-side backstop is GitHub's secret scanning + push
protection. Re-run `setup.sh` any time; it's idempotent.

## Workflow

- Branch from `main` and keep changes focused — every changed line should trace to one goal.
- Add or update tests. A good test fails when the behavior it covers breaks, not just when code is absent.
- `uv run pytest` must pass, and CI (Python 3.10 + 3.14, plus the build / `twine check` job) must be
  green before merge.
- Match the existing style; the codebase favors small, single-responsibility modules.

## Platform note

The real meter (`zeus-apple-silicon` / Apple IOReport) is **Apple Silicon only**, and it's gated by a
platform marker so `pip install` still works elsewhere. Because the test suite uses a fake meter, you
can develop and run tests on any platform — but **on-device validation** (real energy, real backends)
needs an Apple Silicon Mac.

## Reporting issues

Open an issue with: the command you ran, what you expected, what happened, your macOS + Python
versions, and — if relevant — a snippet from `~/.tokenwatt/logs/proxy.jsonl` (it logs only sizes and
counts, never prompt/response content).
