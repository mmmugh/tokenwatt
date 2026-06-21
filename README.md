# TokenWatt

[![CI](https://github.com/mmmugh/tokenwatt/actions/workflows/ci.yml/badge.svg)](https://github.com/mmmugh/tokenwatt/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/tokenwatt)](https://pypi.org/project/tokenwatt/)
[![Python](https://img.shields.io/pypi/pyversions/tokenwatt)](https://pypi.org/project/tokenwatt/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Platform](https://img.shields.io/badge/platform-macOS_Apple_Silicon-lightgrey)

**Know what your local LLM inference actually costs you in electricity — per model, per request, on Apple Silicon. No sudo.**

TokenWatt is a transparent, OpenAI-compatible proxy. It sits in front of your local inference server, forwards every request byte-for-byte, measures the real per-rail SoC energy via Apple's IOReport (no password, ever), prices it at your utility rate, and logs a per-request, per-model electricity ledger.

```text
TokenWatt — electricity cost of local inference
  (numbers are ESTIMATED until you calibrate against a wall meter)
  last 24h :    119 req   0.0745 kWh   $0.0231

  model           type   req      kWh        $    J/tok   $/Mtok
  qwen3.6-27b     text   119   0.0745   $0.0231   5.019    0.432
```

*Real capture: an 84-minute agentic coding session (qwen3.6-27B building a Scheme interpreter,
119 streamed tool-calling turns) measured through TokenWatt on an M3 Ultra at $0.31/kWh.*

## The number people actually want

```text
$ tokenwatt compare
local electricity vs cloud LIST price for the SAME tokens (cloud snapshot 2026-06; list
price incl. their compute+margin, not just power; edit cloud.py):
  qwen3.6-27b   $0.0231 electricity — ~27.2× cheaper than gemini-2.5-flash-lite ($0.6286 vs $0.0231)
```

That session re-sent its growing context every turn — **6.07M input tokens for 53K output**, the
shape of high-context agentic coding. Locally, re-processing that context costs only electricity,
not a per-token API charge. A cloud API bills those re-sent tokens too — and even prompt caching
(which discounts re-sent context) only narrows it: local stays well ahead in this regime.

That leverage is specific to high-context agentic loops. For **balanced chat** (input ≈ output),
the cheapest cloud models are now *competitive* — gemini-2.5-flash-lite's $0.40/Mtok output is about
the same as local's $0.432, so there the win is privacy and control, not raw cost. TokenWatt shows
you which regime you're actually in — honestly, in both directions.

> Numbers are labeled **estimated (±15–30%)** until you calibrate against a wall meter, and the
> local figure is electricity only — it doesn't amortize the Mac. The point isn't one headline
> multiple; it's that you can see your real number, honestly, for *your* machine and *your* rate.

## Install

```bash
uv tool install tokenwatt        # or:  uvx tokenwatt  ·  pip install tokenwatt
```

## Use

```bash
tokenwatt init                         # scaffold a commented tokenwatt.yaml (routes + your $/kWh)
tokenwatt serve -c tokenwatt.yaml      # one port in front of your local backends; no API key, no sudo
# point your OpenAI client (Pi / OpenClaw / Claude Code / any SDK) at http://127.0.0.1:7000/v1

tokenwatt report                       # today/month $, per-model $/Mtok and J/token
tokenwatt compare                      # your electricity vs named cloud prices, for the same tokens
tokenwatt wrap                         # a shareable "my inference bill" card
```

A single `--upstream` shortcut works too, with no config file:

```bash
tokenwatt serve --upstream http://127.0.0.1:8080 --rate 0.31
```

## What it does

For each request it: forwards byte-exact to your local server (`mlx-openai-server`, `mlx-vlm`,
Ollama, LM Studio, llama.cpp…) so response bodies are byte-identical to hitting the backend
directly; brackets
the request with Apple IOReport per-rail SoC energy (sudoless); subtracts a rolling idle baseline;
books model-load (cold-start) energy to a separate row; classifies the request type (text / vision /
embedding) and counts tokens from the backend's own usage when available; prices it at your flat or
time-of-use rate; and logs a per-request row — with a `request_id` that ties each cost row to a
structured JSONL operations log.

It is a *meter*, not a gateway: no API key, no rewriting, no buffering. Streaming, tool calls,
sampling params, chat templates, and `response_format` all pass straight through.

## Honesty

- Costs read **estimated (±15–30%)** until you run the wall-meter calibration.
- A model with no rate set shows `—`, never a fabricated `$0`; a real sub-cent cost shows `<$0.0001`,
  never a fake zero.
- `$/Mtok` for text uses completion tokens **including** reasoning/`<think>` tokens; the cloud
  comparison uses total input+output cost at dated list prices.

See the design spec and milestone plans under `docs/design/`.
