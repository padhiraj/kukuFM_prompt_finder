# Find the Promos

A command-line prototype for Kuku Problem 1. It scans timestamped English
microdrama scripts and exports ranked 20–60-second continuous promo cuts.
Each row gives the episode, cut boundaries, hook, cliffhanger, score, and a
reason to post it. An episode can yield zero cuts.

## Run the synthetic demo

Use Python 3.10 or newer. No model download or dependencies are needed here.

```bash
python find_promos.py --input examples/scripts --engine rules --out demo --expected-episodes 3
python -m unittest -v
```

The keyword engine is a baseline and smoke test.The bundled demo output comes from the keyword
engine. It intentionally includes a bad pick: EP03 treats a routine question
about a bus as a cliffhanger. The hook is clear, but the cut leaves no meaningful
question about the firing. The test suite exercises local HTTP requests with
fixture responses. It checks wiring and timestamps, independently of live model
quality.

## Run the full input pack

Install and start [Ollama](https://ollama.com/). Keep it on your machine when
running Kuku's screenplay pack. Install the Python dependencies only for
PDF or DOCX input.

```bash
ollama pull qwen2.5:3b
python -m pip install -r requirements.txt
python find_promos.py --input /path/to/screenplays --out private/kuku-run --expected-episodes 50
```

First run one real episode to check the timestamp parsing and the quality of
the picks. For the full run, use a fresh output directory. `--model` takes another
installed Ollama model name. `--shortlist 4` limits individual reviews per
episode. The default is one selection request plus at most four reviews per
episode with valid candidates, up to 250 calls for 50 episodes. Model call
counts, token usage, errors, and runtime are in the run report.
For slow local inference, add `--request-timeout 600` to give each request
more time. Loopback requests connect directly, including on machines with a
configured HTTP proxy. Episode errors also show their cause in the terminal.

### Use a hosted model API

For a provider with an OpenAI-compatible Chat Completions endpoint, supply
its base URL, model ID, and key. Start with the included synthetic episode.
Use an endpoint approved for the screenplay pack before running private inputs.

```bash
read -rsp 'API key: ' PROMO_API_KEY
export PROMO_API_KEY
printf '\n'
python find_promos.py --engine api \
  --api-base-url 'https://api.openai.com/v1' \
  --model 'YOUR_MODEL_ID' \
  --input examples/scripts/EP02.csv --out demo-api --expected-episodes 1
```

Use the base URL before `/chat/completions`, including the provider's version
prefix. For example, Google's Gemini compatibility base URL is
`https://generativelanguage.googleapis.com/v1beta/openai/`. Set
`--api-key-env GEMINI_API_KEY` or `--api-key-env OPENAI_API_KEY` to reuse an
existing environment variable. You can also set `PROMO_API_BASE_URL` and
`PROMO_MODEL` instead of passing those two flags.

The prompts ask for JSON. Add `--api-json-mode` if the endpoint and model
support the JSON object response format. If an endpoint uses a different
protocol, authentication mechanism, or response shape, use a provider adapter.

### Plug in a provider-specific adapter

Place a module in the project directory or on `PYTHONPATH`. Its factory
accepts `model` as a keyword argument and returns an object with
`ask(instruction, payload, audit, stage)`. The adapter sends the instruction
and payload to the provider and returns a parsed Python dictionary. For an API
that returns a JSON string, `models.parse_model_json()` can decode it. Append
optional timing and token counts to `audit` using the supplied stage
(`select` or `judge`). Raise `RuntimeError` or `ValueError` for request or
response failures. Keep authentication and any SDK setup inside the adapter.
For an implementation reference, see `ChatCompletions` in `models.py`.

```bash
python find_promos.py --engine adapter --model-adapter provider_adapter:build \
  --model 'YOUR_MODEL_ID' --input examples/scripts/EP02.csv --out demo-adapter
```

### Input formats

Use one episode per file. Paths are scanned recursively. JSON can assign an
episode ID explicitly. The other formats use the filename stem, so use unique
filenames across the input directory. Keep only episode files in that directory.

| Format | Expected input |
|---|---|
| TXT | The Kuku screenplay export is supported directly. Extract the pack into a private directory and point `--input` at the episode files. Generic TXT can use one timed record per line, e.g. `00:12 - 00:15 CLARA: Where is my sister?`. |
| CSV | Columns `start,end,text`. Values can be seconds or timecodes. Quote text containing commas. |
| JSON | An object with `episode`, optional `duration`, and an `events` array containing `start`, `end`, and `text`. A bare events array also works. |
| SRT | Standard timestamp ranges and cue text. Add key action cues if available. |
| PDF / DOCX | Text-layer PDFs and DOCX paragraphs with the same timed-line syntax as TXT. For scanned pages, tables, or a combined document, normalize to one JSON or CSV per episode. |

Use actual event start and end times when available. A missing end uses the
next distinct onset as an estimate, including onsets shared by multiple events.
For an onset-only final event, supply the episode duration in JSON or give a
known end explicitly. In Kuku's format, dialogue and action intervals protect
against cuts through a beat. Camera blocks can supply visual evidence and
cut points, but a long camera direction does not prevent a cut. Only complete
camera blocks inside a cut are shown to its reviewer. Sound effects, scene
headers, loglines, and continuity notes do not become scoring evidence.

### Output files

| File | Purpose |
|---|---|
| `promos.csv` | Global posting order with editor-facing cuts, reasons, and rubric scores. |
| `promos.json` | Structured output with ordered segments and evidence event IDs. IDs refer to numbered records after sorting by start time. |
| `final_window_baseline.json` | Final legal cut per episode, with length closest to 30 seconds at that endpoint. Use it for an editorial comparison. |
| `run_report.json` | Coverage, timing, model usage, and rejected or failed cases. Exit status is nonzero if episode counts differ or parsing/model errors leave episodes incomplete. |


