"""Find and rank continuous promos across a folder of timestamped screenplays."""
import argparse
from collections import Counter
import csv
from dataclasses import asdict
from importlib import import_module
import json
import os
from pathlib import Path
import re
import sys
from time import perf_counter

from models import JUDGE, SELECT, ChatCompletions, Ollama
from screenplay import SUPPORTED, clips_for, clock, read_episode

WEIGHTS = {"hook_score": 40, "cliffhanger_score": 35, "clarity_score": 25}
HOOK_CUE = re.compile(
    r"\b(?:lied|stole|forged|arrested|betrayed|fired|evicted|confess(?:es|ed)?|"
    r"slaps?|collapses?|screams?|fraud|blackmail|missing|pregnant|divorce|impostor|"
    r"not your|my sister|where is|who are you)\b", re.I
)
SUSPENSE_CUE = re.compile(
    r"\?|\b(?:opens?|about to|reaches? for|starts? to|hesitates?|freezes?|wait|"
    r"don't open|before (?:he|she|they) can)\b|[—…]$", re.I
)


def validate_judgment(clip, result):
    if not isinstance(result, dict):
        raise ValueError("Expected model judgment to be a JSON object")
    for field in WEIGHTS:
        if type(result.get(field)) is not int or not 0 <= result[field] <= 4:
            raise ValueError(f"Invalid model score for {field}")
    if min(result[field] for field in WEIGHTS) < 2:
        return None
    evidence = {e.id: e for e in clip.events}
    hook_id, ending_id = result.get("hook_event"), result.get("cliffhanger_event")
    if any(type(id_) is not int or id_ not in evidence for id_ in (hook_id, ending_id)):
        raise ValueError("Model cited an event outside the clip")
    if evidence[hook_id].start - clip.start > 3000:
        raise ValueError("Model cited a hook after the first 3 seconds")
    if evidence[ending_id].end != clip.end:
        raise ValueError("Model did not cite a cliffhanger at the end of the clip")
    judgment = {field: result[field] for field in WEIGHTS}
    for field in ("hook", "cliffhanger", "why"):
        value = result.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Missing model explanation for {field}")
        judgment[field] = value.strip()
    return {**judgment, "hook_event": hook_id, "cliffhanger_event": ending_id}


def episode_promos(episode, events, clips, model, shortlist, audit, rejected):
    payload = {
        "events": [asdict(event) for event in events if event.kind != "sound"],
        "units": "milliseconds",
        "candidate_columns": ["id", "start", "end", "first_event", "last_event"],
        "candidates": [[c.id, c.start, c.end, c.events[0].id, c.events[-1].id] for c in clips],
        "maximum_picks": shortlist,
    }
    selection_response = model.ask(SELECT, payload, audit, "select")
    if not isinstance(selection_response, dict) or not isinstance(selection_response.get("clip_ids"), list):
        raise ValueError("Expected model selection to contain a clip_ids array")
    selection = selection_response["clip_ids"]
    by_id = {c.id: c for c in clips}
    seen = set()
    picks = []
    for id_ in selection:
        if type(id_) is not int or id_ not in by_id or id_ in seen:
            rejected["invalid_or_repeated_clip_id"] += 1
            continue
        seen.add(id_)
        picks.append(by_id[id_])
    if len(picks) > shortlist:
        rejected["over_shortlist_cap"] += len(picks) - shortlist

    for clip in picks[:shortlist]:
        local_events = dict(enumerate(clip.events, start=1))
        payload = {
            "duration_ms": clip.duration,
            "hook_candidates": [i for i, e in local_events.items() if e.start - clip.start <= 3000],
            "cliffhanger_candidates": [i for i, e in local_events.items() if e.end == clip.end],
            "events": [
                {**asdict(e), "id": i, "start": e.start - clip.start, "end": e.end - clip.start}
                for i, e in local_events.items()
            ],
        }
        try:
            result = model.ask(JUDGE, payload, audit, "judge")
            if isinstance(result, dict):
                result = result.copy()
                for field in ("hook_event", "cliffhanger_event"):
                    local_id = result.get(field)
                    event = local_events.get(local_id) if type(local_id) is int else None
                    result[field] = event.id if event is not None else None
            judgment = validate_judgment(clip, result)
        except (RuntimeError, ValueError) as exc:
            rejected[str(exc)] += 1
            continue
        if judgment is None:
            rejected["below_editorial_threshold"] += 1
            continue
        yield promo(episode, clip, judgment)


def rules_promos(episode, clips):
    """Offline smoke test and transparent keyword baseline."""
    for clip in clips:
        hook = next((e for e in clip.events
                     if e.start - clip.start <= 3000 and HOOK_CUE.search(e.text)), None)
        ending = next((e for e in reversed(clip.events)
                       if e.end == clip.end and SUSPENSE_CUE.search(e.text)), None)
        if hook is None or ending is None or len(clip.events) < 3:
            continue
        yield promo(episode, clip, {
            "hook_score": 3,
            "cliffhanger_score": 3 if "?" in ending.text else 2,
            "clarity_score": 2,
            "hook_event": hook.id,
            "cliffhanger_event": ending.id,
            "hook": hook.text,
            "cliffhanger": ending.text,
            "why": "The opening matches a conflict cue and the final beat matches a suspense cue.",
        })


def promo(episode, clip, judgment):
    return {
        "episode": episode,
        "segments": [{"episode": episode, "start": clock(clip.start), "end": clock(clip.end)}],
        "duration_seconds": clip.duration / 1000,
        "score": sum(judgment[field] * weight for field, weight in WEIGHTS.items()) / 4,
        **judgment,
        "_start_ms": clip.start,
        "_end_ms": clip.end,
    }


def overlap(left, right):
    if left["episode"] != right["episode"]:
        return 0
    intersection = max(0, min(left["_end_ms"], right["_end_ms"])
                       - max(left["_start_ms"], right["_start_ms"]))
    union = max(left["_end_ms"], right["_end_ms"]) - min(left["_start_ms"], right["_start_ms"])
    return intersection / union


def rank(promos):
    kept = []
    for item in sorted(promos, key=lambda p: (-p["score"], -p["hook_score"], p["episode"], p["_start_ms"])):
        if all(overlap(item, other) < 0.65 for other in kept):
            kept.append(item)
    return [
        {"rank": index, **{key: value for key, value in item.items() if not key.startswith("_")}}
        for index, item in enumerate(kept, 1)
    ]


def write_outputs(out, ranked, baselines, reports, settings):
    out.mkdir(parents=True, exist_ok=True)
    products = {
        "promos.json": ranked,
        "final_window_baseline.json": baselines,
        "run_report.json": {
            "settings": settings,
            "episode_count": len(reports),
            "failed_episode_count": sum(r["status"] == "error" for r in reports),
            "partial_episode_count": sum(r["status"] == "partial" for r in reports),
            "ranked_promo_count": len(ranked),
            "episodes": reports,
        },
    }
    for name, data in products.items():
        (out / name).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    columns = ("rank", "episode", "start", "end", "duration_seconds", "score",
               "hook_score", "cliffhanger_score", "clarity_score", "hook", "cliffhanger", "why")
    with (out / "promos.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, columns, extrasaction="ignore")
        writer.writeheader()
        for row in ranked:
            flat = {**row, **row["segments"][0]}
            # Prevent screenplay text from being interpreted as a spreadsheet formula.
            for key, value in flat.items():
                if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
                    flat[key] = "'" + value
            writer.writerow(flat)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Episode file or directory")
    parser.add_argument("--out", type=Path, required=True, help="Output directory outside input")
    parser.add_argument("--engine", choices=("ollama", "api", "adapter", "rules"), default="ollama")
    parser.add_argument("--model", default=os.getenv("PROMO_MODEL"))
    parser.add_argument("--ollama-url", default=os.getenv("PROMO_OLLAMA_URL", "http://127.0.0.1:11434"))
    parser.add_argument("--request-timeout", type=int, default=180, metavar="SECONDS",
                        help="HTTP timeout for Ollama or hosted API requests (default: 180)")
    parser.add_argument("--api-base-url", default=os.getenv("PROMO_API_BASE_URL"),
                        help="Chat Completions base URL, including any version prefix")
    parser.add_argument("--api-key-env", default="PROMO_API_KEY",
                        help="Name of the environment variable containing the API key")
    parser.add_argument("--api-json-mode", action="store_true",
                        help="Request JSON object mode when the API and model support it")
    parser.add_argument("--model-adapter", help="Import path for a trusted custom factory, module:factory")
    parser.add_argument("--shortlist", type=int, default=4, help="Maximum individual model reviews per episode")
    parser.add_argument("--expected-episodes", type=int, help="Fail the run if the episode count differs")
    args = parser.parse_args()
    if args.request_timeout < 1:
        parser.error("--request-timeout must be positive")
    if args.shortlist < 1 or (args.expected_episodes is not None and args.expected_episodes < 1):
        parser.error("Episode count and shortlist must be positive")
    if args.input.is_dir():
        source = args.input.resolve()
        if args.out.resolve().is_relative_to(source):
            parser.error("Use an output directory outside the input directory")
        files = sorted(p for p in args.input.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED)
    else:
        files = [args.input] if args.input.is_file() and args.input.suffix.lower() in SUPPORTED else []
    if not files:
        parser.error("No supported episode files found")

    model_name = args.model
    model = None
    if args.engine == "ollama":
        model_name = model_name or "qwen2.5:3b"
        model = Ollama(model_name, args.ollama_url, timeout=args.request_timeout)
    elif args.engine == "api":
        if not model_name or not args.api_base_url:
            parser.error("--engine api requires a model and API base URL")
        key = os.getenv(args.api_key_env)
        if not key and args.api_key_env != "PROMO_API_KEY":
            parser.error(f"API key environment variable {args.api_key_env!r} is empty")
        model = ChatCompletions(model_name, args.api_base_url, key, args.api_json_mode,
                                timeout=args.request_timeout)
    elif args.engine == "adapter":
        module_name, _, factory_name = (args.model_adapter or "").partition(":")
        if not model_name or not module_name or not factory_name:
            parser.error("--engine adapter requires --model and --model-adapter module:factory")
        try:
            factory = getattr(import_module(module_name), factory_name)
            model = factory(model=model_name)
        except (ImportError, AttributeError, TypeError, ValueError) as exc:
            parser.error(f"Could not load model adapter: {exc}")
        if not callable(getattr(model, "ask", None)):
            parser.error("Model adapter must provide ask(instruction, payload, audit, stage)")
    promos, baselines, reports = [], [], []
    episode_ids = set()
    for path in files:
        started = perf_counter()
        rejected, audit = Counter(), []
        label = path.relative_to(args.input).as_posix() if args.input.is_dir() else path.name
        report = {"file": label, "status": "ok", "rejections": rejected, "model_calls": audit}
        try:
            episode, events, estimated = read_episode(path)
            report.update(episode=episode, events=len(events), estimated_event_ends=estimated)
            if episode in episode_ids:
                raise ValueError(f"Duplicate episode ID: {episode}")
            episode_ids.add(episode)
            clips = clips_for(events)
            report["legal_candidates"] = len(clips)
            if clips:
                baseline = min(clips, key=lambda c: (-c.end, abs(c.duration - 30_000), c.start))
                baselines.append({"episode": episode, "start": clock(baseline.start),
                                  "end": clock(baseline.end), "duration_seconds": baseline.duration / 1000})
                results = list(episode_promos(episode, events, clips, model, args.shortlist, audit, rejected)
                               if model is not None else rules_promos(episode, clips))
                promos.extend(results)
                report["accepted_before_deduplication"] = len(results)
                if any(reason not in ("below_editorial_threshold", "over_shortlist_cap")
                       and count for reason, count in rejected.items()):
                    report["status"] = "partial"
            else:
                report["accepted_before_deduplication"] = 0
        except (OSError, ValueError, RuntimeError, ImportError) as exc:
            report.update(status="error", error=str(exc))
        report["seconds"] = round(perf_counter() - started, 3)
        reports.append(report)
        detail = f' ({report["error"]})' if "error" in report else ""
        if report["status"] == "partial":
            detail = " (" + ", ".join(f"{reason}: {count}" for reason, count in rejected.items()) + ")"
        print(f'{report["status"]:7} {label}{detail}', file=sys.stderr, flush=True)

    ranked = rank(promos)
    settings = {"engine": args.engine, "model": model_name if model is not None else None,
                "shortlist": args.shortlist if model is not None else None, "expected_episodes": args.expected_episodes,
                "weights": WEIGHTS}
    write_outputs(args.out, ranked, baselines, reports, settings)
    print(f"{len(ranked)} ranked promos from {len(files)} episode files. Outputs: {args.out}")
    if args.expected_episodes is not None and len(files) != args.expected_episodes:
        print(f"Expected {args.expected_episodes} files and found {len(files)}.", file=sys.stderr)
        return 2
    return int(any(r["status"] != "ok" for r in reports))


if __name__ == "__main__":
    raise SystemExit(main())
