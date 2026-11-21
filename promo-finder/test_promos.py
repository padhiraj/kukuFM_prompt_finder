"""Regression checks for timestamps and the model-to-editor output contract."""
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Thread
import unittest

from find_promos import validate_judgment
from screenplay import Event, clips_for, read_episode

ROOT = Path(__file__).resolve().parent


class PromoTests(unittest.TestCase):
    def test_kuku_blocks_keep_overlap_without_treating_music_as_a_cut_boundary(self):
        content = """EPISODE NUMBER: Episode 7
--- SCRIPT START ---
SCENE 01
Scene start: 00:00.00
Scene end: 00:30.00
**CAMERA:**
Start: (00:00.00) | End: (00:30.00)
A wide shot of the room.
**SOUND FX:**
Start: (00:00.00) | End: (00:29.00)
Music plays.
**ALEX:**
Start: (00:00.20) | End: (00:05.30)
Alex reads the note.
**ACTION:**
Start: (00:20.00) | End: (00:25.00)
Someone knocks.
--- SCRIPT END ---
"""
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "episode.txt"
            path.write_text(content, encoding="utf-8")
            episode, events, estimated = read_episode(path)
        self.assertEqual((episode, estimated), ("EP007", 0))
        self.assertEqual([e.kind for e in events], ["camera", "sound", "dialogue", "action"])
        self.assertEqual(events[2].start, 200)
        self.assertTrue(any(c.start == 200 and c.end == 25000 for c in clips_for(events)))
        self.assertTrue(all(e.kind != "sound" for c in clips_for(events) for e in c.events))

    def test_overlapping_events_are_never_cut(self):
        events = (Event(1, 0, 5000, "First"), Event(2, 2000, 7000, "Overlap"),
                  Event(3, 7000, 20000, "Middle"), Event(4, 20000, 26000, "End"))
        clips = clips_for(events)
        self.assertTrue(clips)
        for clip in clips:
            self.assertIn(clip.duration, range(20_000, 60_001))
            for time in (clip.start, clip.end):
                self.assertFalse(any(e.start < time < e.end for e in events))

    def test_onset_only_events_use_next_distinct_onset(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "episode.json"
            path.write_text(json.dumps({
                "duration": "00:30.001",
                "events": [{"start": 0, "text": "Dialogue"}, {"start": 0, "text": "Action"},
                           {"start": 4, "text": "Ending"}],
            }), encoding="utf-8")
            _, events, estimated = read_episode(path)
        self.assertEqual([e.end for e in events], [4000, 4000, 30001])
        self.assertEqual(estimated, 3)

    def test_late_hook_and_early_ending_are_rejected(self):
        events = (Event(1, 0, 4000, "Opening"), Event(2, 4000, 10000, "Late hook"),
                  Event(3, 10000, 24000, "Ending"))
        clip = next(c for c in clips_for(events) if c.start == 0)
        result = {"hook_score": 3, "cliffhanger_score": 3, "clarity_score": 3,
                  "hook_event": 1, "cliffhanger_event": 3,
                  "hook": "Opening", "cliffhanger": "Ending", "why": "Example."}
        self.assertIsNotNone(validate_judgment(clip, result))
        for bad_value in ({"hook_event": 2}, {"cliffhanger_event": 2}, {"hook_event": 999}):
            with self.subTest(bad_value=bad_value), self.assertRaises(ValueError):
                validate_judgment(clip, {**result, **bad_value})

    def test_cli_runs_selection_and_isolated_judgment(self):
        requests = []

        class FixtureServer(BaseHTTPRequestHandler):
            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                payload = json.loads(request["messages"][-1]["content"])
                requests.append((self.path, payload, self.headers.get("Authorization"), request))
                if "candidates" in payload:
                    selected = next(c[0] for c in payload["candidates"] if c[1:3] == [8000, 39000])
                    response = {"clip_ids": [selected]}
                else:
                    response = {"hook_score": 4, "cliffhanger_score": 3, "clarity_score": 3,
                                "hook_event": 1, "cliffhanger_event": 6,
                                "hook": "Maya says her signature was forged.",
                                "cliffhanger": "Maya's mother walks out holding the keys.",
                                "why": "The confrontation raises a question about her mother's role."}
                message = {"content": json.dumps(response)}
                if self.path == "/v1/chat/completions":
                    envelope = {"choices": [{"message": message, "finish_reason": "stop"}],
                                "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
                else:
                    envelope = {"message": message, "prompt_eval_count": 10, "eval_count": 5}
                body = json.dumps(envelope).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_):
                pass

        with TemporaryDirectory() as tmp, HTTPServer(("127.0.0.1", 0), FixtureServer) as server:
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base_url = f"http://127.0.0.1:{server.server_port}"
                (Path(tmp) / "fixture_adapter.py").write_text(
                    "import os\nfrom models import ChatCompletions\n"
                    "def build(*, model):\n"
                    "    return ChatCompletions(model, os.environ['FIXTURE_URL'])\n",
                    encoding="utf-8",
                )
                env = {**os.environ, "TEST_PROMO_KEY": "fixture-token",
                       "http_proxy": "http://127.0.0.1:1", "no_proxy": "",
                       "PYTHONPATH": tmp, "FIXTURE_URL": f"{base_url}/v1"}
                configurations = [
                    ("ollama", ["--ollama-url", base_url, "--request-timeout", "10"], "/api/chat", None),
                    ("api", ["--api-base-url", f"{base_url}/v1", "--api-key-env", "TEST_PROMO_KEY",
                             "--api-json-mode"], "/v1/chat/completions", "Bearer fixture-token"),
                    ("adapter", ["--model-adapter", "fixture_adapter:build"], "/v1/chat/completions", None),
                ]
                for engine, options, expected_path, authorization in configurations:
                    with self.subTest(engine=engine):
                        requests.clear()
                        subprocess.run([
                            sys.executable, str(ROOT / "find_promos.py"),
                            "--input", str(ROOT / "examples/scripts/EP02.csv"),
                            "--out", tmp, "--expected-episodes", "1",
                            "--engine", engine, "--model", "fixture-model", *options,
                        ], check=True, capture_output=True, text=True, timeout=15, env=env)
                        output = json.loads((Path(tmp) / "promos.json").read_text())
                        report = (Path(tmp) / "run_report.json").read_text()
                        self.assertNotIn("fixture-token", report)
                        self.assertEqual([r[0] for r in requests], [expected_path] * 2)
                        self.assertEqual([r[2] for r in requests], [authorization] * 2)
                        self.assertEqual([r[3]["model"] for r in requests], ["fixture-model"] * 2)
                        if engine == "api":
                            self.assertEqual(requests[0][3]["response_format"], {"type": "json_object"})
                        self.assertEqual(requests[1][1]["events"][0]["start"], 0)
                        self.assertEqual(requests[1][1]["events"][-1]["id"], 6)
                        self.assertEqual(requests[1][1]["hook_candidates"], [1, 2])
                        self.assertEqual(requests[1][1]["cliffhanger_candidates"], [6])
                        if engine == "ollama":
                            selection_payload, selection_request = requests[0][1], requests[0][3]
                            selection_schema = selection_request["format"]
                            self.assertEqual(selection_schema["required"], ["clip_ids"])
                            clip_ids = selection_schema["properties"]["clip_ids"]
                            self.assertEqual(clip_ids["type"], "array")
                            self.assertEqual(clip_ids["items"]["enum"],
                                             [candidate[0] for candidate in selection_payload["candidates"]])
                            self.assertEqual(clip_ids["maxItems"], selection_payload["maximum_picks"])
                            schema = requests[1][3]["format"]["properties"]
                            self.assertEqual(schema["hook_event"]["enum"], [1, 2, None])
                            self.assertEqual(schema["cliffhanger_event"]["enum"], [6, None])
                        self.assertEqual(output[0]["hook_event"], 3)
                        self.assertEqual(output[0]["cliffhanger_event"], 8)
                        self.assertEqual(output[0]["score"], 85)
                        self.assertEqual(output[0]["segments"], [
                            {"episode": "EP02", "start": "00:00:08.000", "end": "00:00:39.000"}
                        ])
            finally:
                server.shutdown()
                thread.join()



if __name__ == "__main__":
    unittest.main()
