"""Read timestamped screenplay events and build legal continuous clip windows."""
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import re
from decimal import Decimal, InvalidOperation

CLOCK = r"(?:\d+:){1,2}\d{2}(?:[.,]\d{1,3})?"
LINE = re.compile(
    rf"^\[?(?P<start>{CLOCK})(?:\s*(?:-->|[-–—])\s*(?P<end>{CLOCK}))?\]?"
    r"\s+(?P<text>.+)$"
)
BLOCK_TIME = re.compile(
    rf"Start:\s*\(({CLOCK})\)\s*\|\s*End:\s*\(({CLOCK})\)", re.I
)
BLOCK_LABEL = re.compile(r"\*\*(.+?)\*\*")
SUPPORTED = {".json", ".csv", ".txt", ".srt", ".pdf", ".docx"}


@dataclass(frozen=True)
class Event:
    id: int
    start: int
    end: int
    text: str
    kind: str = "beat"


@dataclass(frozen=True)
class Clip:
    id: int
    start: int
    end: int
    events: tuple[Event, ...]

    @property
    def duration(self):
        return self.end - self.start


def milliseconds(value):
    """Accept seconds or MM:SS / HH:MM:SS, with millisecond precision."""
    parts = str(value).strip().replace(",", ".").split(":")
    try:
        seconds = Decimal(parts[-1])
        if len(parts) > 3 or not seconds.is_finite() or seconds < 0:
            raise ValueError
        if len(parts) > 1 and (seconds >= 60 or any(not p.isdigit() for p in parts[:-1])):
            raise ValueError
        if len(parts) == 3 and int(parts[1]) >= 60:
            raise ValueError
        total = seconds + sum(int(p) * 60**i for i, p in enumerate(reversed(parts[:-1]), 1))
        if total * 1000 != int(total * 1000):
            raise ValueError
        return int(total * 1000)
    except (ValueError, InvalidOperation, OverflowError):
        raise ValueError(f"Invalid timestamp: {value!r}") from None


def clock(ms):
    seconds, fraction = divmod(ms, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}.{fraction:03}"


def text_rows(text):
    """Read timed lines. Subsequent lines continue the current event."""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        match = LINE.fullmatch(line)
        if match:
            rows.append(match.groupdict())
        elif re.match(r"^\[?\d+:\d", line):
            raise ValueError(f"Unrecognized timed line: {line[:70]!r}")
        elif rows:
            rows[-1]["text"] += " " + line
    return rows


def screenplay_blocks(text):
    """Read timestamped blocks in the Kuku screenplay export."""
    rows = []
    label = None
    current = None
    for line in text.splitlines():
        line = line.strip()
        if line == "--- SCRIPT END ---":
            break
        title = BLOCK_LABEL.fullmatch(line)
        if title:
            label = title[1].rstrip(":").strip()
            current = None
        elif line.lower().startswith("start:"):
            timing = BLOCK_TIME.fullmatch(line)
            if not label or not timing:
                raise ValueError(f"Malformed screenplay event: {line[:90]}")
            kind = {"CAMERA": "camera", "SOUND FX": "sound", "ACTION": "action"}.get(
                label.upper(), "dialogue"
            )
            current = {"start": timing[1], "end": timing[2], "text": "", "kind": kind}
            rows.append(current)
            if kind == "dialogue":
                current["text"] = f"{label}: "
            label = None
        elif not line or re.match(r"(?:SCENE \d+|Scene (?:heading|start|end):|Actions and dialogue:)", line, re.I):
            if line:
                current = None
        elif current is not None:
            current["text"] += line + " "
    return rows


def read_episode(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader
        text = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    elif suffix == ".docx":
        from docx import Document
        text = "\n".join(p.text for p in Document(path).paragraphs)
    else:
        text = path.read_text(encoding="utf-8-sig")

    if suffix == ".json":
        data = json.loads(text)
        episode = data.get("episode", path.stem) if isinstance(data, dict) else path.stem
        duration = data.get("duration") if isinstance(data, dict) else None
        rows = data.get("events") if isinstance(data, dict) else data
    elif suffix == ".csv":
        episode, duration, rows = path.stem, None, list(csv.DictReader(text.splitlines(keepends=True)))
    elif suffix == ".srt":
        episode, duration, rows = path.stem, None, []
        for block in re.split(r"\n\s*\n", text.strip()):
            lines = block.splitlines()
            if lines and lines[0].strip().isdigit():
                lines.pop(0)
            if len(lines) < 2 or "-->" not in lines[0]:
                raise ValueError("Malformed SRT cue")
            start, end = lines[0].split("-->", 1)
            rows.append({"start": start.strip(), "end": end.strip(), "text": " ".join(lines[1:])})
    else:
        if "--- SCRIPT START ---" in text:
            episode_header = re.search(r"^EPISODE NUMBER:\s*Episode\s+(\d+)", text, re.M | re.I)
            episode = f"EP{int(episode_header[1]):03d}" if episode_header else path.stem
            duration = None
            rows = screenplay_blocks(text.split("--- SCRIPT START ---", 1)[1])
        else:
            episode, duration, rows = path.stem, None, text_rows(text)

    if not isinstance(episode, str) or not episode.strip():
        raise ValueError("Episode ID must be nonempty text")
    if not isinstance(rows, list) or not rows:
        raise ValueError("No events found. See the input formats in README.md.")
    timed = []
    for row in rows:
        if not isinstance(row, dict) or "start" not in row:
            raise ValueError("Each event requires a start and text")
        body = row.get("text")
        if not isinstance(body, str) or not body.strip():
            raise ValueError("Each event requires nonempty text")
        end = row.get("end")
        kind = row.get("kind", "beat")
        if kind not in ("beat", "dialogue", "action", "camera", "sound"):
            raise ValueError(f"Unknown event kind: {kind!r}")
        timed.append((milliseconds(row["start"]), None if end in (None, "") else milliseconds(end),
                      body.strip(), kind))
    timed.sort(key=lambda row: row[0])
    episode_end = None if duration is None else milliseconds(duration)
    next_onset = {}
    last_onset = episode_end
    for start in sorted({row[0] for row in timed}, reverse=True):
        next_onset[start] = last_onset
        last_onset = start

    events = []
    estimated = 0
    for index, (start, end, body, kind) in enumerate(timed, 1):
        if end is None:
            end = next_onset[start]
            estimated += 1
        if end is None:
            raise ValueError("Final event needs an explicit end or a JSON episode duration")
        if end <= start or (episode_end is not None and end > episode_end):
            raise ValueError(f"Invalid event interval at {clock(start)}")
        events.append(Event(index, start, end, body, kind))
    return episode.strip(), tuple(events), estimated


def clips_for(events):
    """Sample several lengths per start and retain every reachable safe ending."""
    events = tuple(e for e in events if e.kind != "sound")
    protected = tuple(e for e in events if e.kind != "camera")
    boundaries = {time for event in events for time in (event.start, event.end)}
    safe = {time for time in boundaries
            if not any(e.start < time < e.end for e in protected)}
    starts = sorted({e.start for e in events} & safe)
    ends = sorted({e.end for e in events} & safe)
    windows = set()
    for start in starts:
        eligible = [end for end in ends if 20_000 <= end - start <= 60_000]
        if not eligible:
            continue
        for length in (20_000, 30_000, 40_000, 50_000, 60_000):
            end = min(eligible, key=lambda end: (abs(end - start - length), end))
            windows.add((start, end))
    for end in ends:
        eligible = [start for start in starts if 20_000 <= end - start <= 60_000]
        if eligible:
            start = min(eligible, key=lambda start: (abs(end - start - 30_000), start))
            windows.add((start, end))
    clips = []
    for index, (start, end) in enumerate(sorted(windows), 1):
        included = tuple(e for e in events if start <= e.start and e.end <= end)
        if included:
            clips.append(Clip(index, start, end, included))
    return clips
