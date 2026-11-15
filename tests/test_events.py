import json

from vigia.core.events import EventLog
from vigia.core.types import Event


def test_eventlog_writes_jsonl(tmp_path):
    p = tmp_path / "out" / "events.jsonl"
    log = EventLog(p)
    log.write(Event(1.5, "table_occupied", 3, {"table": "table-1"}))
    log.write(Event(2.0, "table_freed"))
    log.close()
    rows = [json.loads(line) for line in p.read_text().splitlines()]
    assert rows[0] == {"t": 1.5, "kind": "table_occupied", "track_id": 3, "data": {"table": "table-1"}}
    assert rows[1]["kind"] == "table_freed" and rows[1]["track_id"] is None


def test_eventlog_none_is_noop():
    log = EventLog(None)
    log.write(Event(0, "x"))
    log.close()
